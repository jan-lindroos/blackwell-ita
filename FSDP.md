# Training an 8B judge under FSDP: ten failures and what each one was

The reward models are 4B and fit one H100 at fp32. The judge is 8B, whose
weights, gradients and AdamW moments come to 16 bytes a parameter — 120 GiB
before a single activation, against a card's 79. Sharding those states across
four GPUs is the only way it fits, and `accelerate` with FSDP is the
lowest-friction way to shard them.

It took ten submissions to get past the first round of training. This is what
each one was, because the failure modes are not obvious from their messages and
four of the ten were the same message with three different wrong diagnoses.

## The record

| # | Job | Elapsed | Message |
|---|---|---|---|
| 1 | 44863909 | 12:31 | `ConnectionError: HTTP status 401 Unauthorized` |
| 2 | 44893676 | 26:14 | `RuntimeError: Expected all tensors to be on the same device` |
| 3 | 44900713 | 5:50 | `RuntimeError: 'weight' must be 2-D` |
| 4 | 44903276 | 6:46 | `RuntimeError: 'weight' must be 2-D` |
| 5 | 44913337 | 6:20 | `RuntimeError: 'weight' must be 2-D` |
| 6 | 44918592 | 6:13 | `RuntimeError: 'weight' must be 2-D` |
| 7 | 44970079 | 8:37 | `DistBackendError: NCCL error ... NCCLUtils.cpp:92` |
| 8 | 44986239 | 9:18 | `FileNotFoundError: .../pytorch_model_fsdp_0/.metadata` |
| 9 | 45005451 | 12:28 | `RuntimeError: Missing key in checkpoint state_dict: optimizer.param_groups.` |
| 10 | 45016778 | 13:24 | `RuntimeError: basic_ios::clear: iostream error` |

## 1 — the batch script is snapshotted at submit

A 401 from `huggingface.co/api/models/meta-llama/.../xet-read-token`. The
gated Llama repo needs a token, and the token was there: `hf auth whoami`
answered, and `AutoConfig.from_pretrained` on the same repo succeeded from the
login node.

Two things were true at once. Config files are small and come from the regular
CDN; **weight files go through Xet storage**, which authenticates separately —
so testing with `AutoConfig` proves nothing about a weight download. And the
job was submitted at 19:56:05 while the `export HF_TOKEN` line reached the
sbatch script at 20:00:07, four minutes later.

**Slurm copies the batch script when you submit.** Editing it afterwards does
not reach a queued job. Python files are different: they are read at runtime,
so a fix that lands while a job waits in the queue does apply. Two mechanisms,
one habit needed — resubmit after touching an sbatch.

## 2 — a patch applied to one of two parallel files

`Expected all tensors to be on the same device, but got index is on cuda:0,
different from other tensors on cpu`, in `self.embed_tokens(input_ids)`. The
embedding weight was on the CPU while the input ids were on the GPU.

The cause was an edit applied asymmetrically. `train_bt_reward.py` and
`train_pairwise_preference.py` are deliberate mirrors of each other, and a
batch edit had added `accelerator.prepare(model)` to the first and only the
tokenizer change to the second. **The pairwise script never prepared its model
at all** — so it was never sharded, never moved to the device, and the
embedding was the first operation to notice.

There is a second, real issue underneath, which would have surfaced next:
`TRANSFORMER_BASED_WRAP` wraps only the decoder layers, so the embedding, the
final norm and the scoring head sit outside every wrapped unit and FSDP leaves
them where they are. Moving the model to `accelerator.device` before
`prepare` handles both.

## 3, 4, 5, 6 — four attempts at a configuration that was already correct

`'weight' must be 2-D`, raised inside `F.embedding`. FSDP flattens parameters,
and `use_orig_params=True` is what keeps them as their own tensors, so the
message reads like that setting is off.

It was not off. Each attempt tightened the configuration and changed nothing:

- **3** moved the FSDP settings from `accelerate launch` flags into
  `scripts/fsdp_config.yaml`, on the theory that `--fsdp_use_orig_params true`
  was being parsed away.
- **4** pinned `fsdp_transformer_layer_cls_to_wrap`, because left unset the
  launcher passes the literal string `"None"` as the class to wrap, nothing
  matches, and only the root module gets wrapped.
- **5** moved that pin from an environment variable into the YAML, because the
  variable had not actually been set on the submission.
- **6** ran with all of it in place and failed identically.

By then the configuration had been verified three ways: the YAML parsed with
`fsdp_use_orig_params: True`, `FullyShardedDataParallelPlugin` constructed
from launch-style environment variables reported `use_orig_params True`, and
the wrap class was named explicitly. **Every layer was right and nothing
changed** — which is the signal that the cause is somewhere else entirely, and
is exactly the signal four attempts ignored.

### The actual cause

Neither model class defines `forward`.

```python
class PairwisePreferenceModel(torch.nn.Module):
    def score(self, texts, device): ...
    def batch_logits(self, batch, device): ...
    def compute_loss(self, batch, device): ...
```

The training loop called `model.compute_loss(batch, device)`, and the metrics
called `model.batch_logits(batch, device)`. **FSDP installs its all-gather
hooks on `forward` and nowhere else.** Calling any other method on the wrapper
goes through `FSDP.__getattr__`, which delegates to the bound method of the
*unwrapped* module — so `self` inside is the original module, its parameters
are still one-dimensional shards, and no gather ever happens. `F.embedding` is
simply the first operation to look at a weight and object.

This explains every observation the configuration theories could not: the
failure came on the very first forward, on every rank, at every batch size,
under every wrap policy, with `use_orig_params` demonstrably enabled.

The fix is a single entry point:

```python
def forward(self, batch, device, *, as_loss: bool = False):
    logits = self.batch_logits(batch, device)
    if not as_loss:
        return logits
    return masked_binary_cross_entropy(logits, batch, device)
```

A flag rather than two methods, because both the training loss and the metrics
need a forward and only one of them can be hooked. All three call sites go
through `__call__`.

Five test doubles defined `compute_loss` or `batch_logits` and no `forward`,
so the suite failed immediately with *"Module is missing the required forward
function"* — the contract change surfaced itself, which is the tests earning
their keep.

## 7 — the cluster's NVLink SHARP

`ncclUnhandledCudaError`, and underneath it a message that names its own fix:

```
Failed to bind NVLink SHARP (NVLS) Multicast memory of size 2097152 :
CUDA error 1 'invalid argument'.
This is usually caused by a system or configuration error in the Fabric
Manager or NVSwitches.
Disable NVLS (NCCL_NVLS_ENABLE=0) if you wish to avoid this error.
```

Nothing to do with the code. `export NCCL_NVLS_ENABLE=0` in the sbatch;
collectives fall back to the ring algorithm, which at four GPUs costs little.

The forward fix had worked: the error had moved from the first forward to
process initialisation.

## 8 — every rank named its own snapshot directory

`FileNotFoundError: /var/tmp/tmpamn8_vql/pytorch_model_fsdp_0/.metadata`, and
in another rank's traceback `/var/tmp/tmpvlj0rtv2/...`. Two different paths.

A sharded model's `state_dict` is not portable between ranks, so the
best-weights snapshot goes through `accelerator.save_state` instead of memory
— and the directory came from `tempfile.mkdtemp()`, which each rank called for
itself. `save_state` is collective: all four ranks write one shard each into
**the same** directory. Four directories meant four scattered shards and no
complete set anywhere.

The directory has to be derived from something every rank agrees on. It is now
`args.output_dir / ".fsdp-best"` — all ranks parse the same arguments, so all
ranks name the same path, and it lives on the shared filesystem rather than a
node-local `/var/tmp`.

Training reached `step 10: validation_loss=0.7579` before this, so the loop,
the sharded forward and backward, and the gradient reduction were all correct
by now.

## 9 — sharded optimizer state, and the wrong fix for it

`Missing key in checkpoint state_dict: optimizer.param_groups.` — raised on
load, saying the checkpoint did not contain optimizer state that the load
expected. Under `SHARDED_STATE_DICT` the optimizer round-trips through
`FSDP.optim_state_dict`, and a mismatch between the save and load contexts
loses it.

Switching to `FULL_STATE_DICT` fixed the read. It was still the wrong fix: it
addressed the symptom and tripled what a snapshot writes.

## 10 — out of disk, caused by fixing 9 that way

`torch.save` failed with `basic_ios::clear: iostream error` and
`[enforce fail at inline_container.cc:672] unexpected pos 44288 vs 44182` — a
half-written zip, which is what a full filesystem looks like from inside
serialization.

`checkpoints/judge-smoke/.fsdp-best` held **168 GB**. Under
`FULL_STATE_DICT`, `save_state` gathers and writes the model *and* the
optimizer, and a fp32 AdamW's two moments are twice the weights:

```
weights        8.03B x 4 = 32 GB
optimizer      8.03B x 8 = 64 GB
per snapshot             = 96 GB
```

The `FULL_STATE_DICT` of attempt 9 had turned a read problem into a write
problem three times its size.

### The fix that should have been attempt 9

**Do not hand the optimizer to `accelerator.prepare`.** Preparing it registers
it for checkpointing, and nothing in this loop ever reads optimizer state
back: the snapshot exists to restore the best weights when training ends, not
to resume. With `use_orig_params` the parameters keep their own identities, so
a plain `AdamW` steps the local shards correctly.

```python
train_loader = accelerator.prepare(train_loader)   # not the optimizer
```

96 GB becomes 32 GB, and the missing-key problem of attempt 9 disappears with
it, because state that is never saved is never missed. The snapshot directory
is also removed once `restore_best` has read it.

## What the ten attempts are worth

**A message can name a setting and mean a call path.** `'weight' must be 2-D`
points at parameter flattening, which points at `use_orig_params`. The cause
was that FSDP's hooks were never reached. Four submissions went into the
configuration because the message pointed there.

**Verified-correct configuration that changes nothing is evidence, not
reassurance.** By attempt 6 the settings had been confirmed at three layers
and the failure was byte-identical. That combination should have redirected
the search much earlier than it did.

**Single-process local runs give false confidence.** `accelerate launch --cpu
--num_processes 1` exercises the API surface and none of the sharding. It
passed before attempts 8, 9 and 10, each of which failed in the sharded
checkpoint path. It is worth running — it caught a dtype mismatch and the
missing-`forward` contract — but "it works locally" says nothing about
whether it works sharded.

**Mirrored files need mirrored edits.** The two training scripts are
deliberate near-duplicates, which makes a one-sided patch invisible on
inspection and fatal at runtime. `grep -c` on both after every shared change
is cheap.

**Fixes have footprints.** `FULL_STATE_DICT` was a real fix for a real
problem and it filled a filesystem two attempts later. The question to ask of
a fix is not only whether it works but what it now costs.

**Slurm snapshots the batch script; Python is read at runtime.** Editing an
sbatch under a queued job does nothing. Editing a `.py` under a queued job
applies — and under a *running* job leaves you unable to say which version ran,
which is its own reason not to.
