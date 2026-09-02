#!/bin/bash
# 把本项目上传到 Bridges-2 的 Ocean 目录。在自己电脑上、项目根目录运行:
#
#     scripts/sync_to_psc.sh
#
# PSC 官方文档明确要求:文件传输必须走数据传输节点 (DTN)
# data.bridges2.psc.edu,不能从登录节点发起 —— 登录节点上连 rsync 都没装。
# 所以下面两个 host 分工不同:
#   bridges2      登录节点,只用来问路径 / 建目录 / 之后 sbatch
#   bridges2-dtn  传输节点,真正搬文件
#
# 两条连接各自复用一个 ControlMaster,所以密码最多各问一次。
#
# 只传源码:venv、各种 cache、checkpoint 和作业日志都留在原地。
# 想连 .git 一起传就 WITH_GIT=1 scripts/sync_to_psc.sh
set -euo pipefail

LOGIN="${LOGIN:-bridges2}"
DTN="${DTN:-bridges2-dtn}"

EXCLUDES=(
    .venv __pycache__ .pytest_cache .ruff_cache
    wandb wandb_logs hf_cache checkpoints
    .DS_Store 'slurm-*.out'
    # --delete would otherwise remove one-off files that only exist remotely,
    # a running job's log included. Put scratch work in scratch/
    scratch '*.log' 'upload*'
)
[ "${WITH_GIT:-0}" = "1" ] || EXCLUDES+=(.git)

# 目标路径:优先用登录节点上的 $PROJECT,拿不到就回退到手动指定
if [ -z "${DEST_ROOT:-}" ]; then
    DEST_ROOT="$(ssh "$LOGIN" 'echo "${PROJECT:-}"' | tr -d '\r')"
fi
if [ -z "$DEST_ROOT" ]; then
    echo "拿不到 \$PROJECT;请显式设置,例如" >&2
    echo "  DEST_ROOT=/ocean/projects/cis260181p/\$USER scripts/sync_to_psc.sh" >&2
    exit 1
fi
DEST="$DEST_ROOT/blackwell-ita"

echo "syncing $(pwd)"
echo "     -> $DTN:$DEST"

flags=()
for pattern in "${EXCLUDES[@]}"; do
    flags+=(--exclude "$pattern")
done

# macOS 自带的是 openrsync,没有 --mkpath,所以目标目录先在登录节点上建好
# (Ocean 是共享文件系统,登录节点和 DTN 看到的是同一份)
ssh "$LOGIN" "mkdir -p '$DEST'"

# --delete 让远端跟本地保持一致;被 --exclude 挡掉的路径不会被删,
# 所以远端的 checkpoint、cache、日志都安全
rsync -az --delete "${flags[@]}" ./ "$DTN:$DEST/"

echo
echo "done. next:"
echo "    ssh $LOGIN"
echo "    cd $DEST && scripts/psc_setup.sh"
