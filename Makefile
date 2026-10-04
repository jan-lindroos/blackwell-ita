BACKBONE ?= RLHFlow/LLaMA3-SFT-v2
SELECTOR ?= qwen3_4b_pairwise
CLAUDE_MODEL ?=

.PHONY: helpsteer-tune helpsteer-evaluate check

helpsteer-tune:
	uv run python -m blackwell_ita.helpsteer_cli tune --backbone "$(BACKBONE)" --selector "$(SELECTOR)" --judge-model "$(CLAUDE_MODEL)"

helpsteer-evaluate:
	uv run python -m blackwell_ita.helpsteer_cli evaluate --backbone "$(BACKBONE)" --selector "$(SELECTOR)" --judge-model "$(CLAUDE_MODEL)"

check:
	uv run ruff check .
	uv run marimo check notebooks/*.py
	uv run pytest
