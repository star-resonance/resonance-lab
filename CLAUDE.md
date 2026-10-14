# Agent Operating Rules — resonance-lab

resonance-lab fine-tunes the translator model of
[resonance-stream](https://github.com/enjay27/resonance-stream): Japanese Blue Protocol:
Star Resonance chat → natural Korean, as a GGUF `q4_k_m` for resonance-stream's llama.cpp
server. The base model is being re-chosen —
`.memory/roadmap/translator-shortlist-2026-10-01.md` is the plan.

**Pipelines are switchable:** `python run_pipeline.py --pipeline <name>` (registry:
`pipelines.py`). `llamafactory` (default; LLaMA-Factory SFT + LoRA, one model per profile in
`configs/llamafactory/<profile>/`, ends in a q4_k_m GGUF) and `unsloth` (Qwen3 1.7B). Each has its
own folder under `scripts/`, its own stage list, and its own requirements file / virtualenv
(`requirements-<pipeline>.txt`; the stacks pin different trl/transformers). They share
`validate.py` and `preprocess.py` (`--format instruction|pair` is the row layout each one reads).
The shipped model (TranslateGemma-4B, gist 1.1.0) was trained on `experiment/translategemma`,
whose features are being brought in — plan and what is left in `.memory/roadmap/next.md`.

The repo is two parts. **Which part you touch decides which gate applies.** That is the
most important thing on this page.

| tree | part | runs on | gate |
|---|---|---|---|
| `scripts/fetch_data.py` `scripts/validate.py` `scripts/preprocess.py` `scripts/unsloth/split_dataset.py` `scripts/llamafactory/lf_tools.py` `scripts/llamafactory/update_dataset_info.py` `scripts/llamafactory/watch_training.py` `scripts/mlflow_compare.py` `scripts/mlflow_genai_eval.py` `scripts/judge_check.py` `judge_client.py` `judge_local.py` `gate_pipeline.py` `gate_compare.py` `glossary.py` `translation_check.py` `translation_assemble.py` `scripts/translate_agents.py` `labeling_tools.py` `scripts/label_lines.py` `judge_prompts.py` `scripts/compare_judges.py` `compare_runs.py` `eval_metrics.py` `genai_eval.py` `text_rules.py` `hf_data.py` `runs.py` `tracking.py` `run_queue.py` `tracker.py` `track_records.py` `stage_tracking.py` `parameter_test.py` `overlap.py` `valsplit.py` `manifest.py` `config.py` `pipelines.py` `run_pipeline.py` `configs/` `deploy/` | **data** — raw chat logs → checked, LoRA-ready train/val JSONL | any OS, CPU | `just check` (ruff lint + pytest) |
| `scripts/unsloth/train.py` `scripts/unsloth/eval.py` `scripts/unsloth/fix_metadata.py` `scripts/llamafactory/train.py` `merge.py` `gguf.py` `eval.py`, GGUF conversion (README) | **model** — fine-tune, merge, clean, evaluate | **Windows + CUDA GPU only** (`requirements-<pipeline>.txt`) | a manual run, reported; `just mock-check` fakes the GPU tools (plumbing only) |

`just check` needs only `requirements-dev.txt` (`pip install -r requirements-dev.txt`;
`pip install rust-just` for `just`). CI (`.github/workflows/ci.yml`) runs it on Linux on
every push to `main` and every PR. The model part never runs in CI or in a cloud session:
no GPU, and torch/unsloth are not installed there. `tests/test_mock_pipeline.py` runs the
`llamafactory` stages with only the GPU tools faked: it checks the plumbing between stages, never the training.

**New pure logic goes in the data part**, where it is tested on every OS. Model-part
scripts run at import time (no `main()`); keep anything worth testing out of them.

**This repo owns the prompt.** The prompt the model is trained on is decided here, and
resonance-stream follows it (`crates/core/src/text.rs`, `translation_prompt`) — decision of
the maintainer, 2026-10-01. So a change to the prompt or chat template is made here first,
written down in `.memory/active-issues/stream-contract.md`, and then copied to
resonance-stream in its own PR there; the app must send exactly what training saw.
The training data comes from the app (`dataset_<CHANNEL>.jsonl`: `pid`, `original`,
`translated` — `null` when untranslated, `timestamp`).

---

## Where the rest is

- **Tech stack** (Python, the pipelines and their profiles, conversion): `.claude/rules/tech-stack.md`;
  loads when pipeline, config or requirements files are touched.
- **Conventions** (where paths live, stage exit codes, how scripts import `config`, adding a
  pipeline): `.claude/rules/conventions.md`; loads when Python files, tests or `pyproject.toml` are touched.
- **State** is the organization Project "Resonance" (star-resonance): read
  `git fetch origin status && git show origin/status:STATUS.md` (a snapshot 3 times a day; run
  `project-snapshot.yml` for a fresh one). No `MEMORY.md`, no handoff files; `.memory/` stays the
  detail store.
- **Repository layout** (what every file is for): `docs/repository-layout.md`; not loaded
  automatically, read it when you need to find a file.

## Guardrails

- **The plan's evidence** is an impact analysis: `g callers` from the `code-graph` skill. See
  `.claude/skills/workflow-control/SKILL.md`.
- **Changing the model is a feature, not a refactor.** `INSTRUCTION`, the chat template and a
  training hyper-parameter each change the model: own PR and an eval run. A move/split commit
  changes none of them, nor a file format.
- **No data or weights in git.** Chat logs are players' messages; they stay local.
  Test fixtures are short, made-up lines.

---

## Definition of Done

1. **Model-part changes need a manual run** on the Windows/CUDA machine (the affected
   stage, plus `eval.py` when the model changes); say in the commit body whether it was
   done, or `NOT VERIFIED: model part -- no GPU in this session`.
2. **Record state on the issue**: a `cmd:` label (`cmd:status-now`, `cmd:verify-needs-windows`,
   `cmd:verify-not-verified`, ...); a decision for Kade: assign the issue to him; a follow-up
   gets an issue. Detail goes in `.memory/` (see its README), in the task's own branch.

---

## Version Control

- `.github/workflows/auto-merge.yml` (not GitHub auto-merge) merges the PR and deletes its
  `claude/*` branch (never any other branch) once the CI workflow passes on the PR's latest
  commit. If CI fails, fix on the same branch and push again: the run for the new commit decides.
- Only `claude/*` PRs into `main` auto-merge; a draft, or a PR that changes
  `.github/workflows/`, waits for Kade. `workflow_run` workflows are read from `main`, so a
  change to `auto-merge.yml` itself takes effect after it has been merged once.
- Local sessions follow the same flow: a `claude/<task>` branch and a PR, opened with Kade's
  `gh` login (without it, print the commands for Kade).
- Only a green local gate is pushed.

### Never commit
- Chat logs and datasets (`data/**`, `lora_dataset/`), weights (`*.safetensors`,
  `*.gguf`, `model_f16*/`, `outputs/`), `llama.cpp/`, `unsloth_compiled_cache/`.
- `graft/` (regenerable), `__pycache__/`, virtualenvs, IDE folders.

## Overrides of global rules

None.
