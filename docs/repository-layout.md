# Repository Layout

```
.claude/              skills/: workflow-control, season-data (the runbook for labelling + translating a season's chat with agents)
.memory/              working memory; see .memory/README.md
.github/workflows/    CI (data gate), auto-merge, line endings, CLAUDE.md check, branch cleanup (the last four call star-resonance/.github), actionlint (workflow PRs), Project sync and snapshot
justfile              the gates as commands
config.py             every path and hyper-parameter; INSTRUCTION (system prompt)
eval_metrics.py       shared eval scoring + report (chrF/BLEU/TER, COMET, JP/think leakage, terms); tested
genai_eval.py         per-sample eval for `mlflow.genai.evaluate`: the predictions JSONL, the per-line scores (chrF, JP leak, term, discord, ...), `build_scorers`; tested
text_rules.py         JP / Hangeul patterns shared by preprocess and the eval metrics
hf_data.py            the HF dataset: config, `hf download` command, merge of the per-channel files (each row tagged with its `channel`, `MERGE_FORMAT`), fetch state; tested (no network)
runs.py               one training = one run dir `<output_dir>/<run id>/` + run.json status; merge takes the latest complete run; tested
tracking.py           what a run records in MLflow, pure: .env.mlflow settings, params/tags/metrics from yaml, manifest, fetch state, eval report, trainer files (no mlflow import); tested
run_queue.py          the local record of every run: an append-only JSON-lines journal `.run.result.backup.jsonl` (one line per write; sent events acknowledged by a `sent` line and dropped at the next start, the run header + server id kept; the old TinyDB file is migrated once); written BEFORE anything is sent; tested
track_records.py      what each stage records (params/tags/metrics for train, merge, gguf, eval; the tracker run id `<profile>-<run id>`), pure, no mlflow; tested
stage_tracking.py     the stages' use of the tracker (open_tracker never raises; start/finish_training, resume_stage, record_stage, fail_stage); tested
compare_runs.py       the experiment's runs as one table (profile, lr, epochs, best eval loss, eval scores, data revision), sorted/filtered; pure; tested
tracker.py            sends queued runs to the NAS oldest-first, resumable, never raises into a training; `from_environment()` -> Tracker or NullTracker; tested with a fake client
judge_local.py        the same judge WITHOUT llama-server: `LocalKevClient` (a `SystemOneClient` whose `post` runs the Kev model loaded in this process; own venv `.venv-kev`, requirements-kev.txt); the loader is injected, only `load_kev` imports torch/kev (the part that is NOT tested here); tested with a stub engine
judge_prompts.py     what the Gate judge is asked, apart from the taxonomy: a variant (`configs/judge_prompts/<name>.json`: `instructions`, per-root `descriptions`, `drop`) gives the question's instructions and options; `default` = no file = the taxonomy's root descriptions as they are; every variant is another judge id (journals and probes never mix); `categorize.py --judge-prompt NAME`; pure; tested against the real taxonomy
gate_compare.py      which Gate judge is better: a probe per judge saved as data/eval/gate1-compare/<label>.json (answers keyed by line sha1, never chat text), one table on the CURRENT labelled sample (accuracy / coverage / precision at a cutoff, argmax accuracy with a Wilson interval, coverage at a target precision, speed, per pair: who is right where they differ); pure; tested
gate_pipeline.py     the logic of notebooks/gate_pipeline.ipynb: draft of the labelled sample (stratified, judge pre-labels, `?` until a human labels it), channel mix, recipe preview, decision text, does-the-model-fit-the-card; pure; tested
judge_client.py       client of llama-server's `POST /v1/systemone` (a decision model answers typed questions): `SystemOneClient.choice` / `check_server`, `JudgeError`; injected `post`, stdlib only; fixtures recorded from a real server; tested
categorizer.py        Gate 1 baseline: rules put a chat line in a root category of the taxonomy, or nothing; pure; tested
gate_judge.py         Gate 1 with a judge (llama-server decision model): the question (`choice_options` from the taxonomy, `INSTRUCTIONS`, `judge_id` = model + prompt hash), `GateJudge` (cutoff on the margin, cached answers), `cutoff_sweep`, the resumable journal (`read_journal`, `trim_torn_tail`) and `categories_from` (the categories file is derived from the journal for a cutoff); pure; tested with `tests/judge_stub.py`
gate_eval.py          scores a categorizer on a hand-labelled sample (accuracy, coverage, precision / recall per root, confusions); pure; tested
taxonomy.py           configs/category_taxonomy.json: the categories (roots, children, descriptions), `coverage`: eval lines per root; pure; tested
dataset_recipe.py     a recipe (configs/datasets/<name>.json: category -> weight) + a categories file choose the training lines, seeded and nested; pure; `preprocess.py --recipe` applies it; tested
glossary.py           configs/glossary/<season>.json (the translation glossary as data: `required` names, `banned` renderings, `fixes`) + the version of docs/translation-glossary.md; `doc_version` ties the two; pure; tested
translation_check.py  is an agent's output complete, clean (no kana/kanji left), numbers kept, terms verbatim, in line with the glossary? rows in, problems out; pure; tested
translation_assemble.py  translating with agents, minus the agents: choose the lines (guild/non-Japanese skipped by default), batches, rounds (latest wins), the glossary's fixes, term table, the brief, `require_current` (assemble REFUSES when the glossary document changed since `prepare`); pure; tested
labeling_tools.py    labelling a season with agents, minus the agents: distinct lines of the raw logs, batches + brief (docs/labeling-guide.md), the check of the agents' labels, labels.jsonl, the judge's sample (dev/test by a hash of the line; labels the taxonomy lacks -> configs/label_map.json), the counts table; pure; tested
overlap.py            is a training line also an eval line? (normalised + near-duplicate); preprocess drops them; tested
valsplit.py           which lines are validation: sha1 of the normalised line, so pairs/variants stay together and lines keep their side as data grows; tested
manifest.py           lora_train_data.meta.json: how the training file was made (style, reverse, shas, counts); update_dataset_info/train check it; tested
parameter_test.py     the logic of notebooks/parameter_test.ipynb: ParamSet, the stage commands, run_command (streams, stops the process tree on interrupt), curves from trainer_log.jsonl, sweep, decision text; tested
pipelines.py          registry: pipeline name -> ordered stage scripts (no torch; tested)
prompts.py            instruction text per model family and direction (translategemma, hy); used by preprocess + eval; tested
run_pipeline.py       --pipeline <name>: runs its stages in order, stops at the first failure; --from/--only <stage> run part of it
scripts/
  fetch_data.py         shared data: `hf download` the app's dataset_<CHANNEL>.jsonl at the revision pinned in configs/hf_dataset.yaml, merge -> raw log;
                          only when missing/changed; skipped without a repo or with RESONANCE_RAW_LOGS; `--pin` writes the latest commit; `--force`
  mlflow_compare.py     data: `python scripts/mlflow_compare.py [--profile hy] [--sort eval-loss|chrf|term] [--markdown] [--all]` prints compare_runs' table from the NAS's MLflow
  mlflow_genai_eval.py  data: `python scripts/mlflow_genai_eval.py [--model M] [--prompt ...]` evaluates `eval.py`'s saved predictions line by line in MLflow (traces + assessments, experiment `resonance-lab-eval`)
  label_lines.py        data: `python scripts/label_lines.py prepare|check|assemble|export|report --season S1 ...` the agent labelling workflow in data/labeling/<season>/ (gitignored): batches + brief, checks, labels.jsonl, judge-sample.jsonl, labels.meta.json (guide version, git sha); assemble refuses when docs/labeling-guide.md changed since prepare
  translate_agents.py   data: `python scripts/translate_agents.py prepare|check|assemble|revise|report --season S1 ...` the agent translation workflow in data/translation/<season>/ (gitignored): batches + brief per round, checks, final.jsonl + terms.tsv + final.meta.json (glossary version, git sha); agents are started by the session (skill season-data)
  compare_judges.py     data: `python scripts/compare_judges.py [--cutoff X] [--only a,b]` compares the saved probes (made with `categorize.py --probe ... --save-probe LABEL`) on the labelled sample; needs no model
  judge_check.py        data: `python scripts/judge_check.py [--url U]` is the judge server up, new enough, serving a decision model? (exit 1 with the reason)
  categorize.py         data: `python scripts/categorize.py [--judge-url URL | --judge-local [RUN]] [--judge-prompt NAME] [--cutoff X] [--use-channel]] [--probe [SAMPLE] [--save-probe LABEL]]` raw log -> data/processed/categories.jsonl (rule baseline, or the judge: resumable via data/processed/gate1_judge.jsonl; falls back to the rules with a warning when the server is unreachable), or score the baseline / the judge (+ cutoff sweep) on a labelled sample; `tests/test_judge_live.py` runs against a real server when `RESONANCE_JUDGE_URL` is set
  validate.py           shared data: raw-log sanity gate (empty/missing file, >1% damaged lines, >10% Hangeul in `original` -> ValidationError; limits in config.py)
  preprocess.py         shared data: raw {original, translated} -> {instruction, input, output};
                          `clean_reason` drops empty/untranslated, Hangeul-in-source,
                          JP-left-in-output, 10x-long, recruitment-spam, duplicate rows
                          and counts each reason; exits 1 when >30% of the usable rows are suspicious (--max-drop);
                          `--recipe` keeps only the lines a dataset recipe selects by category weight (dataset_recipe.py)
  unsloth/              pipeline `unsloth` (Qwen3 1.7B)
    split_dataset.py      data: dedup by input, shuffle (seed 42), train/val -> lora_dataset/
    train.py              model: LoRA fine-tune (unsloth), merge -> model_f16/
    fix_metadata.py       model: drop `score.weight` -> model_f16_clean/
    eval.py               model: demo lines, then the shared eval report (when data/eval/ has a dataset)
  llamafactory/         pipeline `llamafactory` (default)
    lf_tools.py           data: profiles, dataset_info.json, command lines (argument lists), run helpers
    update_dataset_info.py  data: writes data/dataset_info.json -> the processed pair file
    train.py              model: `llamafactory-cli train` into a fresh run dir (`output_dir=` override), log -> <run dir>/train_stdout.log
    merge.py              model: `llamafactory-cli export` (adapter -> full model)
    gguf.py               model: convert to F16 GGUF, quantize to q4_k_m -> model_gguf/
    eval.py               model: generate on the eval set (--prompt chat-template|training), shared report
    watch_training.py     data: live training monitor (`rich`); TrainingState parses the two logs, tested
notebooks/            parameter_test.ipynb: one parameter test / a sweep, train -> merge -> eval -> compare -> decide (Jupyter Lab from the project .venv, requirements-notebook.txt); gate_pipeline.ipynb: the dataset pipeline's Gate steps, raw log -> judge (local Kev or llama-server, `BACKEND`) -> draft sample -> probe -> cutoff -> full pass -> recipe preview (Jupyter Lab from `.venv-kev`); both committed with outputs CLEARED (tests/test_notebooks.py)
configs/llamafactory/<profile>/   train.yaml + merge.yaml per model (tests check they agree)
configs/label_map.json  labels of the labeling guide the taxonomy lacks -> the taxonomy path the judge sees (tests keep it in step with docs/labeling-guide.md); configs/glossary/     the translation glossary as data, one JSON per season (tests keep it in step with docs/translation-glossary.md); configs/judge_prompts/ judge-facing wordings of the Gate question (default, clean, clean-no-other, v2, v2-no-other); configs/datasets/     dataset recipes (category -> weight); configs/category_taxonomy.json: the categories + descriptions (tests check they agree)
deploy/mlflow/        the MLflow tracking server for the maintainer's NAS (Dockerfile, compose, basic_auth.ini, .env.example, README); tests/test_deploy_mlflow.py guards it
docs/                 versioned guides, edited per season: translation-glossary.md (official + decided Korean game terms, translation rules), labeling-guide.md (how a chat line gets its category); tests/test_docs.py pins header, changelog and taxonomy coverage
tests/                pytest for the data part; conftest.py has the JSONL fixtures; test_mock_pipeline.py runs the llamafactory stages end to end in a temp copy of the code with only the GPU tools faked (mock_gpu/: harness.py, contract.py, fake `llamafactory-cli` / `convert_hf_to_gguf.py` / `llama-quantize`, stub torch / tqdm / transformers in site/; made-up lines in fixtures/mock_pipeline/); Linux / macOS only
data/raw/ data/processed/   stage inputs/outputs (config.py paths) -- GITIGNORED
```
