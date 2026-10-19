"""Per-sample evaluation data and scores, for `mlflow.genai.evaluate` (scripts/mlflow_genai_eval.py). Pure: no mlflow import
until `build_scorers`, no network, no GPU.

`eval.py` generates the translations and saves them next to its report (`outputs/eval/<profile>-<prompt>.jsonl`, one row per eval
line: original, reference, prediction, raw_output, category); this module reads them back and turns every line into an MLflow
trace with one score per check. The checks are the ones the run-level report already has (eval_metrics.py), so a trace and the
report cannot disagree (a test compares them); chrF is the sentence-level one, not the corpus one in the report.
"""

import json
import os

from eval_metrics import discord_violation, has_jp, term_results, think_leaked

_REQUIRED = ("original", "reference", "prediction")


# --- the predictions file ----------------------------------------------------------------------------


def prediction_rows(samples, predictions, raw_outputs):
    """One row per eval sample: what the model was given, what it should say, what it said, and its raw generation."""
    if not (len(samples) == len(predictions) == len(raw_outputs)):
        raise ValueError(f"{len(samples)} samples but {len(predictions)} predictions")
    return [
        {"original": sample["original"], "reference": sample["translated"], "prediction": prediction, "raw_output": raw,
         "category": sample.get("category", "unknown")}
        for sample, prediction, raw in zip(samples, predictions, raw_outputs)
    ]


def predictions_path(directory, profile_name, prompt_mode):
    """Where eval.py leaves the predictions of one model and prompt, next to the text report of the same name."""
    return os.path.join(directory, f"{profile_name}-{prompt_mode}.jsonl")


def write_predictions(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_predictions(path):
    """The rows eval.py saved. A row needs original, reference and prediction; raw_output defaults to the prediction and
    category to 'unknown'."""
    try:
        f = open(path, encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"no predictions at {path}: run the eval stage (scripts/llamafactory/eval.py) for this model first") from None
    rows = []
    with f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as e:
                raise ValueError(f"{path} line {number}: not valid JSON ({e})") from None
            if not isinstance(row, dict) or any(key not in row for key in _REQUIRED):
                raise ValueError(f"{path} line {number}: needs {', '.join(_REQUIRED)}")
            rows.append({**row, "raw_output": row.get("raw_output", row["prediction"]), "category": row.get("category", "unknown")})
    return rows


def evaluation_data(rows):
    """The rows in the shape `mlflow.genai.evaluate(data=...)` reads: the model's input, its (precomputed) output, the reference."""
    return [
        {
            "inputs": {"text": row["original"], "category": row["category"]},
            "outputs": {"translation": row["prediction"], "raw_output": row["raw_output"]},
            "expectations": {"expected_response": row["reference"]},
        }
        for row in rows
    ]


# --- the per-sample scores ---------------------------------------------------------------------------
# Every score takes (original, prediction, reference, raw_output) and returns a number or a bool; None means "does not apply
# to this line" (MLflow leaves it out of the run mean).


def chrf_score(prediction, reference):
    """Sentence-level chrF, 0-100 (higher is better)."""
    import sacrebleu

    return sacrebleu.sentence_chrf(prediction, [reference]).score


def jp_leak(prediction):
    """Japanese left in the translation."""
    return has_jp(prediction)


def think_leak(raw_output):
    """Real reasoning inside <think> in the raw generation."""
    return think_leaked(raw_output)


def term_ok(original, prediction):
    """True when every game term in the source is used in the translation; None when the line holds no term."""
    results = term_results(original, prediction)
    return all(used for _, _, used in results) if results else None


def exact_match(prediction, reference):
    return prediction == reference


SCORERS = (
    ("chrf", lambda original, prediction, reference, raw: chrf_score(prediction, reference)),
    ("jp_leak", lambda original, prediction, reference, raw: jp_leak(prediction)),
    ("think_leak", lambda original, prediction, reference, raw: think_leak(raw)),
    ("term_ok", lambda original, prediction, reference, raw: term_ok(original, prediction)),
    ("discord_violation", lambda original, prediction, reference, raw: discord_violation(original, prediction)),
    ("exact_match", lambda original, prediction, reference, raw: exact_match(prediction, reference)),
)


def build_scorers(scorer=None):
    """SCORERS as MLflow scorers. `scorer` is mlflow's decorator factory (`scorer(name=...)`); the default imports it, so this
    module needs mlflow only here."""
    if scorer is None:
        from mlflow.genai.scorers import scorer

    def wrap(name, function):
        @scorer(name=name)
        def score(inputs, outputs, expectations):
            return function(inputs["text"], outputs["translation"], expectations["expected_response"], outputs["raw_output"])

        return score

    return [wrap(name, function) for name, function in SCORERS]
