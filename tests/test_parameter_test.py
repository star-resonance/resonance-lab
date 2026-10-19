import ast
import json
import os
import subprocess
import sys
import time

import pytest

import eval_metrics
import parameter_test as pt
import pipelines
from config import BASE_DIR
from runs import finish_run, start_run


def script(*parts):
    return os.path.join(BASE_DIR, *parts)


# --- the parameter set ---------------------------------------------------------------------------------------------

def test_profile_is_the_models_fast_profile():
    assert pt.ParamSet(model="hy-mt2-1.8b").profile == "hy-mt2-1.8b-fast"
    assert pt.ParamSet(model="hy-mt2-1.8b", fast=False).profile == "hy-mt2-1.8b"


def test_label_names_only_what_the_set_overrides():
    assert pt.ParamSet().label() == "profile defaults"
    assert pt.ParamSet(lr=4e-4, epochs=6).label() == "lr=0.0004 epochs=6"
    assert pt.ParamSet(lr=1e-5).label() == "lr=1e-05"


@pytest.mark.parametrize("kwargs", [{"lr": 0}, {"lr": -1e-4}, {"epochs": float("nan")}, {"prompts": ()}, {"prompts": ("nope",)}])
def test_a_bad_parameter_set_is_refused(kwargs):
    with pytest.raises(ValueError):
        pt.ParamSet(**kwargs).validate()


def test_a_good_parameter_set_validates():
    pt.ParamSet(lr=2e-4, epochs=6, prompts=("training", "chat-template")).validate()


def test_sweep_params_layer_each_entry_on_the_defaults():
    defaults = pt.ParamSet(model="hy-mt2-1.8b", epochs=3)
    sets = pt.sweep_params(defaults, [{"lr": 4e-4}, {"lr": 2e-4, "epochs": 6}, {"lr": 1e-4, "prompts": ["training", "chat-template"]}])
    assert [s.lr for s in sets] == [4e-4, 2e-4, 1e-4]
    assert [s.epochs for s in sets] == [3, 6, 3]
    assert all(s.model == "hy-mt2-1.8b" for s in sets)
    assert sets[2].prompts == ("training", "chat-template")


def test_sweep_params_refuse_unknown_keys_and_bad_values():
    with pytest.raises(ValueError, match="learning_rate"):
        pt.sweep_params(pt.ParamSet(), [{"learning_rate": 1e-4}])
    with pytest.raises(ValueError):
        pt.sweep_params(pt.ParamSet(), [{"lr": -1}])


# --- the commands --------------------------------------------------------------------------------------------------

def test_the_lifecycle_runs_train_merge_then_one_eval_per_prompt():
    params = pt.ParamSet(model="hy-mt2-1.8b", lr=4e-4, epochs=6, prompts=("training", "chat-template"))
    commands = pt.lifecycle(params, python="PY")
    assert [c.name for c in commands] == ["Fine-Tuning", "Merge LoRA", "Evaluation (training)", "Evaluation (chat-template)"]
    model = ["--model", "hy-mt2-1.8b", "--fast"]
    assert commands[0].argv == ["PY", script("scripts", "llamafactory", "train.py"), *model, "--lr", "0.0004", "--epochs", "6.0"]
    assert commands[1].argv == ["PY", script("scripts", "llamafactory", "merge.py"), *model]
    assert commands[2].argv == ["PY", script("scripts", "llamafactory", "eval.py"), *model, "--prompt", "training"]
    assert commands[3].argv[-2:] == ["--prompt", "chat-template"]


def test_no_override_means_no_lr_or_epochs_argument():
    train = pt.lifecycle(pt.ParamSet(), python="PY")[0].argv
    assert "--lr" not in train and "--epochs" not in train
    assert "--model" not in train and "--fast" in train


def test_a_full_profile_run_leaves_out_fast():
    train = pt.lifecycle(pt.ParamSet(model="hy-mt2-1.8b", fast=False), python="PY")[0].argv
    assert "--fast" not in train and train[-2:] == ["--model", "hy-mt2-1.8b"]


def test_refreshing_the_data_puts_its_stages_first_each_through_run_pipeline():
    commands = pt.lifecycle(pt.ParamSet(model="hy-mt2-1.8b"), refresh_data=True, python="PY")
    data, rest = commands[:4], commands[4:]
    assert [c.name for c in data] == pt.DATA_STAGES
    assert [c.name for c in rest][:2] == ["Fine-Tuning", "Merge LoRA"]
    for command, stage in zip(data, pt.DATA_STAGES, strict=True):
        assert command.argv == ["PY", script("run_pipeline.py"), "--model", "hy-mt2-1.8b", "--fast", "--only", stage]


def test_a_recipe_rebuilds_the_training_file_with_it_before_training():
    commands = pt.lifecycle(pt.ParamSet(model="hy-mt2-1.8b", recipe="balanced-v1"), python="PY")
    assert [c.name for c in commands][:3] == ["Preprocessing", "Update Dataset", "Fine-Tuning"]
    for command in commands[:2]:
        assert command.argv == ["PY", script("run_pipeline.py"), "--model", "hy-mt2-1.8b", "--fast", "--only", command.name,
                                "--recipe", "balanced-v1"]


def test_a_recipe_with_refreshed_data_runs_every_data_stage_once_with_the_recipe():
    commands = pt.lifecycle(pt.ParamSet(recipe="balanced-v1"), refresh_data=True, python="PY")
    assert [c.name for c in commands][:5] == [*pt.DATA_STAGES, "Fine-Tuning"]
    assert all(c.argv[-2:] == ["--recipe", "balanced-v1"] for c in commands[:4])


def test_the_label_names_the_recipe():
    assert pt.ParamSet(lr=4e-4, recipe="balanced-v1").label() == "lr=0.0004 recipe=balanced-v1"
    assert pt.ParamSet(recipe="balanced-v1").label() == "recipe=balanced-v1"


def test_a_recipe_that_is_not_in_the_repo_is_refused_before_anything_runs():
    with pytest.raises(ValueError, match="no-such-mix"):
        pt.ParamSet(recipe="no-such-mix").validate()
    with pytest.raises(ValueError, match="not a recipe name"):
        pt.ParamSet(recipe="../balanced-v1").validate()


def test_a_sweep_over_recipes_gives_every_set_a_recipe():
    sets = pt.sweep_params(pt.ParamSet(), [{"recipe": "balanced-v1"}, {"recipe": "balanced-v1-detailed"}])
    assert [s.recipe for s in sets] == ["balanced-v1", "balanced-v1-detailed"]
    # a set without one would train on whatever file the set before it left: refused
    with pytest.raises(ValueError, match="recipe"):
        pt.sweep_params(pt.ParamSet(), [{"recipe": "balanced-v1"}, {"lr": 4e-4}])
    assert [s.recipe for s in pt.sweep_params(pt.ParamSet(recipe="balanced-v1"), [{"lr": 4e-4}, {"lr": 8e-4}])] == ["balanced-v1"] * 2


def test_data_stages_are_the_first_four_of_the_default_pipeline():
    assert pt.DATA_STAGES == [s.name for s in pipelines.stages(pipelines.DEFAULT)[:4]]
    # run_pipeline's --only takes a name or a unique prefix, so each one must select exactly its own stage
    for name in pt.DATA_STAGES:
        assert [s.name for s in pipelines.select_stages(pipelines.stages(pipelines.DEFAULT), only=name)] == [name]


def test_the_eval_prompts_are_the_ones_eval_py_offers():
    with open(script("scripts", "llamafactory", "eval.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    prompts = next(node.value for node in ast.walk(tree)
                   if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "PROMPTS" for t in node.targets))
    assert set(pt.EVAL_PROMPTS) == {key.value for key in prompts.keys}


def test_every_script_the_commands_run_exists():
    commands = pt.lifecycle(pt.ParamSet(prompts=("training",)), refresh_data=True, python="PY")
    for command in commands:
        assert os.path.isfile(command.argv[1]), command


# --- running a command ---------------------------------------------------------------------------------------------

def py(code):
    return pt.Command("demo", [sys.executable, "-c", code])


def test_run_command_streams_each_line_and_reports_the_time():
    lines = []
    seconds = pt.run_command(py("print('one'); print('two')"), emit=lines.append)
    assert lines == ["one", "two"]
    assert seconds >= 0


def test_a_failing_command_raises_with_the_stage_and_the_tail_of_its_output():
    code = "import sys\nfor i in range(100): print('line', i)\nsys.exit(3)"
    with pytest.raises(pt.StageFailed) as caught:
        pt.run_command(py(code), emit=lambda line: None, tail_lines=5)
    error = caught.value
    assert error.stage == "demo" and error.returncode == 3
    assert error.tail == [f"line {i}" for i in range(95, 100)]
    assert "demo" in str(error) and "line 99" in str(error)


def test_a_program_that_is_not_installed_is_a_stage_failure_not_a_traceback():
    with pytest.raises(pt.StageFailed, match="not found"):
        pt.run_command(pt.Command("Fine-Tuning", ["definitely-not-a-program-xyz"]), emit=lambda line: None)


def alive(pid):
    """Running (a zombie nobody reaped, as in a container, does not count)."""
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not os.path.isdir("/proc"), reason="the posix path: SIGINT to the process group, checked through /proc")
def test_an_interrupt_stops_the_child_and_its_children(tmp_path):
    pid_file = tmp_path / "grandchild.pid"
    code = (
        "import subprocess, sys, time\n"
        f"p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(p.pid))\n"
        "print('started', flush=True)\n"
        "time.sleep(60)\n"
    )

    def interrupt(line):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        pt.run_command(py(code), emit=interrupt, grace_seconds=5)
    grandchild = int(pid_file.read_text())
    for _ in range(50):
        if not alive(grandchild):
            break
        time.sleep(0.1)
    assert not alive(grandchild)


class FakeProc:
    """A child process for terminate_tree: it dies when `die()` is called (by the fake taskkill or by `kill`)."""

    pid = 4242

    def __init__(self, survives_kill=False):
        self.dead, self.survives_kill, self.killed = False, survives_kill, False

    def die(self):
        self.dead = True

    def poll(self):
        return 0 if self.dead else None

    def kill(self):
        self.killed = True
        if not self.survives_kill:
            self.die()

    def wait(self, timeout=None):
        if not self.dead:
            raise subprocess.TimeoutExpired("fake", timeout)
        return 0


def fake_taskkill(proc, returncode=0, stderr="", kills=True):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if kills:
            proc.die()
        return subprocess.CompletedProcess(argv, returncode, stdout="", stderr=stderr)

    run.calls = calls
    return run


def test_on_windows_the_whole_tree_is_killed_with_taskkill():
    proc = FakeProc()
    run = fake_taskkill(proc)
    pt.terminate_tree(proc, grace_seconds=0.01, platform="win32", run=run)
    assert run.calls == [["taskkill", "/PID", "4242", "/T", "/F"]]
    assert not proc.killed


def test_a_taskkill_that_fails_is_reported_not_swallowed():
    """The maintainer's interrupted sweep left a training running to the end: the failure must be visible."""
    proc = FakeProc()
    run = fake_taskkill(proc, returncode=1, stderr="ERROR: The process with PID 4242 could not be terminated. Access is denied.", kills=False)
    with pytest.raises(pt.StopFailed, match="Access is denied") as caught:
        pt.terminate_tree(proc, grace_seconds=0.01, platform="win32", run=run)
    assert "4242" in str(caught.value) and "child processes may still be running" in str(caught.value)
    assert proc.killed  # the fallback: at least the process itself


def test_a_process_that_is_still_alive_after_the_kill_is_reported():
    proc = FakeProc(survives_kill=True)
    run = fake_taskkill(proc, kills=False)
    with pytest.raises(pt.StopFailed, match="still running") as caught:
        pt.terminate_tree(proc, grace_seconds=0.01, platform="win32", run=run)
    assert "4242" in str(caught.value)


def test_a_process_that_already_ended_is_left_alone():
    proc = FakeProc()
    proc.die()
    run = fake_taskkill(proc)
    pt.terminate_tree(proc, platform="win32", run=run)
    assert run.calls == []


# --- the curves ----------------------------------------------------------------------------------------------------

def write_log(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write((row if isinstance(row, str) else json.dumps(row)) + "\n")


def test_read_curves_splits_train_and_eval_rows_and_skips_noise(tmp_path):
    log = tmp_path / "trainer_log.jsonl"
    write_log(log, [
        {"current_steps": 1, "total_steps": 80, "loss": 2.0, "epoch": 0.1},
        "not json",
        {"current_steps": 10, "eval_loss": 1.5, "epoch": 1.0},
        {"current_steps": 11, "loss": 1.4, "epoch": 1.1},
        {"current_steps": 20, "eval_loss": 1.2, "epoch": 2.0},
        {"current_steps": 30, "eval_loss": 1.3, "epoch": 3.0},
    ])
    curves = pt.read_curves(str(log))
    assert curves.train == [(1, 2.0), (11, 1.4)]
    assert curves.evals == [(10, 1.5), (20, 1.2), (30, 1.3)]
    assert curves.best_eval == (20, 1.2)
    assert not curves.still_falling  # the best is not the last evaluation: it turned up


def test_still_falling_means_the_best_eval_is_the_last_one(tmp_path):
    log = tmp_path / "trainer_log.jsonl"
    write_log(log, [{"current_steps": 10, "eval_loss": 1.5}, {"current_steps": 20, "eval_loss": 1.2}])
    assert pt.read_curves(str(log)).still_falling


def test_a_missing_log_is_empty_curves(tmp_path):
    curves = pt.read_curves(str(tmp_path / "nope.jsonl"))
    assert curves.train == [] and curves.evals == [] and curves.best_eval is None and not curves.still_falling


def test_run_curves_reads_every_run_directory_oldest_first(tmp_path):
    first = start_run(str(tmp_path), "p", now=None)
    write_log(os.path.join(first.dir, "trainer_log.jsonl"), [{"current_steps": 10, "eval_loss": 1.0}])
    second = start_run(str(tmp_path), "p")
    write_log(os.path.join(second.dir, "trainer_log.jsonl"), [{"current_steps": 10, "eval_loss": 0.5}])
    os.mkdir(tmp_path / "not-a-run")  # no run.json: ignored
    curves = pt.run_curves(str(tmp_path))
    assert list(curves) == sorted([first.id, second.id])
    assert curves[second.id].best_eval == (10, 0.5)
    assert pt.run_curves(str(tmp_path / "missing")) == {}


# --- one lifecycle -------------------------------------------------------------------------------------------------

def fake_run(adapter_dir, eval_losses, log):
    """A stand-in for run_command: the train command makes a run dir with a trainer log, the others do nothing."""
    def run(command, emit=print):
        log.append(command.name)
        if command.name == "Fine-Tuning":
            made = start_run(str(adapter_dir), "hy-mt2-1.8b-fast")
            write_log(os.path.join(made.dir, "trainer_log.jsonl"),
                      [{"current_steps": 5, "loss": 1.8}] + [{"current_steps": 10 * (i + 1), "eval_loss": v} for i, v in enumerate(eval_losses)])
            finish_run(made.dir, "complete")
        return 1.5
    return run


def test_run_lifecycle_runs_every_stage_and_summarises_the_new_run(tmp_path):
    log = []
    params = pt.ParamSet(model="hy-mt2-1.8b", lr=4e-4, prompts=("training", "chat-template"))
    result = pt.run_lifecycle(params, run=fake_run(tmp_path, [1.4, 0.9, 0.8], log), emit=lambda line: None, adapter_dir=str(tmp_path))
    assert log == ["Fine-Tuning", "Merge LoRA", "Evaluation (training)", "Evaluation (chat-template)"]
    assert result.status == "done" and result.params is params
    assert result.best_eval_loss == 0.8 and result.best_step == 30 and result.still_falling
    assert result.train_loss == 1.8
    assert result.run_id and os.path.isdir(os.path.join(tmp_path, result.run_id))
    assert result.seconds == pytest.approx(6.0)  # 4 commands x 1.5 s


def test_run_lifecycle_lets_a_stage_failure_stop_it(tmp_path):
    seen = []

    def run(command, emit=print):
        seen.append(command.name)
        raise pt.StageFailed(command.name, 1, ["boom"])

    with pytest.raises(pt.StageFailed):
        pt.run_lifecycle(pt.ParamSet(), run=run, emit=lambda line: None, adapter_dir=str(tmp_path))
    assert seen == ["Fine-Tuning"]  # merge and eval never ran


# --- the eval scores of a run --------------------------------------------------------------------------------------

def real_report(predictions):
    samples = [{"original": "消化", "translated": "숙제한다"}, {"original": "消化", "translated": "숙제한다"}]
    return eval_metrics.format_report(eval_metrics.evaluate(samples, predictions), samples, predictions, comet=None)


def test_the_scores_are_read_from_the_real_report_format():
    text = real_report(["숙제한다", "소화한다"])
    scores = pt.parse_report(text)
    assert scores["chrf"] == pytest.approx(float(next(line for line in text.splitlines() if line.startswith("chrF")).split(":")[1].split()[0]))
    assert (scores["term_hits"], scores["term_total"], scores["term_pct"]) == (1, 2, 50.0)


def test_a_report_without_terms_has_no_term_accuracy():
    samples = [{"original": "おやすみ", "translated": "잘 자"}]
    text = eval_metrics.format_report(eval_metrics.evaluate(samples, ["잘 자"]), samples, ["잘 자"])
    scores = pt.parse_report(text)
    assert scores["chrf"] is not None and scores["term_pct"] is None


def write_report(eval_dir, profile, prompt, text, mtime):
    path = eval_dir / f"{profile}-{prompt}.txt"
    path.write_text(text, encoding="utf-8")
    os.utime(path, (mtime, mtime))


def test_summary_reads_the_reports_written_after_the_training(tmp_path):
    adapters, eval_dir = tmp_path / "adapters", tmp_path / "eval"
    eval_dir.mkdir()
    made = start_run(str(adapters), "hy-mt2-1.8b-fast")
    log = os.path.join(made.dir, "trainer_log.jsonl")
    write_log(log, [{"current_steps": 5, "loss": 1.8}, {"current_steps": 10, "eval_loss": 0.9}])
    finish_run(made.dir, "complete")
    os.utime(log, (1000, 1000))
    write_report(eval_dir, "hy-mt2-1.8b-fast", "training", real_report(["숙제한다", "소화한다"]), mtime=2000)
    write_report(eval_dir, "hy-mt2-1.8b-fast", "chat-template", real_report(["숙제한다", "숙제한다"]), mtime=500)  # an older run's
    params = pt.ParamSet(model="hy-mt2-1.8b", prompts=("training", "chat-template"))
    result = pt.summarize_latest_run(params, str(adapters), eval_dir=str(eval_dir))
    assert list(result.scores) == ["training"]  # the stale report of the other prompt is not this run's
    assert result.scores["training"]["term_pct"] == 50.0


def test_a_summary_without_reports_has_no_scores(tmp_path):
    made = start_run(str(tmp_path), "hy-mt2-1.8b-fast")
    write_log(os.path.join(made.dir, "trainer_log.jsonl"), [{"current_steps": 5, "loss": 1.8}])
    assert pt.summarize_latest_run(pt.ParamSet(), str(tmp_path), eval_dir=str(tmp_path / "none")).scores == {}


# --- the sweep -----------------------------------------------------------------------------------------------------

def done(lr, eval_loss, step=80, falling=False, run_id="r"):
    return pt.Result(pt.ParamSet(lr=lr), "done", run_id, eval_loss, step, 1.2, 240.0, still_falling=falling)


def test_run_sweep_runs_each_set_in_order():
    sets = [pt.ParamSet(lr=1e-4), pt.ParamSet(lr=2e-4)]
    ran = []

    def execute(params):
        ran.append(params.lr)
        return done(params.lr, 1.0 / (1 + params.lr * 1e3))

    results = pt.run_sweep(sets, execute)
    assert ran == [1e-4, 2e-4] and [r.status for r in results] == ["done", "done"]


def test_a_failed_set_stops_the_sweep_and_is_recorded():
    sets = [pt.ParamSet(lr=1e-4), pt.ParamSet(lr=2e-4), pt.ParamSet(lr=3e-4)]

    def execute(params):
        if params.lr == 2e-4:
            raise pt.StageFailed("Merge LoRA", 1, ["no space left on device"])
        return done(params.lr, 0.9)

    results = pt.run_sweep(sets, execute)
    assert [r.status for r in results] == ["done", "failed"]  # the third never started
    assert "Merge LoRA" in results[1].error and "no space left" in results[1].error


def test_best_result_is_the_lowest_eval_loss_of_the_finished_ones():
    results = [done(1e-4, 0.99), done(4e-4, 0.71), pt.Result(pt.ParamSet(lr=8e-4), "failed", error="x"), done(2e-4, 0.83)]
    assert pt.best_result(results).params.lr == 4e-4
    assert pt.best_result([pt.Result(pt.ParamSet(), "failed", error="x")]) is None
    assert pt.best_result([]) is None


# --- the decision and the notes ------------------------------------------------------------------------------------

CONFIG = {"learning_rate": 2.0e-4, "num_train_epochs": 3.0}


def test_decision_names_the_profile_lines_to_change():
    text = pt.decision_text([done(2e-4, 0.83), done(4e-4, 0.71, run_id="20261002-1")], "hy-mt2-1.8b-fast", CONFIG)
    assert "lr=0.0004" in text and "0.7100" in text and "20261002-1" in text
    assert "configs/llamafactory/hy-mt2-1.8b-fast/train.yaml" in text
    assert "learning_rate: 0.0004" in text and "(now 0.0002)" in text
    assert "num_train_epochs" not in text  # unchanged: no line to edit


def test_decision_takes_unset_parameters_from_the_profile():
    best = pt.Result(pt.ParamSet(epochs=6), "done", "r", 0.7, 150, 1.0, 10.0)
    text = pt.decision_text([best], "hy-mt2-1.8b-fast", CONFIG)
    assert "num_train_epochs: 6" in text and "learning_rate:" not in text


def test_decision_says_when_the_profile_already_has_the_best():
    text = pt.decision_text([done(2e-4, 0.83)], "hy-mt2-1.8b-fast", CONFIG)
    assert "already" in text and "learning_rate:" not in text


def test_decision_warns_when_eval_loss_was_still_falling_and_to_confirm_on_the_full_profile():
    text = pt.decision_text([done(4e-4, 0.71, falling=True)], "hy-mt2-1.8b-fast", CONFIG)
    assert "still falling" in text
    assert "hy-mt2-1.8b" in text and "full profile" in text


def test_decision_with_nothing_finished_says_so():
    assert "no finished" in pt.decision_text([], "hy-mt2-1.8b-fast", CONFIG).lower()


def test_results_markdown_is_a_table_with_a_row_per_set():
    results = [done(1e-4, 0.99, run_id="a"), pt.Result(pt.ParamSet(lr=8e-4), "failed", error="Fine-Tuning failed")]
    lines = pt.results_markdown(results).splitlines()
    assert lines[0].startswith("| params |") and set(lines[1]) <= {"|", "-", " "}
    assert len(lines) == 4
    assert "0.9900" in lines[2] and "| done |" in lines[2]
    assert "failed" in lines[3] and lines[3].count("|") == lines[0].count("|")


# --- the kernel ----------------------------------------------------------------------------------------------------

def test_kernel_warning_only_when_not_the_project_venv(tmp_path):
    venv_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    assert pt.kernel_warning(str(venv_python), str(tmp_path)) is None
    warning = pt.kernel_warning(str(tmp_path / "other" / "python.exe"), str(tmp_path))
    assert ".venv" in warning and "requirements-llamafactory.txt" in warning


def test_kernel_warning_tells_how_to_get_a_jupyter_kernel_from_the_venv(tmp_path):
    warning = pt.kernel_warning(str(tmp_path / "other" / "python.exe"), str(tmp_path))
    assert "ipykernel install" in warning and "jupyter lab" in warning


def scored(lr, eval_loss, chrf, term_pct, run_id="r"):
    scores = {"training": {"chrf": chrf, "term_hits": 0, "term_total": 21, "term_pct": term_pct}}
    return pt.Result(pt.ParamSet(lr=lr), "done", run_id, eval_loss, 80, 1.0, 240.0, scores=scores)


def test_results_markdown_shows_the_eval_scores():
    lines = pt.results_markdown([scored(8e-4, 0.7938, 62.86, 61.9), done(2e-4, 0.83)]).splitlines()
    assert "chrF / term acc" in lines[0] and lines[0].count("|") == lines[2].count("|") == lines[3].count("|")
    assert "training 62.9 / 61.9%" in lines[2]


def test_decision_says_when_the_scores_pick_another_run_than_the_eval_loss():
    results = [scored(4e-4, 0.7724, 53.8, 42.9, run_id="a"), scored(8e-4, 0.7938, 62.9, 61.9, run_id="b"), scored(2e-4, 0.8317, 58.1, 33.3)]
    text = pt.decision_text(results, "hy-mt2-1.8b-fast", CONFIG)
    assert "eval loss 0.7724" in text and "lr=0.0004" in text
    assert "By chrF the best is lr=0.0008" in text and "62.9" in text
    assert "By term accuracy the best is lr=0.0008" in text and "61.9%" in text
    assert "disagree" in text and "full profile" in text


def test_decision_stays_quiet_when_eval_loss_and_scores_agree_or_are_missing():
    agree = pt.decision_text([scored(4e-4, 0.7724, 62.9, 61.9), scored(2e-4, 0.8317, 58.1, 33.3)], "hy-mt2-1.8b-fast", CONFIG)
    assert "By chrF" not in agree and "disagree" not in agree
    assert "disagree" not in pt.decision_text([done(2e-4, 0.83), done(4e-4, 0.71)], "hy-mt2-1.8b-fast", CONFIG)
