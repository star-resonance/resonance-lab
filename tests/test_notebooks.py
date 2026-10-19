"""The committed notebooks hold no outputs: eval lines are players' chat (CLAUDE.md: no data in git)."""

import ast
import glob
import json
import os

import pytest

from config import BASE_DIR

NOTEBOOKS = sorted(glob.glob(os.path.join(BASE_DIR, "notebooks", "*.ipynb")))
COMMITTED = [path for path in NOTEBOOKS if ".executed." not in os.path.basename(path)]


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_the_parameter_test_notebook_exists():
    assert os.path.join(BASE_DIR, "notebooks", "parameter_test.ipynb") in COMMITTED


@pytest.mark.parametrize("path", COMMITTED, ids=os.path.basename)
def test_a_notebook_has_no_outputs_and_no_execution_counts(path):
    for index, cell in enumerate(load(path)["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell["outputs"] == [], f"cell {index} of {os.path.basename(path)} has saved output: strip it before committing"
        assert cell["execution_count"] is None, f"cell {index} of {os.path.basename(path)} has an execution count"


@pytest.mark.parametrize("path", COMMITTED, ids=os.path.basename)
def test_a_notebook_is_well_formed_and_its_code_parses(path):
    notebook = load(path)
    assert notebook["nbformat"] == 4
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids)), "cell ids must be unique"
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"{os.path.basename(path)} cell {index}")


@pytest.mark.parametrize("path", COMMITTED, ids=os.path.basename)
def test_a_notebook_carries_no_machine_specific_metadata(path):
    metadata = load(path)["metadata"]
    assert set(metadata) <= {"kernelspec", "language_info"}
    assert "path" not in json.dumps(metadata).lower().replace("pathlib", "")


def source(path):
    return ["".join(cell["source"]) for cell in load(path)["cells"]]


def test_the_notebook_is_for_jupyter_not_pycharm():
    notebook = os.path.join(BASE_DIR, "notebooks", "parameter_test.ipynb")
    text = "\n".join(source(notebook))
    assert "pycharm" not in text.lower()
    assert "jupyter lab" in text
    with open(os.path.join(BASE_DIR, "requirements-notebook.txt"), encoding="utf-8") as f:
        requirements = [line.split("#")[0].strip().lower() for line in f]
    assert "jupyterlab" in requirements


def test_run_all_does_not_start_the_sweep():
    """The sweep trains, merges and evaluates every parameter set (hours of GPU): it needs a switch that is off."""
    notebook = os.path.join(BASE_DIR, "notebooks", "parameter_test.ipynb")
    (cell,) = [text for text in source(notebook) if "pt.run_sweep(" in text]
    assert "RUN_SWEEP = False" in cell
    call = next(line for line in cell.splitlines() if "pt.run_sweep(" in line)
    assert call.startswith(" ") and "if RUN_SWEEP:" in cell.split(call)[0]


GATE_NOTEBOOK = os.path.join(BASE_DIR, "notebooks", "gate_pipeline.ipynb")


def test_the_gate_notebook_exists_and_is_for_jupyter_in_its_own_venv():
    assert GATE_NOTEBOOK in COMMITTED
    text = "\n".join(source(GATE_NOTEBOOK))
    assert "pycharm" not in text.lower()
    assert "jupyter lab" in text and ".venv-kev" in text
    with open(os.path.join(BASE_DIR, "requirements-kev.txt"), encoding="utf-8") as f:
        requirements = [line.split("#")[0].strip().lower() for line in f]
    assert "jupyterlab" in requirements and "ipykernel" in requirements


def test_run_all_in_the_gate_notebook_starts_nothing_long_and_overwrites_nothing():
    """Re-fetching, the draft (a file a human edits) and the full pass (hours) each sit behind a switch that is off."""
    cells = source(GATE_NOTEBOOK)
    text = "\n".join(cells)
    for switch in ("REFRESH_DATA", "DRAFT_SAMPLE", "RUN_FULL_PASS"):
        assert f"{switch} = False" in text
    (pass_cell,) = [cell for cell in cells if "categorize.judge_pass(" in cell]
    assert "if RUN_FULL_PASS:" in pass_cell.split("categorize.judge_pass(")[0]
    (draft_cell,) = [cell for cell in cells if "gate_pipeline.write_draft(" in cell]
    assert "if DRAFT_SAMPLE:" in draft_cell.split("gate_pipeline.write_draft(")[0]
    assert "overwrite=True" not in text
    (fetch_cell,) = [cell for cell in cells if "subprocess.run(" in cell]
    assert "if REFRESH_DATA:" in fetch_cell.split("subprocess.run(")[0]


def test_the_gate_notebook_imports_torch_and_kev_only_where_they_are_needed():
    """The server backend must work in a kernel without torch: judge_local is imported under the local branch, torch guarded."""
    for index, cell in enumerate(source(GATE_NOTEBOOK)):
        for number, line in enumerate(cell.splitlines(), 1):
            if "import judge_local" in line or "from judge_local" in line:
                assert line.startswith("    "), f"cell {index} line {number}: judge_local must be imported inside `if BACKEND == \"local\":`"
            assert not line.startswith(("import kev", "from kev")), f"cell {index}: kev is imported by judge_local, lazily"
    setup = source(GATE_NOTEBOOK)[1]
    assert "except ImportError" in setup.split("import torch")[1]


def test_the_gate_notebooks_default_judge_is_the_configs():
    import config

    assert "RUN = config.JUDGE_LOCAL_RUN" in "\n".join(source(GATE_NOTEBOOK))
    assert config.JUDGE_LOCAL_RUN.startswith("jaredpalmer/kev-")


def test_the_gate_notebook_saves_no_probe_unless_asked_and_compares_without_a_model():
    cells = source(GATE_NOTEBOOK)
    text = "\n".join(cells)
    assert "PROBE_LABEL = None" in text
    (save_cell,) = [cell for cell in cells if "gate_compare.save_probe(" in cell]
    assert "if PROBE_LABEL:" in save_cell.split("gate_compare.save_probe(")[0]
    (compare_cell,) = [cell for cell in cells if "gate_compare.compare_probes(" in cell]
    assert "JUDGE" not in compare_cell and "client" not in compare_cell, "the comparison reads saved files: it must work in a kernel with no judge loaded"


def test_the_gate_notebook_asks_the_question_of_its_prompt_variant():
    text = "\n".join(source(GATE_NOTEBOOK))
    assert 'JUDGE_PROMPT = "default"' in text
    assert "judge_prompts.judge_question(" in text and "GateJudge(client, OPTIONS, CUTOFF, INSTRUCTIONS)" in text
    assert "gate_judge.INSTRUCTIONS" not in text, "the question comes from the variant, never from the default constant"
    assert 'notes=f"prompt {JUDGE_PROMPT}"' in text


def test_the_gate_notebook_previews_the_training_side_and_saves_a_recipe_only_when_asked():
    """The recipe numbers come from the lines preprocess would train on; a chosen mix becomes a file only behind a switch that is off."""
    cells = source(GATE_NOTEBOOK)
    text = "\n".join(cells)
    assert "gate_pipeline.train_side(" in text and "gate_pipeline.train_side_preview(" in text
    assert "recipe_preview_by_name(" not in text, "the preview of every distinct line is an upper bound: the notebook shows the training side"
    assert "SAVE_RECIPE = False" in text
    (save_cell,) = [cell for cell in cells if "gate_pipeline.save_recipe(" in cell]
    assert "if SAVE_RECIPE:" in save_cell.split("gate_pipeline.save_recipe(")[0]
    assert "gate_pipeline.choose_recipe(" in save_cell


def test_the_parameter_notebook_trains_on_its_recipe_and_runs_the_recipes_data_stages():
    notebook = os.path.join(BASE_DIR, "notebooks", "parameter_test.ipynb")
    cells = source(notebook)
    text = "\n".join(cells)
    assert "RECIPE = None" in text and "recipe=RECIPE" in text
    (data_cell,) = [cell for cell in cells if "pt.run_command(COMMANDS[name])" in cell]
    assert "pt.DATA_STAGES if name in COMMANDS" in data_cell, "a recipe adds data stages without REFRESH_DATA: run what the lifecycle has"
