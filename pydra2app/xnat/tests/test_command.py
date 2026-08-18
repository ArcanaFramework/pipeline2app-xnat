from operator import mul
from functools import reduce
import logging
from pathlib import Path
import random
import typing as ty
from frametree.xnat.api import Xnat
from frametree.xnat.testing import TestXnatDatasetBlueprint
from frametree.axes.medimage import MedImage
import pytest
from fileformats.text import TextFile
from pydra2app.xnat import XnatCommand
from conftest import TEST_XNAT_DATASET_BLUEPRINTS, access_dataset


@pytest.mark.parametrize("access_method", ["api", "cs", "cs-internal"])
def test_command_execute(
    xnat_repository: Xnat,
    command_spec: dict[str, ty.Any],
    work_dir: Path,
    run_prefix: str,
    access_method: ty.Literal["api"] | ty.Literal["cs"] | ty.Literal["cs-internal"],
    xnat_archive_dir: Path,
    caplog: pytest.LogCaptureFixture,
):
    # Get CLI name for dataset (i.e. file system path prepended by 'file_system//')

    duplicates = 1
    bp = TEST_XNAT_DATASET_BLUEPRINTS["concatenate_test"]
    project_id = (
        run_prefix
        + "contenatecommand"
        + access_method
        + str(hex(random.getrandbits(16)))[2:]
    )
    bp.make_dataset(
        dataset_id=project_id,
        store=xnat_repository,
        name="",
    )
    dataset = access_dataset(
        project_id, access_method, xnat_repository, xnat_archive_dir, run_prefix
    )
    dataset.save()

    command = XnatCommand(**command_spec)

    logging.getLogger("pydra2app").setLevel(logging.DEBUG)
    logging.getLogger("frametree").setLevel(logging.DEBUG)

    # Start generating the arguments for the CLI
    # Add source to loaded dataset
    command.execute(
        address=dataset.address,
        input_values=[
            ("in_file1", "scan1"),
            ("in_file2", "scan2"),
        ],
        output_values=[
            ("out_file", "sink_file"),
        ],
        parameter_values=[
            ("duplicates", str(duplicates)),
        ],
        raise_errors=True,
        worker="debug",
        work_dir=str(work_dir),
        loglevel="debug",
        dataset_hierarchy=",".join(bp.hierarchy),
        pipeline_name="test_pipeline",
        save_frameset=True,
    )
    # Add source column to saved dataset
    reloaded = dataset.reload()
    sink = reloaded["sink_file"]
    assert len(sink) == reduce(mul, bp.dim_lengths)
    fnames = ["file1.txt", "file2.txt"]
    expected_contents = "\n".join(fnames * duplicates)
    for item in sink:
        with open(item) as f:
            contents = f.read()
        assert contents == expected_contents


def test_command_execute_single_session_load(
    xnat_repository: Xnat,
    command_spec: dict[str, ty.Any],
    work_dir: Path,
    run_prefix: str,
    xnat_archive_dir: Path,
    caplog: pytest.LogCaptureFixture,
):
    # Get CLI name for dataset (i.e. file system path prepended by 'file_system//')

    logging.getLogger("pydra2app").setLevel(logging.DEBUG)
    logging.getLogger("frametree").setLevel(logging.DEBUG)

    duplicates = 1
    bp = TEST_XNAT_DATASET_BLUEPRINTS["concatenate_test"]
    project_id = run_prefix + "singlesession"
    bp.make_dataset(
        dataset_id=project_id,
        store=xnat_repository,
        name="",
    )
    dataset = access_dataset(
        project_id, "api", xnat_repository, xnat_archive_dir, run_prefix
    )

    command = XnatCommand(**command_spec)

    SESSION_ID = "visit0group0member0"
    EMPTY_SESSION_ID = "visit0group0member1"

    # Start generating the arguments for the CLI
    # Add source to loaded dataset
    command.execute(
        address=dataset.address,
        input_values=[
            ("in_file1", "scan1"),
            ("in_file2", "scan2"),
        ],
        output_values=[
            ("out_file", "sink_file"),
        ],
        parameter_values=[
            ("duplicates", str(duplicates)),
        ],
        raise_errors=True,
        worker="debug",
        work_dir=str(work_dir),
        loglevel="debug",
        dataset_hierarchy=",".join(bp.hierarchy),
        pipeline_name="test_pipeline",
        ids=[SESSION_ID],
    )
    # Full dataset tree is accessed twice (4 leaves) in creation and then
    # one leaf is accessed 3 times in: to-process, source and sink nodes
    assert caplog.text.count("Adding leaf to data tree at path") == 7
    # Add source column to saved dataset
    reloaded = dataset.reload()
    assert not reloaded.columns
    reloaded.add_sink("sink_file", datatype=TextFile)
    sink = reloaded["sink_file"]
    assert len(sink) == 2
    fnames = ["file1.txt", "file2.txt"]
    expected_contents = "\n".join(fnames * duplicates)
    item = sink[SESSION_ID]
    with open(item) as f:
        contents = f.read()
    assert contents == expected_contents


def test_command_execute_at_root_frequency(
    xnat_repository: Xnat,
    work_dir: Path,
    run_prefix: str,
) -> None:
    """Checks that a command can be run against a real XNAT project with
    `operates_on` set to the dataset-wide root frequency (`MedImage.constant`), so it
    executes exactly once for the whole project - taking its inputs from, and writing
    its output to, the project's root row - rather than once per session, regardless
    of how many sessions the project contains.
    """

    duplicates = 1
    # Create a project with several unrelated sessions, to check that the
    # root-frequency command isn't run once per session and only touches the root row
    bp = TestXnatDatasetBlueprint(dim_lengths=[1, 1, 3], scans=[])
    project_id = run_prefix + "rootfreq" + str(hex(random.getrandbits(16)))[2:]
    dataset = bp.make_dataset(xnat_repository, project_id, name="")

    # Blueprint scans are always created per-session, so the two inputs the command
    # will concatenate are inserted directly into the project's root row as sinks
    # instead (mirroring how derivatives are inserted in frametree-xnat's own store
    # tests)
    fnames = ["file1.txt", "file2.txt"]
    root_row = dataset.root
    with dataset.store.connection:
        for name, fname in zip(["file1", "file2"], fnames):
            dataset.add_sink(name, datatype=TextFile, row_frequency=MedImage.constant)
            src_path = work_dir / fname
            src_path.write_text(fname)
            root_row[name] = TextFile(src_path)
    dataset.save()

    assert len(dataset.rows(MedImage.session)) == 3  # sanity check on session rows

    command = XnatCommand(
        name="concatenate",
        task="frametree.testing.tasks:Concatenate",
        operates_on=MedImage.constant,
    )

    command.execute(
        address=dataset.address,
        input_values=[
            ("in_file1", "<file1>"),
            ("in_file2", "<file2>"),
        ],
        output_values=[
            ("out_file", "sink_1"),
        ],
        parameter_values=[
            ("duplicates", str(duplicates)),
        ],
        raise_errors=True,
        worker="debug",
        work_dir=str(work_dir),
        loglevel="debug",
        dataset_hierarchy=",".join(bp.hierarchy),
        pipeline_name="test_pipeline",
        save_frameset=True,
    )

    reloaded = dataset.reload()
    sink = reloaded["sink_1"]
    # Exactly one output for the whole project, not one per session
    assert len(sink) == 1
    expected_contents = "\n".join(fnames * duplicates)
    item = next(iter(sink))
    with open(item) as f:
        contents = f.read()
    assert contents == expected_contents
