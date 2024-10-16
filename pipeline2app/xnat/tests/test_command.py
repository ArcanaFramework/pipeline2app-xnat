from operator import mul
from functools import reduce
import random
import pytest
from pipeline2app.core import ContainerCommand
from conftest import TEST_XNAT_DATASET_BLUEPRINTS, access_dataset


@pytest.mark.parametrize("access_method", ["api", "cs", "cs-internal"])
def test_command_execute(
    xnat_repository, command_spec, work_dir, run_prefix, access_method, xnat_archive_dir
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

    command = ContainerCommand(**command_spec)

    # Start generating the arguments for the CLI
    # Add source to loaded dataset
    command.execute(
        address=dataset.locator,
        input_values=[
            ("first_file", "scan1"),
            ("second_file", "scan2"),
        ],
        output_values=[
            ("concatenated_file", "sink_file"),
        ],
        parameter_values=[
            ("number_of_duplicates", str(duplicates)),
        ],
        raise_errors=True,
        plugin="serial",
        work_dir=str(work_dir),
        loglevel="debug",
        dataset_hierarchy=",".join(bp.hierarchy),
        pipeline_name="test_pipeline",
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
