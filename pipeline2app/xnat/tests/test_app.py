from pathlib import Path
import random
from copy import deepcopy
import pytest
from conftest import (
    TEST_XNAT_DATASET_BLUEPRINTS,
    TestXnatDatasetBlueprint,
    ScanBP,
    FileBP,
    access_dataset,
)
from frametree.xnat import Xnat
from pipeline2app.xnat.image import XnatApp
from pipeline2app.xnat.command import XnatCommand
from pipeline2app.xnat.deploy import (
    install_and_launch_xnat_cs_command,
)
from fileformats.medimage import NiftiGzX, NiftiGzXBvec
from fileformats.text import Plain as Text
from frametree.common import Clinical


PIPELINE_NAME = "test-concatenate"


@pytest.fixture(
    params=["func_api", "bidsapp_api", "func_internal", "bidsapp_internal"],
    scope="session",
)
def run_spec(
    command_spec,
    bids_command_spec,
    xnat_repository,
    xnat_archive_dir,
    request,
    nifti_sample_dir,
    mock_bids_app_image,
    run_prefix,
):
    spec = {}
    task, upload_method = request.param.split("_")
    run_prefix += upload_method
    access_method = "cs" + ("_internal" if upload_method == "internal" else "")
    if task == "func":
        cmd_spec = command_spec
        spec["build"] = {
            "org": "pipeline2app-tests",
            "name": run_prefix + "-concatenate-xnat-cs",
            "version": "1.0",
            "title": "A pipeline to test Pipeline2app's deployment tool",
            "commands": {"concatenate-test": command_spec},
            "authors": [{"name": "Some One", "email": "some.one@an.email.org"}],
            "docs": {
                "info_url": "http://concatenate.readthefakedocs.io",
            },
            "readme": "This is a test README",
            "registry": "a.docker.registry.io",
            "packages": {
                "system": ["git", "vim"],
                "pip": [
                    "fileformats",
                    "fileformats-extras",
                    "fileformats-medimage",
                    "fileformats-medimage-extras",
                    "frametree",
                    "frametree-xnat",
                    "pipeline2app",
                    "pipeline2app-xnat",
                    "pydra",
                ],
            },
        }
        blueprint = TEST_XNAT_DATASET_BLUEPRINTS["concatenate_test"]
        project_id = run_prefix + "concatenate_test"
        blueprint.make_dataset(
            store=xnat_repository,
            dataset_id=project_id,
        )
        spec["dataset"] = access_dataset(
            project_id, access_method, xnat_repository, xnat_archive_dir, run_prefix
        )
        spec["params"] = {"number_of_duplicates": 2}
    elif task == "bidsapp":
        bids_command_spec["configuration"]["executable"] = "/launch.sh"
        cmd_spec = bids_command_spec
        spec["build"] = {
            "org": "pipeline2app-tests",
            "name": run_prefix + "-bids-app-xnat-cs",
            "version": "1.0",
            "title": "A pipeline to test wrapping of BIDS apps",
            "base_image": {
                "name": mock_bids_app_image,
                "package_manager": "apt",
            },
            "packages": {
                "system": ["git", "vim"],
                "pip": [
                    "fileformats",
                    "fileformats-extras",
                    "fileformats-medimage",
                    "fileformats-medimage-extras",
                    "frametree",
                    "frametree-bids",
                    "frametree-xnat",
                    "pydra",
                    "pipeline2app",
                    "pipeline2app-xnat",
                ],
            },
            "commands": {"bids-test-command": bids_command_spec},
            "authors": [
                {"name": "Some One Else", "email": "some.oneelse@an.email.org"}
            ],
            "docs": {
                "info_url": "http://a-bids-app.readthefakedocs.io",
            },
            "readme": "This is another test README for BIDS app image",
            "registry": "another.docker.registry.io",
        }
        blueprint = TestXnatDatasetBlueprint(
            dim_lengths=[1, 1, 1],
            scans=[
                ScanBP(
                    "anat/T1w",
                    [
                        FileBP(
                            path="NiftiGzX",
                            datatype=NiftiGzX,
                            filenames=["anat/T1w.nii.gz", "anat/T1w.json"],
                        )
                    ],
                ),
                ScanBP(
                    "anat/T2w",
                    [
                        FileBP(
                            path="NiftiGzX",
                            datatype=NiftiGzX,
                            filenames=["anat/T2w.nii.gz", "anat/T2w.json"],
                        )
                    ],
                ),
                ScanBP(
                    "dwi/dwi",
                    [
                        FileBP(
                            path="NiftiGzXBvec",
                            datatype=NiftiGzXBvec,
                            filenames=[
                                "dwi/dwi.nii.gz",
                                "dwi/dwi.json",
                                "dwi/dwi.bvec",
                                "dwi/dwi.bval",
                            ],
                        )
                    ],
                ),
            ],
            derivatives=[
                FileBP(
                    path="file1",
                    row_frequency=Clinical.session,
                    datatype=Text,
                    filenames=["file1_sink.txt"],
                ),
                FileBP(
                    path="file2",
                    row_frequency=Clinical.session,
                    datatype=Text,
                    filenames=["file2_sink.txt"],
                ),
            ],
        )
        project_id = run_prefix + "xnat_cs_bidsapp"
        blueprint.make_dataset(
            store=xnat_repository,
            dataset_id=project_id,
            source_data=nifti_sample_dir,
        )
        spec["dataset"] = access_dataset(
            project_id, access_method, xnat_repository, xnat_archive_dir, run_prefix
        )
        spec["params"] = {}
    else:
        assert False, f"unrecognised request param '{task}'"
    cmd_spec["internal_upload"] = upload_method == "internal"
    return spec


def test_xnat_cs_pipeline(xnat_repository, run_spec, run_prefix, work_dir):
    """Tests the complete XNAT deployment pipeline by building and running a
    container"""

    # Retrieve test dataset and build and command specs from fixtures
    build_spec = run_spec["build"]
    dataset = run_spec["dataset"]
    params = run_spec["params"]
    blueprint = dataset.__annotations__["blueprint"]

    image_spec = XnatApp(**build_spec)

    image_spec.make(
        build_dir=work_dir,
        pipeline2app_install_extras=["test"],
        use_local_packages=True,
        for_localhost=True,
    )

    # We manually set the command in the test XNAT instance as commands are
    # loaded from images when they are pulled from a registry and we use
    # the fact that the container service test XNAT instance shares the
    # outer Docker socket. Since we build the pipeline image with the same
    # socket there is no need to pull it.

    cmd = image_spec.command()
    xnat_command = cmd.make_json()

    launch_inputs = {}

    for inpt, scan in zip(xnat_command["inputs"], blueprint.scans):
        launch_inputs[XnatCommand.path2xnatname(inpt["name"])] = scan.name

    for pname, pval in params.items():
        launch_inputs[pname] = pval

    if cmd.internal_upload:
        # If using internal upload, the output names are fixed
        output_values = {o: o for o in cmd.output_names}
    else:
        output_values = {o: o + "_sink" for o in cmd.output_names}
        launch_inputs.update(output_values)

    with xnat_repository.connection:

        xlogin = xnat_repository.connection

        test_xsession = next(iter(xlogin.projects[dataset.id].experiments.values()))

        workflow_id, status, out_str = install_and_launch_xnat_cs_command(
            command_json=xnat_command,
            project_id=dataset.id,
            session_id=test_xsession.id,
            inputs=launch_inputs,
            xlogin=xlogin,
        )

        assert status == "Complete", f"Workflow {workflow_id} failed.\n{out_str}"

        access_type = "direct" if cmd.internal_upload else "api"

        assert f"via {access_type} access" in out_str.lower()

        assert sorted(r.label for r in test_xsession.resources.values()) == sorted(
            output_values.values()
        )

        for output_name, sinked_name in output_values.items():
            deriv = next(d for d in blueprint.derivatives if d.path == output_name)
            uploaded_files = sorted(
                Path(f).name.lstrip("sub-DEFAULT_")
                for f in test_xsession.resources[sinked_name].files
            )
            if cmd.internal_upload:
                reference = sorted(
                    d.rstrip("_sink.txt") + ".txt" for d in deriv.filenames
                )
            else:
                reference = sorted(deriv.filenames)
            assert uploaded_files == reference


def test_multi_command(xnat_repository: Xnat, tmp_path: Path, run_prefix) -> None:

    bp = TestXnatDatasetBlueprint(
        dim_lengths=[1, 1, 1],
        scans=[
            ScanBP(
                name="scan1",
                resources=[FileBP(path="TEXT", datatype=Text, filenames=["file1.txt"])],
            ),
            ScanBP(
                name="scan2",
                resources=[FileBP(path="TEXT", datatype=Text, filenames=["file2.txt"])],
            ),
        ],
    )

    project_id = run_prefix + "multi_command" + str(hex(random.getrandbits(16)))[2:]

    dataset = bp.make_dataset(
        dataset_id=project_id,
        store=xnat_repository,
        name="",
    )

    two_dup_spec = dict(
        name="concatenate",
        task="pipeline2app.testing.tasks:concatenate",
        row_frequency=Clinical.session.tostr(),
        inputs=[
            {
                "name": "first_file",
                "datatype": "text/text-file",
                "field": "in_file1",
                "help": "dummy",
            },
            {
                "name": "second_file",
                "datatype": "text/text-file",
                "field": "in_file2",
                "help": "dummy",
            },
        ],
        outputs=[
            {
                "name": "concatenated",
                "datatype": "text/text-file",
                "field": "out_file",
                "help": "dummy",
            }
        ],
        parameters={
            "duplicates": {
                "datatype": "field/integer",
                "default": 2,
                "help": "dummy",
            }
        },
    )

    three_dup_spec = deepcopy(two_dup_spec)
    three_dup_spec["parameters"]["duplicates"]["default"] = 3

    test_spec = {
        "name": "test_multi_commands",
        "title": "a test image for multi-image commands",
        "commands": {
            "two_duplicates": two_dup_spec,
            "three_duplicates": three_dup_spec,
        },
        "version": "1.0",
        "packages": {
            "system": ["vim"],  # just to test it out
            "pip": {
                "fileformats": None,
                "pipeline2app": None,
                "frametree": None,
            },
        },
        "authors": [{"name": "Some One", "email": "some.one@an.email.org"}],
        "docs": {
            "info_url": "http://concatenate.readthefakedocs.io",
        },
    }

    app = XnatApp.load(test_spec)

    app.make(
        build_dir=tmp_path / "build-dir",
        pipeline2app_install_extras=["test"],
        use_local_packages=True,
        for_localhost=True,
    )

    fnames = ["file1.txt", "file2.txt"]

    base_launch_inputs = {
        "first_file": "scan1",
        "second_file": "scan2",
    }

    command_names = ["two_duplicates", "three_duplicates"]

    with xnat_repository.connection as xlogin:

        test_xsession = next(iter(xlogin.projects[dataset.id].experiments.values()))
        for command_name in command_names:

            launch_inputs = deepcopy(base_launch_inputs)
            launch_inputs["concatenated"] = command_name

            workflow_id, status, out_str = install_and_launch_xnat_cs_command(
                command_json=app.command(command_name).make_json(),
                project_id=project_id,
                session_id=test_xsession.id,
                inputs=launch_inputs,
                xlogin=xlogin,
            )

            assert status == "Complete", f"Workflow {workflow_id} failed.\n{out_str}"

        assert sorted(r.label for r in test_xsession.resources.values()) == sorted(
            command_names
        )

    # Add source column to saved dataset
    reloaded = dataset.reload()
    for command_name in command_names:
        sink = reloaded[command_name]
        duplicates = 2 if command_name == "two_duplicates" else 3
        expected_contents = "\n".join(fnames * duplicates)
        for item in sink:
            with open(item) as f:
                contents = f.read()
            assert contents == expected_contents
