from frametree.core.utils import show_cli_trace
from pydra2app.core.cli import make


def test_quickly(cli_runner):

    result = cli_runner(
        make,
        [
            "xnat",
            (
                "/Users/tclose/git/pipelines-community/specs/"
                "australian-imaging-service-community/examples/zip.yaml"
            ),
            "--spec-root",
            "/Users/tclose/git/pipelines-community/specs",
            "--for-localhost",
            "--export-file",
            "xnat_command.json",
            "~/zip-xnat-command.json",
        ],
    )

    assert result.exit_code == 0, show_cli_trace(result)
