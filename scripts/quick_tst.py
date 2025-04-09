from frametree.core.utils import show_cli_trace
from pydra2app.core.cli import make
from pydra2app.xnat.cli import save_token, install_command, launch_command


def test_make(cli_runner):

    result = cli_runner(
        make,
        [
            "xnat",
            (
                "/Users/tclose/git/pipelines-community/specs/"
                "australian-imaging-service-community/examples/bet.yaml"
            ),
            "--spec-root",
            "/Users/tclose/git/pipelines-community/specs",
            "--for-localhost",
            "--use-local-packages",
        ],
    )

    assert result.exit_code == 0, show_cli_trace(result)


def test_save_token(cli_runner):

    result = cli_runner(
        save_token,
        [
            "--server",
            "http://localhost:8080",
            "--user",
            "admin",
            "--password",
            "admin",
        ],
    )

    assert result.exit_code == 0, show_cli_trace(result)


def test_install_command(cli_runner):

    result = cli_runner(
        install_command,
        [
            "australian-imaging-service-community/examples.bet:6.0.6.4-1",
            "--enable",
            "--enable-project",
            "OPENNEURO_T1W",
            "--replace-existing",
        ],
    )

    assert result.exit_code == 0, show_cli_trace(result)


def test_launch_command(cli_runner):

    result = cli_runner(
        launch_command,
        [
            "examples.bet",
            "OPENNEURO_T1W",
            "subject01_MR01",
            "--input",
            "T1w",
            "t1w",
        ],
    )

    assert result.exit_code == 0, show_cli_trace(result)
