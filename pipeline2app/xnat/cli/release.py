import re
import json
from pathlib import Path
import yaml
import logging
import click
import xnat
import os
from .base import xnat_group
from ..deploy import install_cs_command, launch_cs_command


# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

XNAT_HOST_KEY = "XNAT_HOST"
XNAT_USER_KEY = "XNAT_USER"
XNAT_PASS_KEY = "XNAT_PASS"
XNAT_AUTH_FILE_KEY = "XNAT_AUTH_FILE"
XNAT_AUTH_FILE_DEFAULT = Path("~/.pipeline2app_xnat_user_token.json").expanduser()


def load_auth(server, user, password, auth_file):
    if server is not None:
        if user is None:
            raise RuntimeError(f"A user must be provided if a server ({server}) is")
        if password is None:
            raise RuntimeError(f"a password must be provided if a server ({server}) is")
    else:
        if auth_file == XNAT_AUTH_FILE_DEFAULT and not Path(auth_file).exists():
            raise RuntimeError(
                "An auth file must be provided if no server is. "
                "Use pipeline2app ext xnat save-token to create one"
            )
        click.echo(f"Reading existing alias/token pair from '{str(auth_file)}")
        with open(auth_file) as fp:
            auth = json.load(fp)
        server = auth["server"]
        user = auth["alias"]
        password = auth["secret"]
    return server, user, password


@xnat_group.command(
    name="install-command",
    help="""Installs a container service pipelines command on an XNAT server

IMAGE_OR_COMMAND_FILE the name of the Pydra2App container service pipeline Docker image or
the path to a command JSON file to install
""",
)
@click.argument("image_or_command_file", type=str)
@click.option(
    "--enable/--disable",
    type=bool,
    default=False,
    help=("Whether to enable the command globally"),
)
@click.option(
    "--enable-project",
    "projects_to_enable",
    type=str,
    multiple=True,
    help=("Enable the command for the given project"),
)
@click.option(
    "--replace-existing/--no-replace-existing",
    default=False,
    help=("Whether to replace existing command with the same name"),
)
@click.option(
    "--server",
    envvar=XNAT_HOST_KEY,
    default=None,
    help=("The XNAT server to save install the command on"),
)
@click.option(
    "--user",
    envvar=XNAT_USER_KEY,
    help=("the username used to authenticate with the XNAT instance to update"),
)
@click.option(
    "--password",
    envvar=XNAT_PASS_KEY,
    help=("the password used to authenticate with the XNAT instance to update"),
)
@click.option(
    "--auth-file",
    type=click.Path(path_type=Path),
    default=XNAT_AUTH_FILE_DEFAULT,
    envvar=XNAT_AUTH_FILE_KEY,
    help=("The path to save the alias/token pair to"),
)
def install_command(
    image_or_command_file,
    enable,
    projects_to_enable,
    replace_existing,
    server,
    user,
    password,
    auth_file,
):
    server, user, password = load_auth(server, user, password, auth_file)

    if Path(image_or_command_file).exists():
        with open(image_or_command_file) as f:
            image_or_command_file = json.load(f)

    with xnat.connect(server=server, user=user, password=password) as xlogin:
        install_cs_command(
            image_or_command_file,
            xlogin=xlogin,
            enable=enable,
            projects_to_enable=projects_to_enable,
            replace_existing=replace_existing,
        )

    click.echo(
        f"Successfully installed the '{image_or_command_file}' pipeline on '{server}'"
    )


@xnat_group.command(
    name="launch-command",
    help="""Launches a container service pipelines command on an XNAT server

COMMAND_NAME the name of the command to launch

PROJECT_ID of the project to launch the command on

SESSION_ID of the session to launch the command on
""",
)
@click.argument("command_name", type=str)
@click.argument("project_id", type=str)
@click.argument("session_id", type=str)
@click.option(
    "--input",
    "inputs",
    type=(str, str),
    multiple=True,
    help=("The input values to pass to the command"),
)
@click.option(
    "--timeout",
    type=int,
    default=1000,
    help=("The time to wait for the command to complete"),
)
@click.option(
    "--poll-interval",
    type=int,
    default=10,
    help=("The time to wait between polling the command status"),
)
@click.option(
    "--server",
    envvar=XNAT_HOST_KEY,
    default=None,
    help=("The XNAT server to save install the command on"),
)
@click.option(
    "--user",
    envvar=XNAT_USER_KEY,
    help=("the username used to authenticate with the XNAT instance to update"),
)
@click.option(
    "--password",
    envvar=XNAT_PASS_KEY,
    help=("the password used to authenticate with the XNAT instance to update"),
)
@click.option(
    "--auth-file",
    type=click.Path(path_type=Path),
    default=XNAT_AUTH_FILE_DEFAULT,
    envvar=XNAT_AUTH_FILE_KEY,
    help=("The path to save the alias/token pair to"),
)
def launch_command(
    command_name,
    project_id,
    session_id,
    inputs,
    timeout,
    poll_interval,
    server,
    user,
    password,
    auth_file,
):

    server, user, password = load_auth(server, user, password, auth_file)

    inputs_dict = {}
    for name, val in inputs:
        if name in inputs_dict:
            raise KeyError(
                f"Duplicate input name '{name}' (values: {inputs_dict[name]}, {val})"
            )
        inputs_dict[name] = val

    with xnat.connect(server=server, user=user, password=password) as xlogin:
        launch_cs_command(
            command_name,
            project_id=project_id,
            session_id=session_id,
            inputs=inputs_dict,
            timeout=timeout,
            poll_interval=poll_interval,
            xlogin=xlogin,
        )

    click.echo(
        f"Successfully launched the '{command_name}' pipeline on '{session_id}' session "
        f"in '{project_id}' project on '{server}'"
    )


@xnat_group.command(
    name="save-token",
    help="""Logs into the XNAT instance, generates a user access token and saves it in an
authentication file. If a username and password are not provided, then it is assumed that
a valid alias/token pair already exists in the authentication file, and they are used to
regenerate a new alias/token pair to prevent them expiring (2 days by default).

CONFIG_YAML a YAML file contains the login details for the XNAT server to update

AUTH_FILE the path at which to save the authentication file containing the alias/token
""",
)
@click.option(
    "--auth-file",
    type=click.Path(path_type=Path),
    default=XNAT_AUTH_FILE_DEFAULT,
    envvar=XNAT_AUTH_FILE_KEY,
    help=("The path to save the alias/token pair to"),
)
@click.option(
    "--server",
    envvar=XNAT_HOST_KEY,
    default=None,
    help=("The XNAT server to save the credentials to"),
)
@click.option(
    "--user",
    envvar=XNAT_USER_KEY,
    default=None,
    help=("the username used to authenticate with the XNAT instance to update"),
)
@click.option(
    "--password",
    envvar=XNAT_PASS_KEY,
    default=None,
    help=("the password used to authenticate with the XNAT instance to update"),
)
def save_token(auth_file, server, user, password):

    server, user, password = load_auth(server, user, password, auth_file)

    with xnat.connect(server=server, user=user, password=password) as xlogin:
        alias, secret = xlogin.services.issue_token()

    with open(auth_file, "w") as f:
        json.dump(
            {
                "server": server,
                "alias": alias,
                "secret": secret,
            },
            f,
        )
    os.chmod(auth_file, 0o600)

    click.echo(
        f"Saved alias/token for '{server}' XNAT in '{str(auth_file)}' file, "
        "please ensure the file is secure"
    )


@xnat_group.command(
    name="deploy-pipelines",
    help=f"""Updates the installed pipelines on an XNAT instance from a manifest
JSON file using the XNAT instance's REST API.

MANIFEST_FILE is a JSON file containing a list of container images built in a release
created by `pipeline2app deploy xnat build`

Authentication credentials can be passed through the {XNAT_USER_KEY}
and {XNAT_PASS_KEY} environment variables. Otherwise, tokens can be saved
in a JSON file passed to '--auth'.

Which of available pipelines to install can be controlled by a YAML file passed to the
'--filters' option of the form
    \b
    include:
    - tag: ghcr.io/Australian-Imaging-Service/mri.human.neuro.*
    - tag: ghcr.io/Australian-Imaging-Service/pet.rodent.*
    exclude:
    - tag: ghcr.io/Australian-Imaging-Service/mri.human.neuro.bidsapps.
""",
)
@click.argument("manifest_file", type=click.File())
@click.option(
    "--server",
    envvar=XNAT_HOST_KEY,
    default=None,
    help=("The XNAT server to save install the command on"),
)
@click.option(
    "--user",
    envvar=XNAT_USER_KEY,
    help=("the username used to authenticate with the XNAT instance to update"),
)
@click.option(
    "--password",
    envvar=XNAT_PASS_KEY,
    help=("the password used to authenticate with the XNAT instance to update"),
)
@click.option(
    "--auth-file",
    type=click.Path(path_type=Path),
    default=XNAT_AUTH_FILE_DEFAULT,
    envvar=XNAT_AUTH_FILE_KEY,
    help=("The path to save the alias/token pair to"),
)
@click.option(
    "--filters",
    "filters_file",
    default=None,
    type=click.File(),
    help=("a YAML file containing filter rules for the images to install"),
)
def deploy_pipelines(manifest_file, server, user, password, auth_file, filters_file):

    server, user, password = load_auth(server, user, password, auth_file)

    manifest = json.load(manifest_file)
    filters = yaml.load(filters_file, Loader=yaml.Loader) if filters_file else {}

    def matches_entry(entry, match_exprs, default=True):
        """Determines whether an entry meets the inclusion and exclusion criteria

        Parameters
        ----------
        entry : ty.Dict[str, Any]
            a image entry in the manifest
        exprs : list[ty.Dict[str, str]]
            match criteria
        default : bool
            the value if match_exprs are empty
        """
        if not match_exprs:
            return default
        return re.match(
            "|".join(
                i["name"].replace(".", "\\.").replace("*", ".*") for i in match_exprs
            ),
            entry["name"],
        )

    with xnat.connect(
        server=server,
        user=user,
        password=password,
    ) as xlogin:

        for entry in manifest["images"]:
            if matches_entry(entry, filters.get("include")) and not matches_entry(
                entry, filters.get("exclude"), default=False
            ):
                tag = f"{entry['name']}:{entry['version']}"
                xlogin.post(
                    "/xapi/docker/pull", query={"image": tag, "save-commands": True}
                )

                # Enable the commands in the built image
                for cmd in xlogin.get("/xapi/commands").json():
                    if cmd["image"] == tag:
                        for wrapper in cmd["xnat"]:
                            xlogin.put(
                                f"/xapi/commands/{cmd['id']}/"
                                f"wrappers/{wrapper['id']}/enabled"
                            )
                click.echo(f"Installed and enabled {tag}")
            else:
                click.echo(f"Skipping {tag} as it doesn't match filters")

    click.echo(
        f"Successfully updated all container images from '{manifest['release']}' of "
        f"'{manifest['package']}' package that match provided filters"
    )
