from __future__ import annotations
import typing as ty
import re
import attrs
from fileformats.core import FileSet, to_mime
from pipeline2app.core.command.base import ContainerCommand
from frametree.xnat import XnatViaCS
from frametree.core.axes import Axes
from frametree.core.utils import path2label
from frametree.common import Clinical


if ty.TYPE_CHECKING:
    from .image import XnatApp


@attrs.define(kw_only=True, auto_attribs=False)
class XnatCommand(ContainerCommand):  # type: ignore[misc]

    image: XnatApp = attrs.field(default=None)
    internal_upload: bool = attrs.field(default=False)

    # Hard-code the axes of XNAT commands to be clinical
    AXES: ty.Optional[ty.Type[Axes]] = Clinical

    def make_json(self) -> ty.Dict[str, ty.Any]:
        """Constructs the XNAT CS "command" JSON config, which specifies how XNAT
        should handle the containerised pipeline

        Returns
        -------
        dict
            XNAT container service command specification in JSON-like dict, which can be
            stored within the "org.nrg.commands" label of the container to allow the
            images to be automatically recognised.
        """

        cmd_json = self.init_command_json()

        input_args = self.add_input_fields(cmd_json)

        param_args = self.add_parameter_fields(cmd_json)

        output_args = self.add_output_fields(cmd_json)

        flag_arg = self.add_pipeline2app_flags_field(cmd_json)

        xnat_input_args = self.add_inputs_from_xnat(cmd_json)

        cmd_json["command-line"] = " ".join(
            self.image.activate_conda()
            + ["pipeline2app", "ext", "xnat", "cs-entrypoint", "xnat-cs//[PROJECT_ID]"]
            + input_args
            + output_args
            + param_args
            + xnat_input_args
            + [flag_arg]
        )

        return cmd_json

    def init_command_json(self) -> ty.Dict[str, ty.Any]:
        """Initialises the command JSON that specifies to the XNAT Cs how the command
        should be run

        Returns
        -------
        ty.Dict[str, *]
            the JSON-like dictionary to specify the command to the XNAT CS
        """

        cmd_json = {
            "name": self.name,
            "description": (f"{self.name} {self.image.version}: {self.image.title}"),
            "label": self.name,
            "schema-version": "1.0",
            "image": self.image.reference,
            "index": self.image.registry,
            "datatype": "docker",
            # "command-line": cmdline,
            "override-entrypoint": True,
            "mounts": [
                {"name": "in", "writable": False, "path": str(XnatViaCS.INPUT_MOUNT)},
                {"name": "out", "writable": True, "path": str(XnatViaCS.OUTPUT_MOUNT)},
                {  # Saves the Pydra-cache directory outside of the container for easier debugging
                    "name": "work",
                    "writable": True,
                    "path": str(XnatViaCS.WORK_MOUNT),
                },
            ],
            "ports": {},
            "inputs": [],  # inputs_json,
            "outputs": [],  # outputs_json,
            "xnat": [
                {
                    "name": self.name,
                    "description": self.image.title,
                    "contexts": [],  # context,
                    "external-inputs": [],  # external_inputs,
                    "derived-inputs": [],  # derived_inputs,
                    "output-handlers": [],  # output_handlers,
                }
            ],
        }

        return cmd_json

    def add_input_fields(self, cmd_json: ty.Dict[str, ty.Any]) -> ty.List[str]:
        """Adds pipeline inputs to the command JSON

        Parameters
        ----------
        cmd_json : dict
            JSON-like dictionary to be passed to the XNAT container service to specify
            how to run a command

        Returns
        -------
        list[str]
            list of arguments to be appended to the command line
        """
        # Add task inputs to inputs JSON specification
        cmd_args = []
        for inpt in self.inputs:
            replacement_key = f"[{inpt.field.upper()}_INPUT]"
            if issubclass(inpt.datatype, FileSet):
                if inpt.column_defaults.datatype:
                    datatype = inpt.column_defaults.datatype
                else:
                    datatype = inpt.datatype
                desc = f"Match resource ({to_mime(datatype)}) [SCAN-TYPE]: {inpt.help} "
                input_type = "string"
            else:
                desc = f"Match field ({inpt.datatype}) [FIELD-NAME]: {inpt.help} "
                input_type = self.COMMAND_INPUT_TYPES.get(inpt.datatype, "string")
            cmd_json["inputs"].append(
                {
                    "name": self.path2xnatname(inpt.name),
                    "description": desc,
                    "type": input_type,
                    "default-value": inpt.config_dict.get("path", ""),
                    "required": False,
                    "user-settable": True,
                    "replacement-key": replacement_key,
                }
            )
            cmd_args.append(f"--input {inpt.name} '{replacement_key}'")

        return cmd_args

    def add_parameter_fields(self, cmd_json: ty.Dict[str, ty.Any]) -> ty.List[str]:

        # Add parameters as additional inputs to inputs JSON specification
        cmd_args = []
        for param in self.parameters:
            desc = f"Parameter ({param.datatype}): " + param.help

            replacement_key = f"[{param.field.upper()}_PARAM]"

            cmd_json["inputs"].append(
                {
                    "name": param.name,
                    "description": desc,
                    "type": self.COMMAND_INPUT_TYPES.get(param.datatype, "string"),
                    "default-value": (param.default if param.default else ""),
                    "required": param.required,
                    "user-settable": True,
                    "replacement-key": replacement_key,
                }
            )
            cmd_args.append(f"--parameter {param.name} '{replacement_key}'")

        return cmd_args

    def add_output_fields(self, cmd_json: ty.Dict[str, ty.Any]) -> ty.List[str]:

        # Set up output handlers and arguments
        cmd_args = []
        for output in self.outputs:
            out_fname = output.name + (
                output.datatype.ext if output.datatype.ext else ""
            )

            desc = (
                f"Output ({to_mime(output.datatype, official=False)}): " + output.help
            )
            # Set the path to the
            if self.internal_upload:
                cmd_json["outputs"].append(
                    {
                        "name": output.name,
                        "description": desc,
                        "required": True,
                        "mount": "out",
                        "path": out_fname,
                        "glob": None,
                    }
                )
                cmd_json["xnat"][0]["output-handlers"].append(
                    {
                        "name": f"{output.name}-resource",
                        "accepts-command-output": output.name,
                        "via-wrapup-command": None,
                        "as-a-child-of": "SESSION",
                        "type": "Resource",
                        # Shame that the "label" output is fixed, would be good to append
                        # the "dataset_name" to it as we do in the API put. Might be worth
                        # just dropping XNAT outputs and just using API
                        "label": path2label(output.name),
                        "format": output.datatype.mime_like,
                    }
                )
                cmd_args.append(f"--output {output.name} '{output.name}'")
            else:
                replacement_key = f"[{output.field.upper()}_OUTPUT]"
                cmd_json["inputs"].append(
                    {
                        "name": output.name,
                        "description": desc,
                        "type": self.COMMAND_INPUT_TYPES.get(output.datatype, "string"),
                        "default-value": output.name,
                        "required": False,
                        "user-settable": True,
                        "replacement-key": replacement_key,
                    }
                )
                cmd_args.append(f"--output {output.name} '{replacement_key}'")

        if self.internal_upload:
            cmd_args.append("--internal-upload")

        return cmd_args

    def add_pipeline2app_flags_field(self, cmd_json: ty.Dict[str, ty.Any]) -> str:

        # Add input for dataset name
        FLAGS_KEY = "#PIPELINE2APP_FLAGS#"
        cmd_json["inputs"].append(
            {
                "name": "Pipeline2app_flags",
                "description": "Flags passed to `run-pipeline2app-pipeline` command",
                "type": "string",
                "default-value": (
                    "--plugin serial "
                    "--work /wl "  # noqa NB: work dir moved inside container due to file-locking issue on some mounted volumes (see https://github.com/tox-dev/py-filelock/issues/147)
                    "--dataset-name default "
                    "--loglevel info "
                    f"--export-work {XnatViaCS.WORK_MOUNT}"
                ),
                "required": False,
                "user-settable": True,
                "replacement-key": FLAGS_KEY,
            }
        )

        return FLAGS_KEY

    def add_inputs_from_xnat(self, cmd_json: ty.Dict[str, ty.Any]) -> ty.List[str]:

        # Define the fixed subject>session dataset hierarchy of XNAT, i.e. the data
        # tree contains two levels, one for subjects and the other for sessions
        cmd_args = ["--dataset-hierarchy subject,session"]

        # Create Project input that can be passed to the command line, which will
        # be populated by inputs derived from the XNAT object passed to the pipeline
        cmd_json["inputs"].append(
            {
                "name": "PROJECT_ID",
                "description": "Project ID",
                "type": "string",
                "required": True,
                "user-settable": False,
                "replacement-key": "[PROJECT_ID]",
            }
        )

        # Access session via Container service args and derive
        if self.row_frequency == Clinical.session:
            # Set the object the pipeline is to be run against
            cmd_json["xnat"][0]["contexts"] = ["xnat:imageSessionData"]
            # Create Session input that  can be passed to the command line, which
            # will be populated by inputs derived from the XNAT session object
            # passed to the pipeline.
            cmd_json["inputs"].extend(
                [
                    {
                        "name": "SESSION_LABEL",
                        "description": "Imaging session label",
                        "type": "string",
                        "required": True,
                        "user-settable": False,
                        "replacement-key": "[SESSION_LABEL]",
                    },
                    {
                        "name": "SUBJECT_LABEL",
                        "description": "Subject label",
                        "type": "string",
                        "required": True,
                        "user-settable": False,
                        "replacement-key": "[SUBJECT_LABEL]",
                    },
                ]
            )

            # Access the session XNAT object passed to the pipeline
            cmd_json["xnat"][0]["external-inputs"] = [
                {
                    "name": "SESSION",
                    "description": "Imaging session",
                    "type": "Session",
                    "source": None,
                    "default-value": None,
                    "required": True,
                    "replacement-key": None,
                    "sensitive": None,
                    "provides-value-for-command-input": None,
                    "provides-files-for-command-mount": "in",
                    "via-setup-command": None,
                    "user-settable": False,
                    "load-children": True,
                }
            ]
            # Access to project ID and session label from session XNAT object
            cmd_json["xnat"][0]["derived-inputs"] = [
                {
                    "name": "__SESSION_LABEL__",
                    "type": "string",
                    "derived-from-wrapper-input": "SESSION",
                    "derived-from-xnat-object-property": "label",
                    "provides-value-for-command-input": "SESSION_LABEL",
                    "user-settable": False,
                },
                {
                    "name": "__SUBJECT_ID__",
                    "type": "string",
                    "derived-from-wrapper-input": "SESSION",
                    "derived-from-xnat-object-property": "subject-id",
                    "provides-value-for-command-input": "SUBJECT_LABEL",
                    "user-settable": False,
                },
                {
                    "name": "__PROJECT_ID__",
                    "type": "string",
                    "derived-from-wrapper-input": "SESSION",
                    "derived-from-xnat-object-property": "project-id",
                    "provides-value-for-command-input": "PROJECT_ID",
                    "user-settable": False,
                },
            ]

            # Add specific session to process to command line args
            cmd_args.extend(
                [
                    "--ids [SESSION_LABEL]",
                    # "--single-row [SUBJECT_LABEL],[SESSION_LABEL]",
                ]
            )

        else:
            raise NotImplementedError(
                "Wrapper currently only supports session-level pipelines"
            )

        return cmd_args

    @classmethod
    def path2xnatname(cls, path: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]+", "_", path)

    COMMAND_INPUT_TYPES = {bool: "bool", str: "string", int: "number", float: "number"}
    VALID_FREQUENCIES = (Clinical.session, Clinical.constant)
