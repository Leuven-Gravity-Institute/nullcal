from __future__ import annotations

import os
import sys

import bilby_pipe.utils
from bilby_pipe.bilbyargparser import BilbyArgParser, HyphenStr
from bilby_pipe.parser import create_parser

from .._version import __version__
from ..utils import logger

bilby_pipe.utils.logger = logger


def write_to_file(
    self,
    filename,
    args=None,
    overwrite=False,
    include_description=False,
    exclude_default=False,
    comment=None,
):
    if os.path.isfile(filename) and not overwrite:
        logger.warning(f"File {filename} already exists, not writing to file.")
    with open(filename, "w") as ff:
        if include_description:
            print(
                f"## This file was written with nullcal version {__version__}\n",
                file=ff,
            )
        if isinstance(comment, str):
            print("#" + comment + "\n", file=ff)
        for group in self._action_groups[2:]:
            print("#" * 80, file=ff)
            print(f"## {group.title}", file=ff)
            if include_description:
                print(f"# {group.description}", file=ff)
            print("#" * 80 + "\n", file=ff)
            for action in group._group_actions:
                if include_description:
                    print(f"# {action.help}", file=ff)
                dest = action.dest
                hyphen_dest = HyphenStr(dest)
                if isinstance(args, dict):
                    if action.dest in args:
                        value = args[dest]
                    elif hyphen_dest in args:
                        value = args[hyphen_dest]
                    else:
                        value = action.default
                else:
                    value = getattr(args, dest, action.default)

                if exclude_default and value == action.default:
                    continue
                self.write_comment_if_needed(hyphen_dest, ff)
                self.write_line(hyphen_dest, value, ff)
            print("", file=ff)


BilbyArgParser.write_to_file = write_to_file


def create_nullcal_parser(top_level=True):
    """Create the nullpol_pipe parser

    Parameters
    ----------
    top_level: bool, optional
        If true, the top-level parser is created. If false, a subparser is
        created. Default is True.
    usage: str, optional
        The usage string to display. Default is None.

    Returns
    -------
    parser: BilbyArgParser instance
        Argument parser
    """
    def remove_argument(parser, arg):
        action_to_remove = None
        for action in parser._actions:
            opts = action.option_strings
            if (opts and opts[0] == arg) or action.dest == arg:
                parser._remove_action(action)
                action_to_remove = action
                break
        # Remove from _option_string_actions
        if action_to_remove:
            for option_string in action_to_remove.option_strings:
                if option_string in parser._option_string_actions:
                    del parser._option_string_actions[option_string]
        for action in parser._action_groups:
            for group_action in action._group_actions:
                opts = group_action.option_strings
                if (opts and opts[0] == arg) or group_action.dest == arg:
                    action._group_actions.remove(group_action)
                    return

    def add_argument_to_group(parser, group_name, *args, **kwargs):
        # Locate the group by name
        for grp in parser._action_groups:
            if grp.title == group_name:
                grp.add_argument(*args, **kwargs)
                return
        raise ValueError(f"Argument group '{group_name}' not found")

    parser = create_parser(top_level=top_level)
    remove_argument(parser, "--coherence-test")
    remove_argument(parser, "--calibration-marginalization")
    remove_argument(parser, "--calibration-lookup-table")
    remove_argument(parser, "--number-of-response-curves")
    remove_argument(parser, "--distance-marginalization")
    remove_argument(parser, "--distance-marginalization-lookup-table")
    remove_argument(parser, "--phase-marginalization")
    remove_argument(parser, "--time-marginalization")
    remove_argument(parser, "--jitter-time")
    remove_argument(parser, "--roq-folder")
    remove_argument(parser, "--roq-linear-matrix")
    remove_argument(parser, "--roq-quadratic-matrix")
    remove_argument(parser, "--roq-weights")
    remove_argument(parser, "--roq-weight-format")
    remove_argument(parser, "--roq-scale-factor")
    remove_argument(parser, "--fiducial-parameters")
    remove_argument(parser, "--update-fiducial-parameters")
    remove_argument(parser, "--epsilon")
    remove_argument(parser, "--default-prior")
    remove_argument(parser, "--version")
    parser.add("--version",
               action="version",
               version=f"%(prog)s={__version__}")
    return parser


def main():
    filename = sys.argv[1]
    if filename in ["-h", "--help"]:
        logger.info("Write a default config.ini file to the specified filename.")
        logger.info("Example usage: $ nullcal_pipe_write_default_ini config.ini")
        sys.exit()
    else:
        parser = create_nullcal_parser()
        logger.info(f"Default config file written to {os.path.abspath(filename)}")
        parser.write_to_file(
            filename=filename, overwrite=True, include_description=True
        )
