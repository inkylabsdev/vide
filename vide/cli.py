"""Click entry point. Auto-registers every command in vide.commands.*.

Each module under vide.commands must expose a module-level `cli`
click.Command; it gets registered under the command's own name.
"""

import importlib
import pkgutil

import click

from vide import commands


@click.group()
def cli():
    """vide — a video editing utility."""


for _mod_info in pkgutil.iter_modules(commands.__path__):
    _mod = importlib.import_module(f"{commands.__name__}.{_mod_info.name}")
    _command = getattr(_mod, "cli", None)
    if isinstance(_command, click.Command):
        cli.add_command(_command)


if __name__ == "__main__":
    cli()
