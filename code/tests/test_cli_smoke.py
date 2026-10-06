import importlib
import pkgutil

import medrag
from medrag.cli import COMMANDS, build_parser


def test_every_module_imports():
    for m in pkgutil.walk_packages(medrag.__path__, "medrag."):
        importlib.import_module(m.name)


def test_every_subcommand_is_wired():
    sub = next(a for a in build_parser()._actions if a.dest == "cmd")
    assert set(sub.choices) == set(COMMANDS)
