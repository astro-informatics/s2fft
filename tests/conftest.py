"""
Collection of shared fixtures.

To avoid having a single, long ``conftest.py`` file, ``S2FFT`` makes use of "local" pytest plugins
to provide fixtures that can be used across the entire test suite, but organised into (sub)directories
and smaller files.

The ``tests/fixtures`` directory (and subdirectories therein) are examined by ``pytest`` on test
discovery, by virtue of how the ``pytest_plugins`` variable is set below. This effectively causes
``pytest`` to read the content of all ``*.py`` files in the ``fixtures`` directory, and thus load
any fixtures or command-line options in them as if they had been defined inside this file. In turn,
this ensures any such fixtures are available for use within the entire test suite.
"""

from pathlib import Path

_THIS_DIR = Path(__file__).parent.resolve()


def _to_module_string(path: str) -> str:
    """Convert a file path to a module string."""
    return path.replace("/", ".").replace("\\", ".").replace(".py", "")


pytest_plugins = [
    _to_module_string(fixture.relative_to(_THIS_DIR.parent).as_posix())
    for fixture in (Path(f"{_THIS_DIR}/fixtures").rglob("*.py"))
    if "__init__" not in str(fixture)
]
