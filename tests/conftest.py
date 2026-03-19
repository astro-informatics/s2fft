"""Collection of shared fixtures"""

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
