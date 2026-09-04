"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from pathlib import Path

import pytest

COPYRIGHT_NOTICES: tuple[str, ...] = (
    '"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.',
    '"""Copyright (c) RWE Supply & Trading GmbH. Licensed under the MIT license.',
)
"""tuple[str, ...]: Every file must start with one of these notices.

This is a fork of `microsoft/aurora`. Files retained from upstream keep the Microsoft
notice, as the MIT license requires. Files authored here carry the RWE notice.
"""

PYTHON_FILES: list[Path] = []
"""list[Path]: Python files to scan for headers."""

_root = Path(__file__).parents[1]
for path in _root.rglob("**/*.py"):
    relative_path = path.relative_to(_root)

    # Ignore a possible virtual environment.
    if len(relative_path.parents) >= 2 and str(relative_path.parents[-2]) in {"venv", ".venv"}:
        continue

    # Ignore the automatically generated version file.
    if relative_path.name in {"_version.py"}:
        continue

    PYTHON_FILES.append(path)


@pytest.mark.parametrize("python_file", PYTHON_FILES)
def test_presence_of_copyright_header(python_file: Path) -> None:
    with open(python_file) as f:
        lines = list(f.read().splitlines())

    # An executable script may lead with a shebang, which must stay on the first line.
    if lines and lines[0].startswith("#!"):
        lines = lines[1:]

    if not lines or not any(lines[0].startswith(notice) for notice in COPYRIGHT_NOTICES):
        raise AssertionError(f"`{python_file}` must start with a copyright notice.")
