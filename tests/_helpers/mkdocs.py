"""Read ``mkdocs.yml`` as data for the docs tests.

Two test modules check the site configuration: ``tests/docs/test_mkdocs.py``
and ``tests/docs/test_docstring_format.py``. Both read the list of packages the
API reference renders, so they share this reader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]


def mkdocs_config() -> dict[str, Any]:
    """``mkdocs.yml`` as data; its ``!!python/name`` tags load as plain strings."""
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_multi_constructor(
        "tag:yaml.org,2002:python/", lambda loader, suffix, node: suffix
    )
    return yaml.load((REPO / "mkdocs.yml").read_text(), Loader=Loader)


def api_reference_packages() -> list[str]:
    """The packages the API reference renders (``extra.api_reference``)."""
    return list(mkdocs_config()["extra"]["api_reference"])
