"""safetensors, read and written the way the machine underneath can go fastest.

The public API mirrors ``safetensors.torch`` — ``save_file``, ``load_file``,
``save``, ``load`` and ``safe_open`` — in [`causalab.io.fastersafetensors.torch`][].
Files it writes are byte-identical to the reference library's; files it reads
are the reference library's. What differs is the path the bytes take, chosen
per machine: how many shards in flight, GPUDirect Storage or pinned staging,
mmap or pread. ``explain()`` prints that choice.
"""

from __future__ import annotations

from . import errors

__all__ = ["errors"]
