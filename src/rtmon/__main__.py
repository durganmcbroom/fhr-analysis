"""Entry point. Also survives being run by path, which is how an IDE does it.

``python src/rtmon/__main__.py`` puts *this* directory on ``sys.path[0]``, and once
``src/rtmon`` is importable as a top level, ``rtmon/models.py`` answers to the bare name
``models``. That is the name ``lib/neossnet/utils/__init__.py`` imports ``MaskNet``
from, and neossnet is pulled in transitively (``analyze.util`` -> ``utils``), so the
server dies during import with

    ImportError: cannot import name 'MaskNet' from 'models' (.../rtmon/models.py)

which names neither the script that was run nor the shadowing that caused it. Swapping
that entry for the package's parent makes running this file by path behave exactly like
``python -m rtmon``, so the two ways in cannot diverge.
"""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0]) == _here:
    sys.path[0] = os.path.dirname(_here)

from rtmon.server import main  # noqa: E402 - must follow the path fix above

if __name__ == "__main__":
    main()
