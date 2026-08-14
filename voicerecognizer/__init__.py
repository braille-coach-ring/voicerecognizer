"""voicerecognizer - top-level package exposing all project subpackages.

Usage:
    import voicerecognizer
    voicerecognizer.core.factory ...

    # or directly
    from voicerecognizer import core
"""

import importlib
import sys

_SUBPACKAGES = [
    "core",
    "dataset",
    "evaluation",
    "models",
    "preprocessing",
    "recognizers",
    "runtime",
    "utils",
]

for _pkg in _SUBPACKAGES:
    _mod = importlib.import_module(_pkg)
    setattr(sys.modules[__name__], _pkg, _mod)
    # Also register as voicerecognizer.<pkg> in sys.modules
    sys.modules[f"voicerecognizer.{_pkg}"] = _mod

__all__ = _SUBPACKAGES
