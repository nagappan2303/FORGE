"""AAM mapper sub-package.

`reaction_mapper`, `MapIsing`, and `optim_wrapper` were originally laid out
as top-level scripts on PYTHONPATH and use bare `import MapIsing` /
`import optim_wrapper` / `from optim.core.problem import ...` statements.
To keep the sub-package importable as `aam_src.reaction_mapper` without
rewriting those imports, we insert this directory at the front of
sys.path on package init.
"""
import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
