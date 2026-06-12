"""Python 3.10 compatibility shims for local Cosmos Framework preflight.

The upstream framework imports a few Python 3.11 stdlib names while this
project venv is Python 3.10. Keep the shim opt-in through PYTHONPATH so it does
not affect unrelated project code.
"""

import typing

from typing_extensions import Self, override

if not hasattr(typing, "Self"):
    typing.Self = Self

if not hasattr(typing, "override"):
    typing.override = override
