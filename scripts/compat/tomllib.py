"""Compatibility wrapper for Python 3.10.

Python 3.11 added tomllib to the stdlib. Cosmos Framework imports tomllib
directly, so this opt-in module forwards to tomli under Python 3.10.
"""

from tomli import *  # noqa: F401,F403

