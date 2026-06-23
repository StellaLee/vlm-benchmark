"""Importing this package registers all prompting strategies."""

from avbench.inference.strategies import direct  # noqa: F401
from avbench.inference.strategies import verbal_confidence  # noqa: F401
from avbench.inference.strategies import consistency  # noqa: F401
from avbench.inference.strategies import self_reflection  # noqa: F401
from avbench.inference.strategies import vl_uncertainty  # noqa: F401
