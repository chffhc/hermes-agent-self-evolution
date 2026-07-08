"""Domain exceptions for the evolution library.

Library-level evolution functions raise :class:`EvolutionError` instead of
calling ``sys.exit()`` so that programmatic callers (Phase 5 continuous
evolution, tests, notebooks) can handle failures without the process dying.
CLI wrappers catch it and translate to a non-zero exit code.
"""


class EvolutionError(RuntimeError):
    """Raised when an evolution run cannot proceed or must abort safely."""
