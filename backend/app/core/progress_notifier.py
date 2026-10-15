"""
Thread-safe, context-local progress notification mechanism for long-running workflows.
Allows async and worker-thread tasks to emit real-time progress updates without
coupling to specific transport layers or mutating global state.
"""

from contextvars import ContextVar
from typing import Callable, Optional

_progress_notifier: ContextVar[Optional[Callable[[str, Optional[str]], None]]] = ContextVar(
    "_progress_notifier", default=None
)


def set_progress_notifier(callback: Optional[Callable[[str, Optional[str]], None]]):
    """
    Sets the progress notifier callback for the current context.
    Returns a token that can be passed to reset_progress_notifier.
    """
    return _progress_notifier.set(callback)


def reset_progress_notifier(token) -> None:
    """Resets the progress notifier callback for the current context."""
    if token is not None:
        try:
            _progress_notifier.reset(token)
        except Exception:
            pass


def emit_progress(message: str, detail: Optional[str] = None) -> None:
    """
    Emits a progress event if a notifier is registered in the current context.
    Guaranteed never to raise exceptions or disrupt primary business logic.
    """
    notifier = _progress_notifier.get()
    if notifier:
        try:
            notifier(message, detail)
        except Exception:
            # Progress updates must never fail the underlying execution
            pass
