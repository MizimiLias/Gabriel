"""Chat domain exceptions."""


class ChatError(Exception):
    """Base class for expected chat failures."""


class ChatSessionNotFoundError(ChatError):
    """Raised when a requested chat session does not exist."""


class ChatUpstreamError(ChatError):
    """Raised when the configured language model cannot complete a request."""


class DatabaseUnavailableError(ChatError):
    """Raised when a required database check fails."""
