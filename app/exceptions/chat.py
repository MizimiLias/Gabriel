"""Chat domain exceptions."""


class ChatError(Exception):
    """Base class for expected chat failures."""


class ChatSessionNotFoundError(ChatError):
    """Raised when a requested chat session does not exist."""


class ChatUpstreamError(ChatError):
    """Raised when the configured language model cannot complete a request."""


class DatabaseUnavailableError(ChatError):
    """Raised when a required database check fails."""


class PersonaNotFoundError(ChatError):
    """Raised when a requested persona does not exist."""


class PersonaConflictError(ChatError):
    """Raised when a persona name is already in use."""


class PersonaValidationError(ChatError):
    """Raised when persona input violates the configured limits."""
