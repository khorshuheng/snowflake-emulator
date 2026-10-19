"""Translation failure type shared by the translator modules."""


class TranslationError(ValueError):
    """Raised when a Snowflake statement cannot be parsed or transpiled."""
