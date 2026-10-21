"""Translation failure type shared by the translator modules."""


class TranslationError(ValueError):
    """Raised when a Snowflake statement cannot be parsed or transpiled.

    Most translation failures are reported as Snowflake's generic SQL compilation error
    (``001003`` / ``42000``). A rewrite that reproduces a specific Snowflake compile-time
    rejection can carry that condition's own code pair instead.
    """

    def __init__(
        self,
        message: str,
        *,
        errno: str | None = None,
        sqlstate: str | None = None,
    ) -> None:
        super().__init__(message)
        self.errno = errno
        self.sqlstate = sqlstate
