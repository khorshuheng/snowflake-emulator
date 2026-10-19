"""Small AST helpers shared by the translator modules."""

def _quoted_identifier(name: str) -> str:
    """Quote an identifier for use inside a DuckDB SQL string literal."""
    return '"' + name.replace('"', '""') + '"'
