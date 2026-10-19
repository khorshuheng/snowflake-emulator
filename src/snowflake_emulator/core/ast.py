"""Small AST helpers shared by translation, staging and execution.

Only sqlglot is imported here, so this module has no dependency on the emulator's
database or execution layers.
"""

from __future__ import annotations

from sqlglot import exp


def drop_target_table(statement: exp.Drop) -> exp.Table | None:
    """Return the table a ``DROP`` targets.

    Returns ``None`` when there is no single table target (e.g. a multi-target drop,
    which the emulator does not emulate).
    """
    tables = statement.args.get("tables")
    if tables:
        target = tables[0]
        return target if isinstance(target, exp.Table) else None
    return None
