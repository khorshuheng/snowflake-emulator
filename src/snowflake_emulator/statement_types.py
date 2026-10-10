"""Snowflake ``statementTypeId`` values.

Snowflake tags every query response with the type of the statement, and clients
branch on it: ``snowflake-connector-python`` checks whether the id is in the DML
set to decide how to report affected rows, and the JDBC driver rejects
``executeUpdate`` for statements it cannot identify. The ids mirror
``net.snowflake.client.core.SFStatementType`` and the connector's
``STATEMENT_TYPE_ID_DML_SET``.
"""

from __future__ import annotations

from sqlglot import exp

UNKNOWN = 0x0000
SELECT = 0x1000
DML = 0x3000
INSERT = DML + 0x100
UPDATE = DML + 0x200
DELETE = DML + 0x300
MERGE = DML + 0x400
MULTI_INSERT = DML + 0x500
COPY = DML + 0x600
SCL = 0x4000
USE = SCL + 0x300
SHOW = SCL + 0x400
DESCRIBE = SCL + 0x500
TCL = 0x5000
DDL = 0x6000

# Statements clients treat as DML: they report affected rows, not a result set.
DML_TYPE_IDS = frozenset({DML, INSERT, UPDATE, DELETE, MERGE, MULTI_INSERT})

_EXPRESSION_TYPES: dict[type[exp.Expression], int] = {
    exp.Select: SELECT,
    exp.Union: SELECT,
    exp.Subquery: SELECT,
    exp.Insert: INSERT,
    exp.MultitableInserts: MULTI_INSERT,
    exp.Update: UPDATE,
    exp.Delete: DELETE,
    exp.Merge: MERGE,
    exp.Copy: COPY,
    exp.Show: SHOW,
    exp.Describe: DESCRIBE,
    exp.Create: DDL,
    exp.Drop: DDL,
    exp.Alter: DDL,
    exp.TruncateTable: DDL,
    exp.Commit: TCL,
    exp.Rollback: TCL,
    exp.Transaction: TCL,
}


def statement_type_id(statement: exp.Expression | None) -> int:
    """Return the Snowflake statement type id for a parsed statement.

    Unknown statements return :data:`UNKNOWN`, which is what Snowflake reports for
    statements that produce a result set.
    """
    if statement is None:
        return UNKNOWN
    if isinstance(statement, exp.Use):
        return USE
    return _EXPRESSION_TYPES.get(type(statement), UNKNOWN)


def number_of_binds(statement: exp.Expression | None) -> int:
    """Return the number of bind placeholders in a statement."""
    if statement is None:
        return 0
    return len(list(statement.find_all(exp.Placeholder)))
