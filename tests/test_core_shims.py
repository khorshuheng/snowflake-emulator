"""The core extraction keeps the old import locations working.

These are identity checks, not behaviour: they fail if a re-export that other modules (and
downstream importers) still rely on is dropped from its former module.
"""

from __future__ import annotations

from snowflake_emulator import errors, executor, statement_store, translator
from snowflake_emulator.core import ast, constants, errors as core_errors
from snowflake_emulator.core import results as core_results


def test_the_core_extraction_keeps_its_former_import_locations():
    # executor re-exports the result and error types it used to define.
    assert executor.ExecutionResult is core_results.ExecutionResult
    assert executor.ColumnMeta is core_results.ColumnMeta
    assert executor.FileTransfer is core_results.FileTransfer
    assert executor.ExecutionError is core_errors.ExecutionError
    # snowflake_emulator.errors is a compatibility shim.
    assert errors.ExecutionError is core_errors.ExecutionError
    assert errors.error_code_pair is core_errors.error_code_pair
    assert errors.classify_exception is core_errors.classify_exception
    # translator re-exports the helpers that moved into core.
    assert translator.INFORMATION_SCHEMA_SCHEMA is constants.INFORMATION_SCHEMA_SCHEMA
    assert translator.drop_target_table is ast.drop_target_table

    store = statement_store.StatementStore(1)
    result = core_results.ExecutionResult(handle="h", created_on=0, row_type=[], rows=[])
    store.put(result)
    assert store.get("h") is result
