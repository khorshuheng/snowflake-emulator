"""The core extraction keeps the old import locations working."""

from __future__ import annotations

from snowflake_emulator import errors, executor, statement_store, translator
from snowflake_emulator.core import ast, constants, errors as core_errors
from snowflake_emulator.core import results as core_results


def test_executor_reexports_result_and_error_types():
    assert executor.ExecutionResult is core_results.ExecutionResult
    assert executor.ColumnMeta is core_results.ColumnMeta
    assert executor.FileTransfer is core_results.FileTransfer
    assert executor.ExecutionError is core_errors.ExecutionError


def test_error_module_is_a_compatibility_shim():
    assert errors.ExecutionError is core_errors.ExecutionError
    assert errors.error_code_pair is core_errors.error_code_pair
    assert errors.classify_exception is core_errors.classify_exception


def test_translator_reexports_moved_helpers():
    assert translator.INFORMATION_SCHEMA_SCHEMA is constants.INFORMATION_SCHEMA_SCHEMA
    assert translator.drop_target_table is ast.drop_target_table


def test_statement_store_uses_the_core_result_type():
    store = statement_store.StatementStore(1)
    result = core_results.ExecutionResult(
        handle="h", created_on=0, row_type=[], rows=[]
    )
    store.put(result)
    assert store.get("h") is result
