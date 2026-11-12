"""Cross-process definition-store tests for either supported DuckDB CLI.

Run with YARDSTICK_DUCKDB_CLI and YARDSTICK_EXTENSION set to built artifacts:
    uv run test/python/test_definition_store.py
Set YARDSTICK_DUCKLAKE_EXTENSION to also exercise a real DuckLake attachment.
"""

import csv
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


def sql_string(value):
    return "'" + str(value).replace("'", "''") + "'"


class DefinitionStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cli = os.environ.get("YARDSTICK_DUCKDB_CLI")
        extension = os.environ.get("YARDSTICK_EXTENSION")
        if not cli or not extension:
            raise unittest.SkipTest("Set YARDSTICK_DUCKDB_CLI and YARDSTICK_EXTENSION")
        cls.cli = str(Path(cli).resolve())
        cls.extension = str(Path(extension).resolve())

    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="yardstick-definition-tests-")
        self.addCleanup(self.scratch.cleanup)
        self.store = Path(self.scratch.name) / "definitions.duckdb"

    def run_sql(self, sql, *, readonly=False, ducklake=False, expect_error=False):
        args = [self.cli, "-unsigned", "-batch", "-csv", "-noheader", "-init", os.devnull]
        args += ["-cmd", f"LOAD {sql_string(self.extension)}"]
        if ducklake:
            extension = os.environ.get("YARDSTICK_DUCKLAKE_EXTENSION")
            if not extension:
                self.skipTest("Set YARDSTICK_DUCKLAKE_EXTENSION")
            args += ["-cmd", f"LOAD {sql_string(Path(extension).resolve())}"]
        path = f"ducklake:{self.store}" if ducklake else str(self.store)
        attach = f"ATTACH {sql_string(path)} AS store"
        if readonly:
            attach += " (READ_ONLY)"
        args += ["-cmd", attach, "-c", sql]
        environment = os.environ.copy()
        environment.pop("CODEX_CI", None)  # Keep the CLI's ordinary CSV output.
        result = subprocess.run(args, text=True, capture_output=True, env=environment, timeout=60)
        if expect_error:
            self.assertNotEqual(result.returncode, 0, result.stdout)
            return result.stderr
        self.assertEqual(result.returncode, 0, result.stderr)
        return list(csv.reader(io.StringIO(result.stdout)))

    def create_sales_view(self, *, ducklake=False):
        self.run_sql(
            "create table store.main.sales(region varchar, amount integer);"
            "insert into store.main.sales values ('N', 10), ('N', 20), ('S', 40);"
            "set yardstick_definition_table='store.main.definitions';"
            "create view sales_v as select region, sum(amount) as measure revenue from store.main.sales;",
            ducklake=ducklake,
        )

    def test_fresh_process_and_readonly_attachment(self):
        self.create_sales_view()
        rows = self.run_sql(
            "set yardstick_definition_table='store.main.definitions';"
            "select aggregate(revenue) from sales_v;"
            "select count(*) from duckdb_views() where view_name='sales_v';",
            readonly=True,
        )
        self.assertEqual(rows, [["70"], ["0"]])

    def test_rollback_and_committed_drop(self):
        self.create_sales_view()
        rows = self.run_sql(
            "set yardstick_definition_table='store.main.definitions';"
            "begin;"
            "create or replace view sales_v as select region, sum(amount * 2) as measure revenue from store.main.sales;"
            "select aggregate(revenue) from sales_v;"
            "rollback;"
            "select aggregate(revenue) from sales_v;"
            "drop view sales_v;"
            "select count(*) from store.main.definitions where view_name<>'';"
        )
        self.assertEqual(rows, [["140"], ["70"], ["0"]])
        error = self.run_sql(
            "set yardstick_definition_table='store.main.definitions'; select aggregate(revenue) from sales_v;",
            expect_error=True,
        )
        self.assertIn("sales_v", error)

    def test_native_queries_do_not_scan_definitions(self):
        # A scan would fail on the missing definition columns. Native-table and
        # constant queries should not touch this deliberately unrelated table.
        rows = self.run_sql(
            "create table store.main.unrelated(value integer);"
            "insert into store.main.unrelated values (41);"
            "set yardstick_definition_table='store.main.unrelated';"
            "select value + 1 from store.main.unrelated;"
            "select 7;"
        )
        self.assertEqual(rows, [["42"], ["7"]])

    def test_explain_does_not_create_a_second_definition(self):
        self.create_sales_view()
        rows = self.run_sql(
            "set yardstick_definition_table='store.main.definitions';"
            "explain analyze create or replace view sales_v as select 17 as value;"
            "select value from sales_v;"
            "select count(*) from duckdb_views() where view_name='sales_v';"
        )
        self.assertEqual(rows[-2:], [["17"], ["0"]])

    def test_prepared_parameters_and_replacement(self):
        self.create_sales_view()
        rows = self.run_sql(
            "set yardstick_definition_table='store.main.definitions';"
            "prepare sales_total as select aggregate(revenue) from sales_v where region=$1;"
            "execute sales_total('N');"
            "create or replace view sales_v as "
            "select region, sum(amount * 2) as measure revenue from store.main.sales;"
            "execute sales_total('N');"
            "drop view sales_v;"
            "select count(*) from store.main.definitions where view_name<>'';"
        )
        self.assertEqual(rows, [["30"], ["60"], ["0"]])

    def test_drop_respects_temporary_view_shadowing(self):
        self.create_sales_view()
        rows = self.run_sql(
            "set yardstick_definition_table='store.main.definitions';"
            "create temp view sales_v as select 1 as value;"
            "drop view sales_v;"
            "select count(*) from duckdb_views() where temporary and view_name='sales_v';"
            "select count(*) from store.main.definitions where view_name<>'';"
            "select aggregate(revenue) from sales_v;"
            "create temp view sales_v as select 2 as value;"
            "drop view memory.main.sales_v;"
            "select value from sales_v;"
            "select count(*) from store.main.definitions where view_name<>'';"
            "drop view temp.main.sales_v;"
        )
        self.assertEqual(rows, [["0"], ["1"], ["70"], ["2"], ["0"]])

    def test_nested_ctes_do_not_redirect_stored_sources(self):
        self.run_sql(
            "create schema store.original; create schema store.other;"
            "create table store.original.sales as select 7 amount;"
            "create table store.other.sales as select 99 amount;"
            "set schema='store.original';"
            "set yardstick_definition_table='store.main.definitions';"
            "create view store.main.nested_cte as "
            "select sum(amount) as measure total from sales "
            "where exists (with sales as (select 1) select * from sales);"
            "create view store.main.derived_cte as "
            "select sum(amount) as measure total from sales, "
            "(with sales as (select 1 marker) select * from sales) nested where nested.marker=1;"
            "create view store.main.self_named_cte as "
            "with sales as (select amount from sales), copy as (select * from sales) "
            "select sum(amount) as measure total from copy;"
        )
        rows = self.run_sql(
            "set schema='store.other';"
            "set yardstick_definition_table='store.main.definitions';"
            "select aggregate(total) from store.main.nested_cte;"
            "select aggregate(total) from store.main.derived_cte;"
            "select aggregate(total) from store.main.self_named_cte;"
        )
        self.assertEqual(rows, [["7"], ["7"], ["7"]])

    def test_commented_definitions_with_same_name_in_different_schemas(self):
        self.run_sql(
            "create schema store.first_schema; create schema store.second_schema;"
            "create table store.main.sales as select 7 amount;"
            "set yardstick_definition_table='store.main.definitions';"
            "/* Keep the declaration comment. */ create view store.first_schema.same_name as "
            "select sum(amount) as measure total from store.main.sales;"
            "create view store.second_schema.same_name as "
            "select sum(amount * 3) as measure total from store.main.sales;"
        )
        rows = self.run_sql(
            "set yardstick_definition_table='store.main.definitions';"
            "select aggregate(total) from store.first_schema.same_name;"
            "select aggregate(total) from store.second_schema.same_name;"
        )
        self.assertEqual(rows, [["7"], ["21"]])

    def test_existing_table_guard_initialization_preserves_definitions(self):
        self.create_sales_view()
        rows = self.run_sql(
            "delete from store.main.definitions where view_name='';"
            "set yardstick_definition_table='store.main.definitions';"
            "select aggregate(revenue) from sales_v;"
        )
        self.assertEqual(rows, [["70"]])
        error = self.run_sql(
            "set yardstick_definition_table='store.main.definitions'; drop view sales_v;",
            expect_error=True,
        )
        self.assertIn("exclusive writer access", error)
        rows = self.run_sql(
            "insert into store.main.definitions values ('', '', '', 'yardstick_guard_v1:' || uuid()::varchar);"
            "set yardstick_definition_table='store.main.definitions';"
            "create or replace view sales_v as "
            "select region, sum(amount * 2) as measure revenue from store.main.sales;"
            "select aggregate(revenue) from sales_v;"
        )
        self.assertEqual(rows, [["140"]])

    def test_invalid_or_duplicate_guard_is_rejected(self):
        self.create_sales_view()
        self.run_sql("update store.main.definitions set definition_sql='invalid' where view_name='';")
        query = "set yardstick_definition_table='store.main.definitions'; select aggregate(revenue) from sales_v;"
        error = self.run_sql(query, expect_error=True)
        self.assertIn("Invalid or duplicate Yardstick concurrency guard", error)
        self.run_sql(
            "update store.main.definitions set definition_sql='yardstick_guard_v1:' || uuid()::varchar "
            "where view_name='';"
            "insert into store.main.definitions values ('', '', '', 'yardstick_guard_v1:' || uuid()::varchar);"
        )
        error = self.run_sql(query, expect_error=True)
        self.assertIn("Invalid or duplicate Yardstick concurrency guard", error)

    def test_source_binding_survives_schema_changes(self):
        rows = self.run_sql(
            "create schema original; create schema other;"
            "create table original.sales as select 7 amount;"
            "create table other.sales as select 99 amount;"
            "set schema='original';"
            "set yardstick_definition_table='store.main.definitions';"
            "create view main.sales_v as select sum(amount) as measure total from sales;"
            "set schema='other';"
            "select aggregate(total) from main.sales_v;"
        )
        self.assertEqual(rows, [["7"]])
        rows = self.run_sql(
            "create schema other; create table other.sales as select 99 amount;"
            "set schema='other'; set yardstick_definition_table='store.main.definitions';"
            "select aggregate(total) from main.sales_v;",
            expect_error=True,
        )
        # The definition must retain its original source, which belonged to
        # the previous in-memory database, rather than silently reading 99.
        self.assertIn("original", rows)

    def test_ducklake_lifecycle_and_reopen(self):
        self.create_sales_view(ducklake=True)
        rows = self.run_sql(
            "set yardstick_definition_table='store.main.definitions';"
            "select aggregate(revenue) from sales_v;"
            "begin;"
            "create or replace view sales_v as select region, sum(amount * 3) as measure revenue from store.main.sales;"
            "select aggregate(revenue) from sales_v;"
            "rollback; select aggregate(revenue) from sales_v;"
            "create or replace view sales_v as select region, sum(amount * 2) as measure revenue from store.main.sales;"
            "select aggregate(revenue) from sales_v;"
            "drop view sales_v; select count(*) from store.main.definitions where view_name<>'';",
            ducklake=True,
        )
        self.assertEqual(rows, [["70"], ["210"], ["70"], ["140"], ["0"]])


if __name__ == "__main__":
    unittest.main()
