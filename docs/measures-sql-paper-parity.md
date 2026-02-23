# Measures in SQL Paper Parity Matrix

Source paper: `arXiv:2406.00251v2` ("Measures in SQL", Jan 10, 2025), local copy at `.context/measures_in_sql.txt`.

## Scope

This matrix tracks parity for the core language semantics described in sections 3-5 of the paper.

- `Covered`: behavior is validated by existing automated tests.
- `Partial`: behavior is exercised, but not with an explicit parity assertion.
- `Gap`: no direct automated validation yet.

## Matrix

| Paper ref | Requirement | Status | Evidence |
|---|---|---|---|
| §3.2, Listing 3 | `AS MEASURE` in `CREATE VIEW`; no `GROUP BY` keeps base row cardinality | Covered | `test/sql/measures.test:20`, `test/sql/measures.test:1409` |
| §3.3 | `AGGREGATE(measure)` expansion/evaluation in grouped queries | Covered | `test/sql/measures.test:30`, `test/sql/measures.test:459` |
| Table 3 (`ALL`) | `AT (ALL)` removes all filters (grand total) | Covered | `test/sql/measures.test:152`, `test/sql/measures.test:444` |
| Table 3 (`ALL dim`) | `AT (ALL dim)` removes one dimension from context | Covered | `test/sql/measures.test:83`, `test/sql/measures.test:292` |
| Table 3 (`ALL dim1 dim2`) | single-clause multi-dimension `ALL` semantics | Covered | `test/sql/measures.test:1443` |
| Table 3 (modifier sequence) | chained modifiers execute right-to-left | Covered | `test/sql/measures.test:233`, `test/sql/measures.test:548` |
| Table 3 (`SET`) | `AT (SET dim = expr)` changes one dimension, correlates on others | Covered | `test/sql/measures.test:189`, `test/sql/measures.test:975` |
| Table 3 (`SET` + lost rows) | `SET` can reach rows removed by outer `WHERE` | Covered | `test/sql/measures.test:962` |
| Table 3 (`CURRENT`) | `CURRENT` resolves from single-valued context and returns `NULL` otherwise | Covered | `test/sql/measures.test:1665`, `test/sql/measures.test:1675` |
| Table 3 (`WHERE`) | `AT (WHERE predicate)` sets evaluation predicate | Covered | `test/sql/measures.test:165`, `test/sql/measures.test:341` |
| Table 3 (`WHERE`) | qualified refs and nested function predicates in `AT (WHERE ...)` | Covered | `test/sql/measures.test:178`, `test/sql/measures.test:1487` |
| Table 3 (`VISIBLE`) | `AT (VISIBLE)` respects current query visibility | Covered | `test/sql/measures.test:218`, `test/sql/measures.test:329` |
| §3.5 (ad hoc dims) | expression dimensions in `ALL`/`SET` (`MONTH(order_date)`, etc.) | Covered | `test/sql/measures.test:818`, `test/sql/measures.test:824`, `test/sql/measures.test:834` |
| Listing 8 | rollup query with `AGGREGATE`, plain measure ref, and `AT (VISIBLE)` | Covered | `test/sql/measures.test:1530` |
| Listing 9 | joins: weighted aggregate vs measure semantics vs `VISIBLE` | Covered | `test/sql/measures.test:1582`, `test/sql/measures.test:1458` |
| Listing 12 (queries 1-4) | correlated subquery, self-join, window, and measure forms return same rows | Covered | `test/sql/measures.test:1614`, `test/sql/measures.test:1624`, `test/sql/measures.test:1637`, `test/sql/measures.test:1652` |
| §5.1 claim | `AT` can access rows excluded by outer `WHERE` (more expressive than `OVER`) | Covered | `test/sql/measures.test:962` |
| §5.4 composability | derived measures referencing measures in same `SELECT` | Covered | `test/sql/measures.test:772`, `test/sql/measures.test:1499` |
| §5.3 wide-table safety direction | joins with measures avoid double counting in tested cases | Partial | `test/sql/measures.test:889`, `test/sql/measures.test:1473` |
| §5.5 security model | measure views preserve SQL security boundaries | Gap | no privilege-based test in suite |
| §3.4 call-site breadth | explicit use in `HAVING` parity path | Covered | `test/sql/measures.test:1548` |

## Current Verdict

- Core semantics used by the paper’s main language examples are covered, including listings `8`, `9`, and `12` (all four forms), `CURRENT`, rollup behavior, and modifier semantics.
- A strict "100% paper parity" claim is still not justified because of remaining `Gap` items above.

## Minimal Remaining Work for a 100% Claim

1. Add a security-behavior test plan (or explicit out-of-scope declaration if privileges are not testable in this harness).
