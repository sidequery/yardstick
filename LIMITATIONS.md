# Yardstick: Known Limitations

Implementation of Julian Hyde's "Measures in SQL" paper (arXiv:2406.00251).

## Working Features

- `AS MEASURE` in CREATE VIEW to define measures
- `AGGREGATE(measure)` function to evaluate measures
- `AT (ALL)` - grand total (removes all dimensions)
- `AT (ALL dim)` - removes specific dimension from context
- Chained AT modifiers: `AT (ALL dim1) AT (ALL dim2)` correctly correlates on remaining dimensions
- `AT (SET dim = expr)` - fixes dimension to specific value (e.g., YoY comparisons)
- Ad hoc dimensions: `AT (ALL MONTH(date))` and `AT (SET MONTH(date) = 6)` for computed dimensions
- `AT (WHERE condition)` - filters before aggregation
- `AT (VISIBLE)` - uses visible WHERE clause filters
- Multiple measures in same view
- Arithmetic with AGGREGATE results (ratios, percentages, differences)
- All DuckDB aggregate functions (SUM, COUNT, AVG, MIN, MAX, STDDEV, MEDIAN, PERCENTILE, MODE, etc.)
- Non-decomposable aggregates (COUNT(DISTINCT), MEDIAN, PERCENTILE_*, QUANTILE_*, MODE) recompute from base rows and support AT modifiers
  - Recompute is correct but can be more expensive than decomposable measures
- Derived measures: `revenue - cost AS MEASURE profit` expands to `SUM(revenue) - SUM(cost)`
- Multi-fact JOINs: measures from different views can be queried together in a single JOIN

## Known Limitations

### 1. Ambiguous AT Contexts for Window Measures

```sql
-- Supported: window measures in view definitions
CREATE VIEW v AS
SELECT year,
  SUM(revenue) OVER (ORDER BY year) AS MEASURE running_total
FROM t;

-- Supported: direct query and AGGREGATE() without AT
SELECT year, AGGREGATE(running_total)
FROM v
GROUP BY year;

-- Supported: AT modifiers when the evaluated window result is single-valued
SELECT AGGREGATE(running_total) AT (WHERE year = 2024) FROM v;

-- Error: AT context produced multiple distinct window values
SELECT year, AGGREGATE(running_total) AT (ALL) FROM v GROUP BY year;
```

### 2. SET with Mixed VISIBLE Predicates

`AT (VISIBLE SET year = year - 1)` preserves independent visible predicates,
such as `year = 2024 AND region = 'North'`, while replacing the year restriction.
Separable boolean expressions are normalized first, including negated
disjunctions and factored predicates. If a predicate still combines the shifted
dimension with another dimension, such as `year = 2024 OR region = 'North'`,
Yardstick rejects the rewrite because it cannot remove just the year
restriction. Use `AT (WHERE ...)` to state the intended filter explicitly.
Normalization that could duplicate or merge volatile function calls is rejected
as well; independent predicates retain their original calls.

### 3. Grouped Measures Read Through Another Relation

A CTE or derived table that passes a measure through while removing rows
(`WHERE`, `GROUP BY`, `HAVING`, `QUALIFY`, `DISTINCT`, `LIMIT`, or sampling)
cannot feed a grouped `AGGREGATE()`. The grouped form recomputes from the
measure's defining view, and `VISIBLE` covers only the consuming query's own
`WHERE`, so the removed rows would silently return. Yardstick rejects the query;
apply the filter in the consuming query instead. Window measures carry row
lineage and honor the CTE's rows.

```sql
-- Error: measure is read through f, which filters its rows
WITH f AS (SELECT * FROM sales_v WHERE region = 'East')
SELECT year, AGGREGATE(revenue) FROM f;

-- Works
WITH f AS (SELECT * FROM sales_v)
SELECT year, AGGREGATE(revenue) FROM f WHERE region = 'East';
```

A plain view or derived table over a measure view is rejected for the grouped
form whether or not it filters, because its definition is not visible when the
query is rewritten. Query the measure view directly or through a CTE, or define
the measure in the filtering view itself:

```sql
-- Error: measure revenue is read through east_v, which does not define it
CREATE VIEW east_v AS SELECT * FROM sales_v WHERE region = 'East';
SELECT year, AGGREGATE(revenue) FROM east_v;

-- Works: the view's WHERE is part of the measure's rows
CREATE VIEW east_v AS
SELECT year, region, SUM(amount) AS MEASURE revenue FROM sales WHERE region = 'East';
SELECT year, AGGREGATE(revenue) FROM east_v;
```
