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
SEMANTIC SELECT year, AGGREGATE(running_total)
FROM v
GROUP BY year;

-- Supported: AT modifiers when the evaluated window result is single-valued
SEMANTIC SELECT AGGREGATE(running_total) AT (WHERE year = 2024) FROM v;

-- Error: AT context produced multiple distinct window values
SEMANTIC SELECT year, AGGREGATE(running_total) AT (ALL) FROM v GROUP BY year;
```
