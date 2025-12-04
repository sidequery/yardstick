# Yardstick: Known Limitations

Implementation of Julian Hyde's "Measures in SQL" paper (arXiv:2406.00251).

## Working Features

- `AS MEASURE` in CREATE VIEW to define measures
- `AGGREGATE(measure)` function to evaluate measures
- `AT (ALL)` - grand total (removes all dimensions)
- `AT (ALL dim)` - removes specific dimension from context
- Chained AT modifiers: `AT (ALL dim1) AT (ALL dim2)` correctly correlates on remaining dimensions
- `AT (SET dim = expr)` - fixes dimension to specific value (e.g., YoY comparisons)
- `AT (WHERE condition)` - filters before aggregation
- `AT (VISIBLE)` - uses visible WHERE clause filters
- Multiple measures in same view
- Arithmetic with AGGREGATE results (ratios, percentages, differences)
- SUM, COUNT, MIN, MAX, AVG aggregations
- Derived measures: `revenue - cost AS MEASURE profit` expands to `SUM(revenue) - SUM(cost)`

## Known Limitations

### 1. SET Cannot Reach Beyond WHERE Clause

```sql
-- ISSUE: SET cannot access data filtered out by the outer WHERE clause
SEMANTIC SELECT year,
  AGGREGATE(revenue) AT (SET year = CURRENT year - 1) AS prior_year
FROM sales_v
WHERE year = 2023
GROUP BY year;
-- Returns NULL for prior_year instead of 2022's revenue
-- The WHERE clause filters the view before the subquery runs
```

Per the paper, `AT (SET year = CURRENT year - 1)` should evaluate over the *entire* source table, reaching 2022 data even when the outer query has `WHERE year = 2023`. Our implementation queries the already-filtered result.

**Workaround**: Remove the WHERE clause and filter in application code, or use a CTE.

### 2. No Window Function Measures

```sql
-- NOT SUPPORTED: Window functions in measure definitions
CREATE VIEW v AS
SELECT year,
  SUM(revenue) OVER (ORDER BY year) AS MEASURE running_total
FROM t;
```

