# Yardstick: Known Limitations

Implementation of Julian Hyde's "Measures in SQL" paper (arXiv:2406.00251).

## Working Features

- `AS MEASURE` in CREATE VIEW to define measures
- `AGGREGATE(measure)` function to evaluate measures
- `AT (ALL)` - grand total (removes all dimensions)
- `AT (ALL dim)` - removes specific dimension from context
- `AT (SET dim = expr)` - fixes dimension to specific value (e.g., YoY comparisons)
- `AT (WHERE condition)` - filters before aggregation
- `AT (VISIBLE)` - uses visible WHERE clause filters
- Multiple measures in same view
- Arithmetic with AGGREGATE results (ratios, percentages, differences)
- SUM, COUNT, MIN, MAX, AVG aggregations

## Known Limitations

### 1. Chained AT Modifiers

```sql
-- ISSUE: Chaining multiple AT modifiers doesn't work as expected
AGGREGATE(revenue) AT (ALL region) AT (ALL category)
-- Returns grand total instead of year-only total
-- Expected: Remove region, then remove category (keeping year)
-- Actual: Removes all dimensions
```

**Workaround**: Use single `AT (ALL dim1, dim2)` syntax if supported, or restructure query.

### 2. No Derived Measures

```sql
-- NOT YET SUPPORTED: Measures referencing other measures
CREATE VIEW v AS
SELECT year,
  SUM(revenue) AS MEASURE revenue,
  SUM(cost) AS MEASURE cost,
  revenue - cost AS MEASURE profit  -- NOT SUPPORTED
FROM t GROUP BY year;
```

**Workaround**: Calculate derived values in the SELECT using AGGREGATE().

### 3. SET Cannot Reach Beyond WHERE Clause

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

### 4. No Window Function Measures

```sql
-- NOT SUPPORTED: Window functions in measure definitions
CREATE VIEW v AS
SELECT year,
  SUM(revenue) OVER (ORDER BY year) AS MEASURE running_total
FROM t;
```

