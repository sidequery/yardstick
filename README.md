# Yardstick

A DuckDB extension implementing Julian Hyde's "Measures in SQL" paper ([arXiv:2406.00251](https://arxiv.org/abs/2406.00251)).

## What is this?

Yardstick adds **measure-aware SQL** to DuckDB. Measures are aggregations that know how to re-aggregate themselves when the query context changes. This enables:

- **Percent of total** calculations without CTEs or window functions
- **Year-over-year comparisons** with simple syntax
- **Drill-down analytics** that automatically adjust aggregation context

## Quick Start & Demo

```sql
-- Load the extension
INSTALL yardstick FROM community;
LOAD yardstick;

-- Create the sales table
CREATE TABLE sales (
    id INTEGER PRIMARY KEY,
    year INTEGER,
    region VARCHAR(50),
    amount DECIMAL(10, 2)
);

-- Insert sample data
INSERT INTO sales (id, year, region, amount) VALUES
    (1, 2023, 'North', 15000.00),
    (2, 2023, 'North', 22000.00),
    (3, 2023, 'South', 18000.00),
    (4, 2023, 'South', 12000.00),
    (5, 2023, 'East', 25000.00),
    (6, 2023, 'West', 19000.00),
    (7, 2024, 'North', 28000.00),
    (8, 2024, 'North', 31000.00),
    (9, 2024, 'South', 21000.00),
    (10, 2024, 'South', 16000.00),
    (11, 2024, 'East', 33000.00),
    (12, 2024, 'East', 29000.00),
    (13, 2024, 'West', 24000.00),
    (14, 2024, 'West', 27000.00);

-- Create a view with measures
CREATE VIEW sales_v AS
SELECT
    year,
    region,
    SUM(amount) AS MEASURE revenue,
    COUNT(*) AS MEASURE order_count
FROM sales;

-- Query with AGGREGATE() and AT modifiers
-- SEMANTIC is required for AGGREGATE() without AT, optional for AT queries
SEMANTIC SELECT
    year,
    region,
    AGGREGATE(revenue) AS revenue,
    AGGREGATE(revenue) AT (ALL region) AS year_total,
    AGGREGATE(revenue) / AGGREGATE(revenue) AT (ALL region) AS pct_of_year
FROM sales_v;

-- Variance from the global average
SEMANTIC SELECT
    region,
    AGGREGATE(revenue) AS revenue,
    AGGREGATE(revenue) AT (ALL) / 4.0 AS expected_if_equal,  -- 4 regions
    AGGREGATE(revenue) - (AGGREGATE(revenue) AT (ALL) / 4.0) AS variance
FROM sales_v;

-- Nested percentages (% of year, and that year's % of total)
SEMANTIC SELECT
    year,
    region,
    AGGREGATE(revenue) AS revenue,
    100.0 * AGGREGATE(revenue) / AGGREGATE(revenue) AT (ALL region) AS pct_of_year,
    100.0 * AGGREGATE(revenue) AT (ALL region) / AGGREGATE(revenue) AT (ALL) AS year_pct_of_total
FROM sales_v;

-- Compare 2024 performance to 2023 baseline for each region
SEMANTIC SELECT
    region,
    AGGREGATE(revenue) AT (SET year = 2024) AS rev_2024,
    AGGREGATE(revenue) AT (SET year = 2023) AS rev_2023,
    AGGREGATE(revenue) AT (SET year = 2024) - AGGREGATE(revenue) AT (SET year = 2023) AS growth
FROM sales_v;

-- Filter to specific segments
SEMANTIC SELECT
    year,
    AGGREGATE(revenue) AS total_revenue,
    AGGREGATE(revenue) AT (SET region = 'North') AS north_revenue,
    AGGREGATE(revenue) AT (SET region IN ('North', 'South')) AS north_south_combined
FROM sales_v;
```

## Syntax

### Defining Measures

```sql
CREATE VIEW view_name AS
SELECT
    dimension1,
    dimension2,
    AGG(expr) AS MEASURE measure_name
FROM table;
```

Yardstick automatically handles the grouping. All DuckDB aggregate functions are supported; non-decomposable aggregates (COUNT(DISTINCT), MEDIAN, PERCENTILE_*, QUANTILE_*, MODE) are recomputed from base rows at query time and support AT modifiers, but can be more expensive.

### Querying Measures

Queries using `AGGREGATE()` without AT modifiers must use the `SEMANTIC` prefix. If an `AT (...)` modifier is present, the query can run without `SEMANTIC` because the AT syntax routes the statement through the extension parser.

```sql
SEMANTIC SELECT
    dimensions,
    AGGREGATE(measure_name) [AT modifier]
FROM view_name;
```

```sql
SELECT
    dimensions,
    AGGREGATE(measure_name) AT (ALL)
FROM view_name;
```

### AT Modifiers

| Modifier | Description | Example |
|----------|-------------|---------|
| `AT (ALL)` | Grand total across all dimensions | `AGGREGATE(revenue) AT (ALL)` |
| `AT (ALL dim)` | Total excluding specific dimension | `AGGREGATE(revenue) AT (ALL region)` |
| `AT (ALL expr)` | Total excluding ad hoc dimension | `AGGREGATE(revenue) AT (ALL MONTH(date))` |
| `AT (SET dim = val)` | Fix dimension to specific value | `AGGREGATE(revenue) AT (SET year = 2022)` |
| `AT (SET dim = expr)` | Fix dimension to expression | `AGGREGATE(revenue) AT (SET year = year - 1)` |
| `AT (SET expr = val)` | Fix ad hoc dimension to value | `AGGREGATE(revenue) AT (SET MONTH(date) = 6)` |
| `AT (WHERE cond)` | Pre-aggregation filter | `AGGREGATE(revenue) AT (WHERE region = 'US')` |
| `AT (VISIBLE)` | Use query's WHERE clause | `AGGREGATE(revenue) AT (VISIBLE)` |

## Building

Prerequisites:
- CMake 3.5+
- C++17 compiler
- Cargo

```bash
make        # builds Rust library and DuckDB extension
make test   # runs tests
```

The extension will be at `build/release/extension/yardstick/yardstick.duckdb_extension`

## Limitations

See [LIMITATIONS.md](LIMITATIONS.md) for known issues and workarounds.

Key limitations:
- Window function measures not supported

## Testimonials

"I used this to integrate into a copilotkit chat interface serving graphs, works really well for the llm." - [JFox, DuckDB Discord](https://discord.com/channels/909674491309850675/1009741727600484382/1452672154620530749)

## References

- Julian Hyde, "Measures in SQL" (2024). [arXiv:2406.00251](https://arxiv.org/abs/2406.00251)
- [DuckDB Extension Template](https://github.com/duckdb/extension-template)

## License

MIT
