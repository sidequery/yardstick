MODEL (name test_model, table orders, primary_key order_id);


METRIC test_sum AS SUM(amount);


DIMENSION test_dim AS status;
