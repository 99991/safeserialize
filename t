[1mdiff --git a/tests/test_pandas.py b/tests/test_pandas.py[m
[1mindex 2973eda..523a1a4 100644[m
[1m--- a/tests/test_pandas.py[m
[1m+++ b/tests/test_pandas.py[m
[36m@@ -4,6 +4,8 @@[m [mimport numpy as np[m
 import random[m
 from safeserialize.types.numpy import _allowed_dtypes as numpy_dtypes[m
 [m
[32m+[m[32m# Functions to test whether data == loads(dumps(data))[m
[32m+[m[32m# for pd.Series, pd.DataFrame and pd.Index[m
 def roundtrip_series(s):[m
     serialized_data = dumps(s)[m
     deserialized_series = loads(serialized_data)[m
[36m@@ -20,6 +22,7 @@[m [mdef roundtrip_index(index):[m
     pd.testing.assert_index_equal(index, deserialized_index)[m
 [m
 def test_pandas():[m
[32m+[m[32m    # Test various data types[m
     a = pd.Series([1, 2, None, 4], dtype="Int64", name="int_nullable")[m
     b = pd.Series([3.14, np.nan, 2.71828], dtype="Float32", name="float32")[m
     c = pd.Series([True, False, None], dtype="boolean", name="bool_nullable")[m
[36m@@ -41,23 +44,26 @@[m [mdef test_pandas():[m
 [m
     series = [a, a, b, c, d, e, f, g, h, i, j][m
 [m
[32m+[m[32m    # Test individual series[m
     for s in series:[m
         roundtrip_series(s)[m
 [m
[32m+[m[32m    # Test series combined into one DataFrame[m
     df = pd.concat(series, axis=1)[m
 [m
     roundtrip_df(df)[m
 [m
[31m-    # Data frame with duplicate column names[m
[32m+[m[32m    # DataFrame with duplicate column names[m
     df = pd.concat([a, a, b], axis=1)[m
 [m
     roundtrip_df(df)[m
 [m
[31m-    # Data frame with renamed columns[m
[32m+[m[32m    # DataFrame with renamed columns[m
     df = pd.concat([b, d], axis=1)[m
 [m
     df = df.rename(columns={"string": "d", "float32": "b"})[m
 [m
[32m+[m[32m    # Check names[m
     for column, series in df.items():[m
         assert series.name == column[m
 [m
[36m@@ -95,7 +101,6 @@[m [mdef test_numpy_dtypes():[m
         s = pd.Series(data, dtype=dtype, name=f"numpy_{dtype}")[m
         roundtrip_series(s)[m
 [m
[31m-[m
 def test_datetime():[m
     df = pd.DataFrame({[m
         "year": [2025, 2026],[m
[36m@@ -112,6 +117,11 @@[m [mdef test_datetime():[m
 [m
     roundtrip_series(series)[m
 [m
[32m+[m[32m    # Test datetime with timezone. Timezone-aware objects[m
[32m+[m[32m    # should always be checked with and without timezone[m
[32m+[m[32m    # because internally, Pandas often stores timestamps with[m
[32m+[m[32m    # NumPy, which is not timezone aware, and the conversion is[m
[32m+[m[32m    # easy to get wrong.[m
     start = pd.to_datetime("1/1/2025").tz_localize("Europe/Berlin")[m
     end = pd.to_datetime("12/31/2025").tz_localize("Europe/Berlin")[m
     index = pd.date_range(start=start, end=end, name="Gerhardt")[m
[36m@@ -249,7 +259,7 @@[m [mdef test_categorical_index_advanced():[m
 [m
     roundtrip_index(index)[m
 [m
[31m-    # With unused categories[m
[32m+[m[32m    # Unused categories[m
     categories = ["apple", "banana", "cherry", "date"][m
     data = ["apple", "cherry", "apple"][m
     index = pd.CategoricalIndex([m
[36m@@ -259,7 +269,7 @@[m [mdef test_categorical_index_advanced():[m
 [m
     roundtrip_index(index)[m
 [m
[31m-    # With datetime with timezone[m
[32m+[m[32m    # Datetime with timezone[m
     tz = "America/New_York"[m
     dates = ["2023-01-01", "2023-01-02", "2023-01-03"][m
     categories = pd.to_datetime(dates).tz_localize(tz)[m
[36m@@ -271,7 +281,7 @@[m [mdef test_categorical_index_advanced():[m
 [m
     roundtrip_index(index)[m
 [m
[31m-    # Empty with categories[m
[32m+[m[32m    # Empty data[m
     index = pd.CategoricalIndex([m
         [],[m
         categories=["x", "y", "z"],[m
