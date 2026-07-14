/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

/// A minimal Polars stub: `DataFrame` is defined in `polars.dataframe.frame` and
/// re-exported from `polars`, and its column-access methods return an opaque type.
fn env_with_polars_stubs() -> TestEnv {
    let mut env = TestEnv::new();
    env.add(
        "polars.dataframe.frame",
        r#"
from typing import Iterator
class Series: ...
class DataFrame:
    columns: list[str]
    def __init__(self, data: object = None, schema: object = None) -> None: ...
    def __getitem__(self, key: str) -> Series: ...
    def __iter__(self) -> Iterator[Series]: ...
    def __contains__(self, key: str) -> bool: ...
    def head(self, n: int = 5) -> "DataFrame": ...
"#,
    );
    env.add(
        "polars",
        "from polars.dataframe.frame import DataFrame as DataFrame, Series as Series",
    );
    env
}

testcase!(
    test_construct_int_and_str_columns,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1, 2], "b": ["x", "y"]}))  # E: revealed type: DataFrame[a: int, b: str]
"#,
);

testcase!(
    test_columns_in_source_order,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"b": ["x"], "a": [1]}))  # E: revealed type: DataFrame[b: str, a: int]
"#,
);

testcase!(
    test_non_polars_table_untouched,
    env_with_polars_stubs(),
    r#"
from typing import reveal_type
class DataFrame:
    def __init__(self, data: object = None) -> None: ...
reveal_type(DataFrame({"a": [1]}))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_fallback_non_string_key,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({1: [1]}))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_fallback_value_not_list,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": 1}))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_fallback_non_literal_element,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
x: int = 1
reveal_type(pl.DataFrame({"a": [x]}))  # E: revealed type: DataFrame
def g() -> int: ...
reveal_type(pl.DataFrame({"b": [g()]}))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_fallback_mixed_int_and_str,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1, "s"]}))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_construct_float_column,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1.0, 2.0]}))  # E: revealed type: DataFrame[a: float]
"#,
);

testcase!(
    test_construct_bool_column,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [True, False]}))  # E: revealed type: DataFrame[a: bool]
"#,
);

testcase!(
    test_construct_bytes_column,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [b"x", b"y"]}))  # E: revealed type: DataFrame[a: bytes]
"#,
);

testcase!(
    test_fallback_complex_not_modeled,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1j]}))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_fallback_mixed_int_and_bool,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1, True]}))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_fallback_empty_list,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": []}))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_fallback_empty_dict,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({}))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_fallback_keyword_argument,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame(data={"a": [1]}))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_fallback_multiple_positional_args,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1]}, None))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_fallback_one_bad_column_pins_whole_dict,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1], "b": 2}))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_fallback_duplicate_key,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
reveal_type(pl.DataFrame({"a": [1], "a": ["x"]}))  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_subclass_falls_back,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
class MyFrame(pl.DataFrame): ...
reveal_type(MyFrame({"a": [1]}))  # E: revealed type: MyFrame
"#,
);

testcase!(
    test_element_type_error_reported_once,
    env_with_polars_stubs(),
    r#"
import polars as pl
pl.DataFrame({"a": [undefined_name]})  # E: Could not find name `undefined_name`
"#,
);

testcase!(
    test_schema_dataframe_assignable_to_underlying,
    env_with_polars_stubs(),
    r#"
import polars as pl
df: pl.DataFrame = pl.DataFrame({"a": [1]})
def f(x: pl.DataFrame) -> None: ...
f(pl.DataFrame({"a": [1]}))
"#,
);

testcase!(
    test_schema_dataframe_attribute_access,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df.columns)  # E: revealed type: list[str]
reveal_type(df.head())  # E: revealed type: DataFrame
"#,
);

testcase!(
    test_schema_dataframe_subscript,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type(df["a"])  # E: revealed type: Series
"#,
);

testcase!(
    test_schema_dataframe_iteration,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
for col in df:
    reveal_type(col)  # E: revealed type: Series
"#,
);

testcase!(
    test_schema_dataframe_membership,
    env_with_polars_stubs(),
    r#"
import polars as pl
from typing import reveal_type
df = pl.DataFrame({"a": [1]})
reveal_type("a" in df)  # E: revealed type: bool
"#,
);
