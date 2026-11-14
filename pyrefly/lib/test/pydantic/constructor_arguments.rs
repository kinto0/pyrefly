/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::pydantic_testcase;
use crate::test::pydantic::util::pydantic_env;

pydantic_testcase!(
    test_constructor_argument_literal_values,
    r#"
from typing import Literal
from pydantic import BaseModel, Field

class Model(BaseModel):
    number: bool
    bounded: int = Field(ge=0)

def invalid_number() -> Literal["invalid"]:
    return "invalid"

def negative() -> Literal[-1]:
    return -1

def choices(flag: bool) -> Literal["1", "invalid"]:
    return "1" if flag else "invalid"

Model(number="invalid", bounded=0)  # E: is not valid for Pydantic field `number`
Model(number=invalid_number(), bounded=0)  # E: is not valid for Pydantic field `number`
Model(number=choices(True), bounded=0)  # E: is not valid for Pydantic field `number`
Model(number="1", bounded=negative())  # E: violates Pydantic `ge` constraint

def conditional(flag: bool) -> None:
    Model(number="1" if flag else "invalid", bounded=0)  # E: is not valid for Pydantic field `number`
    Model(number="1", bounded=0 if flag else -1)  # E: violates Pydantic `ge` constraint
"#,
);

pydantic_testcase!(
    test_constructor_argument_aliases_and_positions,
    r#"
from typing import Literal
from pydantic import BaseModel, Field

class Positional(BaseModel):
    number: bool = Field(kw_only=False)
    bounded: int = Field(ge=0, kw_only=False)

class Aliased(BaseModel, validate_by_name=True, validate_by_alias=True):
    number: bool = Field(alias="value")

def invalid() -> Literal["invalid"]:
    return "invalid"

def negative() -> Literal[-1]:
    return -1

Positional(invalid(), 0)  # E: is not valid for Pydantic field `number`
Positional(1, negative())  # E: violates Pydantic `ge` constraint
Aliased(number=invalid())  # E: is not valid for Pydantic field `number`
Aliased(value=invalid())  # E: is not valid for Pydantic field `number`
Aliased(value="1")
"#,
);

pydantic_testcase!(
    test_constructor_argument_contextual_types,
    r#"
from typing import assert_type
from pydantic import BaseModel, Field

class Box[T]:
    def __init__(self, value: T) -> None:
        self.value = value

class Model[T](BaseModel):
    value: T = Field(strict=True)

box: Model[Box[float]] = Model(value=Box(1))
assert_type(box, Model[Box[float]])
assert_type(box.value, Box[float])

items: Model[list[float]] = Model(value=[1])
assert_type(items.value, list[float])

def nested() -> Model[Model[list[float]]]:
    return Model(value=Model(value=[1]))

assert_type(nested().value.value, list[float])
"#,
);

pydantic_testcase!(
    test_constructor_argument_recovery_and_custom_init,
    r#"
from typing import Any, Literal, overload
from pydantic import BaseModel, Field

class Recovery(BaseModel):
    value: Literal[1, 2] = Field(gt=1)

Recovery(value="bad")  # E: is not assignable to parameter `value`

class Custom(BaseModel):
    value: int = Field(ge=0)

    def __init__(self, *args: Any, **data: Any) -> None:
        super().__init__(**data)

Custom(value=-1)  # E: violates Pydantic `ge` constraint

args: tuple[int, ...] = (1,)
Custom(*args, value=-1)  # E: violates Pydantic `ge` constraint

class Overloaded(BaseModel):
    value: int = Field(ge=0)

    @overload
    def __init__(self, *, value: Literal[-1], tag: Literal[0] = 0) -> None: ...
    @overload
    def __init__(self, *, value: Literal[1], tag: Literal[1] = 1) -> None: ...
    def __init__(self, *, value: int, tag: int = 0) -> None:
        super().__init__(value=value)

Overloaded(value=-1)  # E: violates Pydantic `ge` constraint
Overloaded(value=-1, tag="bad")  # E: No matching overload # E: violates Pydantic `ge` constraint

def expanded(flag: bool) -> None:
    Overloaded(value=-1 if flag else 1)  # E: violates Pydantic `ge` constraint
"#,
);

pydantic_testcase!(
    test_constructor_argument_union_init,
    r#"
from pydantic import BaseModel, Field

def init_a(self: object, *, value: int) -> None: ...
def init_b(self: object, *, value: int) -> None: ...

def build(flag: bool) -> None:
    class Model(BaseModel):
        value: int = Field(ge=0)
        __init__ = init_a if flag else init_b

    Model(value=-1)  # E: violates Pydantic `ge` constraint
"#,
);

pydantic_testcase!(
    test_constructor_argument_erased_init,
    r#"
from typing import Any, Callable
from pydantic import BaseModel, Field

def erase(fn: Any) -> Any:
    return fn

class Model(BaseModel):
    value: int = Field(ge=0)

    @erase
    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

Model(value=-1)  # E: violates Pydantic `ge` constraint

def erase_callable(fn: Any) -> Callable[..., None]:
    return fn

class CallableModel(BaseModel):
    value: int = Field(ge=0)

    @erase_callable
    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

CallableModel(value=-1)  # E: violates Pydantic `ge` constraint

def make(cls: type["Assigned"], value: int) -> "Assigned":
    return object.__new__(cls)

class Assigned(BaseModel):
    value: int = Field(ge=0, kw_only=False)
    __new__ = staticmethod(make)

    @erase
    def __init__(self, value: int) -> None:
        super().__init__(value=value)

Assigned(-1)  # E: violates Pydantic `ge` constraint

def erase_new(fn: Any) -> Callable[..., "Opaque"]:
    return fn

class Opaque(BaseModel):
    value: int = Field(ge=0, kw_only=False)

    @staticmethod
    @erase_new
    def __new__(cls: type["Opaque"], value: int) -> "Opaque":
        return object.__new__(cls)

    @erase
    def __init__(self, value: int) -> None:
        super().__init__(value=value)

Opaque(-1)  # E: violates Pydantic `ge` constraint
"#,
);

#[test]
fn test_nested_constructor_arguments() -> anyhow::Result<()> {
    let mut code =
        "from pydantic import BaseModel\n\nclass N0(BaseModel):\n    x: int = 0\n".to_owned();
    let mut expression = "N0(x=1)".to_owned();
    let depth = 24;
    for index in 1..=depth {
        code.push_str(&format!(
            "\nclass N{index}(BaseModel):\n    a: N{}\n    c: int = 0\n",
            index - 1,
        ));
        expression = format!("N{index}(a={expression}, c=1)");
    }
    code.push_str(&format!(
        "\nv = {expression}\nannotated: N{depth} = {expression}\n"
    ));
    let mut env = pydantic_env();
    env.add("main", &code);
    let (state, handle) = env.to_state();
    let errors = state.transaction().get_errors([&handle("main")]);
    errors.check_against_expectations()?;
    errors.check_var_leak()
}
