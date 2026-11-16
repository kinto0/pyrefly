# Variance inference reaching a class with specified variance

`a` defines `Wrapper[T]`, whose variance is inferred, and returns
`Wrapper[int]` as `Wrapper[object]`, which needs that variance. `Wrapper`
mentions `Box[T]` from `b`, and `Box` declares its type parameter covariant.

The variance of `Box` is specified, so variance inference uses it without
walking the bases and fields of `Box`. No superfluous demands.

## Files

`a.py`:
```python
from b import Box
class Wrapper[T]:
    def unwrap(self) -> Box[T]: ...
def f(w: Wrapper[int]) -> Wrapper[object]:
    return w
```

`b.py`:
```python
from typing import Generic, TypeVar
from c import compute
T_co = TypeVar("T_co", covariant=True)
class Box(Generic[T_co]):
    def bar(self):
        return compute()
```

`c.py`:
```python
def compute() -> int: return 42
```

## Check `a.py`

```expected
a: Solutions
b: Answers
c: Exports

(55 builtin demands hidden)
a -> b::Exports(is_special_export)
a -> b::Load(module_exists)
a -> b::Exports(export_exists)
a -> b::Exports(is_implicit_reexport)
a -> b::Exports(get_deprecated)
a -> b::KeyExport(Name("Box"))
  b -> c::Exports(is_special_export)
a -> b::KeyTParams(ClassDefIndex(0))
a -> b::KeyTParams(ClassDefIndex(0))
a -> b::KeyClassMetadata(ClassDefIndex(0))
a -> types::KeyClassMetadata(ClassDefIndex(21))
a -> types::KeyClassBaseType(ClassDefIndex(21))
a -> types::KeyClassMro(ClassDefIndex(21))
a -> types::KeyClassMetadata(ClassDefIndex(22))
a -> types::KeyClassBaseType(ClassDefIndex(22))
a -> types::KeyClassMro(ClassDefIndex(22))
a -> types::KeyClassSynthesizedFields(ClassDefIndex(22))
```
