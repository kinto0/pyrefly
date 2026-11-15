# Import generic class subscripted as a value

Importing a generic class and specializing it in a value expression (`y = Foo[int]`).

`a -> b::KeyExport("Foo")` is necessary to resolve the import, and
`a -> b::KeyTParams(0)` to specialize `Foo`. They are the only demands on `b`:
subscripting a class object that has type parameters specializes it, so neither
`__class_getitem__`, the metaclass `__getitem__`, nor enum member lookup is
considered, and the metadata those checks read is not needed.

## Files

`a.py`:
```python
from b import Foo
y = Foo[int]
```

`b.py`:
```python
class Foo[T]:
    x: int = 1
```

## Check `a.py`

```expected
a: Solutions
b: Answers

(34 builtin demands hidden)
a -> b::Exports(is_special_export)
a -> b::Load(module_exists)
a -> b::Exports(export_exists)
a -> b::Exports(is_implicit_reexport)
a -> b::Exports(get_deprecated)
a -> b::KeyExport(Name("Foo"))
a -> b::KeyTParams(ClassDefIndex(0))
```
