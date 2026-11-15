# Import generic class subscripted in an annotation

Importing a generic class and specializing it in an annotation (`x: Foo[int]`).

`a -> b::KeyExport("Foo")` is necessary to resolve the import, and
`a -> b::KeyTParams(0)` to specialize `Foo`. They are the only demands on `b`:
specializing `Foo` does not read its metadata, because the checks that read
metadata only apply to a class without type parameters.

## Files

`a.py`:
```python
from b import Foo
x: Foo[int]
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
