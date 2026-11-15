# Import class as annotation

Importing a class and using it only as a type annotation (`x: Foo`).

`a -> b::KeyExport("Foo")` is necessary to resolve the import, and it
is the only demand on `b`. Promoting `Foo` to an instance type reads
neither its metadata nor its base classes, because a TypedDict instance
is an ordinary `ClassType`.

## Files

`a.py`:
```python
from b import Foo
x: Foo
```

`b.py`:
```python
class Foo:
    x: int = 1
```

## Check `a.py`

```expected
a: Solutions
b: Answers

(32 builtin demands hidden)
a -> b::Load(module_exists)
a -> b::Exports(export_exists)
a -> b::Exports(is_implicit_reexport)
a -> b::Exports(get_deprecated)
a -> b::KeyExport(Name("Foo"))
```
