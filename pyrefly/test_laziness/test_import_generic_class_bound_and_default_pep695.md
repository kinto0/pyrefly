# Import PEP-695 generic classes with a bound and with a default

A class reaches its type parameters by a different route than a function does, so it
gets its own case. Annotating with `G`, constructing it, subclassing it and passing it
to `isinstance` all resolve the export, which is where the bound could be forced, so
plain import is the version of that with the least unrelated noise.

`G` has a bound and no default; `H` has a default and no bound. Neither is used, so
neither the bound nor the default is needed.

## Files

`a.py`:
```python
from b import G, H
```

`b.py`:
```python
from c import Bound, Default

class G[T: Bound]:
    pass

class H[U = Default]:
    pass
```

`c.py`:
```python
class Bound:
    x: int = 1

class Default:
    y: int = 2
```

## Check `a.py`

```expected
a: Solutions
b: Answers
c: Exports

(34 builtin demands hidden)
a -> b::Load(module_exists)
a -> b::Exports(export_exists)
a -> b::Exports(is_implicit_reexport)
a -> b::Exports(get_deprecated)
a -> b::KeyExport(Name("G"))
  b -> c::Exports(is_special_export)
  b -> c::Exports(is_special_export)
a -> b::KeyExport(Name("H"))
```
