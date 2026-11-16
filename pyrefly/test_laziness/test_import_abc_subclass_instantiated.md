# Import ABC subclass instantiated

`a` instantiates `Foo` from `b`. `Foo` extends `ABC`, so instantiating it
checks `Foo` for unimplemented abstract methods.

Whether `bar` is abstract is decided from its definition, so the check
does not infer the return type of `bar`, and `c` is not solved beyond the
exports that binding `b` needs. No superfluous demands.

## Files

`a.py`:
```python
from b import Foo
f = Foo()
```

`b.py`:
```python
from abc import ABC
from c import compute
class Foo(ABC):
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

(117 builtin demands hidden)
a -> b::Exports(is_special_export)
a -> b::Load(module_exists)
a -> b::Exports(export_exists)
a -> b::Exports(is_implicit_reexport)
a -> b::Exports(get_deprecated)
a -> b::KeyExport(Name("Foo"))
  b -> abc::Exports(is_special_export)
  b -> c::Exports(is_special_export)
a -> b::KeyClassMetadata(ClassDefIndex(0))
  b -> abc::Exports(export_exists)
  b -> abc::Exports(is_implicit_reexport)
  b -> abc::Exports(get_deprecated)
  b -> abc::KeyExport(Name("ABC"))
    abc -> _typeshed::Exports(is_special_export)
    abc -> _typeshed::Exports(is_special_export)
    abc -> _typeshed::Exports(is_special_export)
    abc -> _typeshed::Exports(is_special_export)
  b -> abc::KeyClassMetadata(ClassDefIndex(4))
  b -> abc::KeyClassMetadata(ClassDefIndex(4))
  b -> abc::KeyClassMetadata(ClassDefIndex(0))
  b -> abc::KeyClassBaseType(ClassDefIndex(0))
  b -> abc::KeyClassMro(ClassDefIndex(0))
  b -> abc::KeyClassMetadata(ClassDefIndex(0))
  b -> abc::KeyClassMetadata(ClassDefIndex(0))
a -> b::KeyClassMetadata(ClassDefIndex(0))
a -> b::KeyAbstractClassCheck(ClassDefIndex(0))
  b -> abc::KeyAbstractClassCheck(ClassDefIndex(4))
a -> b::KeyClassSynthesizedFields(ClassDefIndex(0))
a -> b::KeyClassMro(ClassDefIndex(0))
  b -> abc::KeyClassBaseType(ClassDefIndex(4))
  b -> abc::KeyClassMro(ClassDefIndex(4))
a -> abc::KeyClassSynthesizedFields(ClassDefIndex(4))
a -> b::KeyClassMro(ClassDefIndex(0))
a -> abc::KeyClassSynthesizedFields(ClassDefIndex(0))
a -> abc::KeyClassMro(ClassDefIndex(0))
```
