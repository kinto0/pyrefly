/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The canonical order of types, which unions and intersections keep their members in.
//!
//! The canonical order compares classes by name and then by the position of their definition.
//! That is definition data, which only a `TypeEnv` can read, since a `Class` is an ID. So the
//! canonical order is `TypeOrd`, a derived order like `Ord` whose context holds the env. It falls
//! back to `Ord` to break ties between classes, so it is a total order as long as the env reads the
//! same data for a class throughout one sort.

use std::cmp::Ordering;
use std::marker::PhantomData;
use std::sync::Arc;

use compact_str::CompactString;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModuleStyle;
use pyrefly_python::qname::QName;
use pyrefly_python::short_identifier::ShortIdentifier;
use pyrefly_util::uniques::Unique;
use ruff_python_ast::name::Name;
use starlark_map::ordered_map::OrderedMap;
use starlark_map::ordered_set::OrderedSet;
use vec1::Vec1;

use crate::env::TypeEnv;
use crate::param_spec::ParamSpec;
use crate::sentinel::Sentinel;
use crate::type_var::TypeVar;
use crate::type_var_tuple::TypeVarTuple;
use crate::types::Type;

/// The context for `TypeOrd`: the env that class names are read from.
#[derive(Clone, Copy)]
pub struct TypeOrdCtx<'a> {
    env: &'a dyn TypeEnv,
}

impl<'a> TypeOrdCtx<'a> {
    pub fn new(env: &'a dyn TypeEnv) -> Self {
        Self { env }
    }

    pub fn env(&self) -> &'a dyn TypeEnv {
        self.env
    }
}

/// The canonical order of types (see the module documentation). Types that hold no class use the
/// default, which is `Ord`.
pub trait TypeOrd: Ord {
    fn type_cmp(&self, other: &Self, ctx: &TypeOrdCtx) -> Ordering {
        let _ = ctx;
        self.cmp(other)
    }
}

/// The order of union members: `TypeOrd`, except that a member marked with its truthiness sorts as
/// the unmarked type, so the marker changes neither the display order of the union nor the order in
/// which its members are solved.
pub fn cmp_union_members(x: &Type, y: &Type, ctx: &TypeOrdCtx) -> Ordering {
    x.ignoring_truthiness_marker()
        .type_cmp(y.ignoring_truthiness_marker(), ctx)
        .then_with(|| x.type_cmp(y, ctx))
}

impl TypeOrd for () {}
impl TypeOrd for bool {}
impl TypeOrd for char {}
impl TypeOrd for i8 {}
impl TypeOrd for i16 {}
impl TypeOrd for i32 {}
impl TypeOrd for i64 {}
impl TypeOrd for i128 {}
impl TypeOrd for isize {}
impl TypeOrd for u8 {}
impl TypeOrd for u16 {}
impl TypeOrd for u32 {}
impl TypeOrd for u64 {}
impl TypeOrd for u128 {}
impl TypeOrd for usize {}
impl TypeOrd for String {}
impl TypeOrd for CompactString {}
impl TypeOrd for str {}
impl<T> TypeOrd for PhantomData<T> {}
impl TypeOrd for Name {}
impl TypeOrd for ModuleName {}
impl TypeOrd for ModulePath {}
impl TypeOrd for ModuleStyle {}
impl TypeOrd for ShortIdentifier {}
impl TypeOrd for QName {}
impl TypeOrd for Unique {}
impl TypeOrd for ParamSpec {}
impl TypeOrd for TypeVar {}
impl TypeOrd for TypeVarTuple {}
impl TypeOrd for Sentinel {}

impl<T0: TypeOrd, T1: TypeOrd> TypeOrd for (T0, T1) {
    fn type_cmp(&self, other: &Self, ctx: &TypeOrdCtx) -> Ordering {
        self.0
            .type_cmp(&other.0, ctx)
            .then_with(|| self.1.type_cmp(&other.1, ctx))
    }
}

impl<T0: TypeOrd, T1: TypeOrd, T2: TypeOrd> TypeOrd for (T0, T1, T2) {
    fn type_cmp(&self, other: &Self, ctx: &TypeOrdCtx) -> Ordering {
        self.0
            .type_cmp(&other.0, ctx)
            .then_with(|| self.1.type_cmp(&other.1, ctx))
            .then_with(|| self.2.type_cmp(&other.2, ctx))
    }
}

/// Lexicographic, like the `Ord` of slices.
fn cmp_iter<'a, T: TypeOrd + 'a>(
    mut xs: impl Iterator<Item = &'a T>,
    mut ys: impl Iterator<Item = &'a T>,
    ctx: &TypeOrdCtx,
) -> Ordering {
    loop {
        match (xs.next(), ys.next()) {
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(x), Some(y)) => match x.type_cmp(y, ctx) {
                Ordering::Equal => {}
                ord => return ord,
            },
        }
    }
}

impl<T: TypeOrd> TypeOrd for [T] {
    fn type_cmp(&self, other: &Self, ctx: &TypeOrdCtx) -> Ordering {
        cmp_iter(self.iter(), other.iter(), ctx)
    }
}

impl<T: TypeOrd> TypeOrd for Vec<T> {
    fn type_cmp(&self, other: &Self, ctx: &TypeOrdCtx) -> Ordering {
        self.as_slice().type_cmp(other.as_slice(), ctx)
    }
}

impl<T: TypeOrd> TypeOrd for Vec1<T> {
    fn type_cmp(&self, other: &Self, ctx: &TypeOrdCtx) -> Ordering {
        self.as_slice().type_cmp(other.as_slice(), ctx)
    }
}

impl<T: TypeOrd + ?Sized> TypeOrd for &T {
    fn type_cmp(&self, other: &Self, ctx: &TypeOrdCtx) -> Ordering {
        (**self).type_cmp(*other, ctx)
    }
}

impl<T: TypeOrd + ?Sized> TypeOrd for Box<T> {
    fn type_cmp(&self, other: &Self, ctx: &TypeOrdCtx) -> Ordering {
        (**self).type_cmp(&**other, ctx)
    }
}

impl<T: TypeOrd + ?Sized> TypeOrd for Arc<T> {
    fn type_cmp(&self, other: &Self, ctx: &TypeOrdCtx) -> Ordering {
        (**self).type_cmp(&**other, ctx)
    }
}

impl<T: TypeOrd> TypeOrd for Option<T> {
    fn type_cmp(&self, other: &Self, ctx: &TypeOrdCtx) -> Ordering {
        match (self, other) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) => Ordering::Less,
            (Some(_), None) => Ordering::Greater,
            (Some(x), Some(y)) => x.type_cmp(y, ctx),
        }
    }
}

impl<K: TypeOrd, V: TypeOrd> TypeOrd for OrderedMap<K, V> {
    fn type_cmp(&self, other: &Self, ctx: &TypeOrdCtx) -> Ordering {
        let mut xs = self.iter();
        let mut ys = other.iter();
        loop {
            match (xs.next(), ys.next()) {
                (None, None) => return Ordering::Equal,
                (None, Some(_)) => return Ordering::Less,
                (Some(_), None) => return Ordering::Greater,
                (Some((xk, xv)), Some((yk, yv))) => {
                    match xk.type_cmp(yk, ctx).then_with(|| xv.type_cmp(yv, ctx)) {
                        Ordering::Equal => {}
                        ord => return ord,
                    }
                }
            }
        }
    }
}

impl<T: TypeOrd> TypeOrd for OrderedSet<T> {
    fn type_cmp(&self, other: &Self, ctx: &TypeOrdCtx) -> Ordering {
        cmp_iter(self.iter(), other.iter(), ctx)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use pyrefly_python::module::Module;
    use pyrefly_python::nesting_context::NestingContext;
    use ruff_python_ast::Identifier;
    use ruff_text_size::TextRange;
    use ruff_text_size::TextSize;

    use super::*;
    use crate::class::Class;
    use crate::class::ClassType;
    use crate::lit_int::LitInt;
    use crate::literal::Lit;
    use crate::literal::LitEnum;
    use crate::test_type_env::TestTypeEnv;
    use crate::types::AnyStyle;
    use crate::types::NeverStyle;
    use crate::types::TArgs;

    fn module(name: &str) -> Module {
        Module::new(
            ModuleName::from_str(name),
            ModulePath::namespace(PathBuf::from(name)),
            Arc::new(String::new()),
        )
    }

    fn identifier(name: &str, position: u32) -> Identifier {
        Identifier::new(Name::new(name), TextRange::empty(TextSize::new(position)))
    }

    /// Define in `env`, as the next class of `module`, a top-level class `name` whose name starts
    /// at `position`.
    fn class(env: &TestTypeEnv, module_name: &str, name: &str, position: u32) -> Class {
        env.define_class(
            identifier(name, position),
            NestingContext::toplevel(),
            module(module_name),
        )
    }

    #[test]
    fn test_classes_order_by_name_then_identity() {
        let env = TestTypeEnv::new();
        let ctx = TypeOrdCtx::new(&env);
        let foo_zed = class(&env, "foo", "Zed", 0);
        let bar_abc = class(&env, "bar", "Abc", 9);
        let bar_zed = class(&env, "bar", "Zed", 5);
        // A later class of `bar` with the same name, whose name starts earlier.
        let bar_zed_again = env.define_class(
            identifier("Zed", 0),
            NestingContext::class(
                ShortIdentifier::new(&identifier("Outer", 20)),
                NestingContext::toplevel(),
            ),
            module("bar"),
        );
        // The name comes first.
        assert_eq!(foo_zed.type_cmp(&bar_abc, &ctx), Ordering::Greater);
        // Then `Ord`: the module, then the index. The position is not read.
        assert_eq!(bar_zed.type_cmp(&foo_zed, &ctx), Ordering::Less);
        assert_eq!(bar_zed.type_cmp(&bar_zed_again, &ctx), Ordering::Less);
        assert_eq!(foo_zed.type_cmp(&foo_zed.clone(), &ctx), Ordering::Equal);
        assert_eq!(
            Type::ClassDef(bar_abc).type_cmp(&Type::None, &ctx),
            Type::ClassDef(bar_zed).type_cmp(&Type::None, &ctx),
            "types of different variants are ordered by variant"
        );
    }

    /// An open buffer and its file on disk are one module to class identity, but are bound
    /// separately, so they can hold different classes at one index. Each class reads its own row,
    /// so the order stays total.
    #[test]
    fn test_type_ord_is_total_with_a_buffer_and_its_file_on_disk() {
        let env = TestTypeEnv::new();
        let ctx = TypeOrdCtx::new(&env);
        let foo = |path: ModulePath| {
            Module::new(ModuleName::from_str("foo"), path, Arc::new(String::new()))
        };
        let on_disk = env.define_class(
            identifier("Foo", 0),
            NestingContext::toplevel(),
            foo(ModulePath::filesystem(PathBuf::from("foo.py"))),
        );
        let in_memory = env.define_class(
            identifier("Bar", 0),
            NestingContext::toplevel(),
            foo(ModulePath::memory(PathBuf::from("foo.py"))),
        );
        assert_eq!(
            on_disk, in_memory,
            "class identity treats them as one class"
        );
        let between = class(&env, "baz", "Cat", 0);
        assert_eq!(in_memory.type_cmp(&between, &ctx), Ordering::Less);
        assert_eq!(between.type_cmp(&on_disk, &ctx), Ordering::Less);
        let classes = [on_disk, in_memory, between];
        for x in &classes {
            for y in &classes {
                let xy = x.type_cmp(y, &ctx);
                assert_eq!(xy, y.type_cmp(x, &ctx).reverse(), "{x:?} vs {y:?}");
                for z in &classes {
                    let yz = y.type_cmp(z, &ctx);
                    if xy != Ordering::Greater && yz != Ordering::Greater {
                        assert_ne!(
                            x.type_cmp(z, &ctx),
                            Ordering::Greater,
                            "{x:?} <= {y:?} <= {z:?}"
                        );
                    }
                }
            }
        }
    }

    /// A deterministic xorshift generator.
    struct Rng(u64);

    impl Rng {
        fn below(&mut self, n: usize) -> usize {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 % n as u64) as usize
        }
    }

    /// A random type over `classes`, which share names and positions in different ways.
    fn random_type(rng: &mut Rng, env: &TestTypeEnv, classes: &[Class], depth: u32) -> Type {
        let sub = |rng: &mut Rng| {
            if depth == 0 {
                Type::None
            } else {
                random_type(rng, env, classes, depth - 1)
            }
        };
        let class = |rng: &mut Rng| classes[rng.below(classes.len())];
        match rng.below(16) {
            0 | 1 => ClassType::new(class(rng), TArgs::default()).to_type(),
            2 => Type::ClassDef(class(rng)),
            3 => Type::ClassType(env.stdlib().list(sub(rng))),
            4 => Lit::Int(LitInt::new(rng.below(3) as i64)).to_implicit_type(),
            5 => Lit::Str(["a", "b"][rng.below(2)].into()).to_implicit_type(),
            6 => Lit::Bool(rng.below(2) == 0).to_implicit_type(),
            7 => Lit::Enum(Box::new(LitEnum {
                class: ClassType::new(class(rng), TArgs::default()),
                member: Name::new(["X", "Y"][rng.below(2)]),
                ty: Type::None,
            }))
            .to_implicit_type(),
            8 => Type::None,
            9 => Type::Any([AnyStyle::Explicit, AnyStyle::Implicit][rng.below(2)]),
            10 => Type::Never(NeverStyle::Never),
            11 => Type::concrete_tuple((0..rng.below(3)).map(|_| sub(rng)).collect()),
            12 => Type::union((0..2 + rng.below(2)).map(|_| sub(rng)).collect()),
            13 => Type::Type(Box::new(sub(rng))),
            14 => {
                let t = sub(rng);
                let marker = if rng.below(2) == 0 {
                    Type::Truthy
                } else {
                    Type::Falsy
                };
                Type::Intersect(Box::new((vec![t.clone(), marker], t)))
            }
            _ => Type::Intersect(Box::new((vec![sub(rng), sub(rng)], Type::None))),
        }
    }

    /// `type_cmp` is a total order, so sorting with it cannot panic and gives one order for any
    /// input order. `Equal` implies `==`. On these types, which hold no open buffer and its file
    /// on disk, `==` also implies `Equal`.
    #[test]
    fn test_type_ord_is_a_total_order() {
        let env = TestTypeEnv::new();
        let ctx = TypeOrdCtx::new(&env);
        let mut classes = Vec::new();
        for module in ["aaa", "zzz"] {
            for (name, position) in [("Abc", 0), ("Abc", 4), ("Zed", 0), ("Zed", 4)] {
                classes.push(class(&env, module, name, position));
            }
        }
        let mut rng = Rng(0x2545F4914F6CDD1D);
        for _ in 0..200 {
            let members: Vec<Type> = (0..rng.below(12))
                .map(|_| random_type(&mut rng, &env, &classes, 2))
                .collect();
            for x in &members {
                for y in &members {
                    let xy = x.type_cmp(y, &ctx);
                    assert_eq!(xy, y.type_cmp(x, &ctx).reverse(), "{x:?} vs {y:?}");
                    assert_eq!(xy == Ordering::Equal, x == y, "{x:?} vs {y:?}");
                    for z in &members {
                        if xy != Ordering::Greater && y.type_cmp(z, &ctx) != Ordering::Greater {
                            assert_ne!(x.type_cmp(z, &ctx), Ordering::Greater, "{x:?} vs {z:?}");
                        }
                    }
                }
            }
            let mut sorted = members.clone();
            sorted.sort_by(|x, y| cmp_union_members(x, y, &ctx));
            let mut reversed = members.clone();
            reversed.reverse();
            reversed.sort_by(|x, y| cmp_union_members(x, y, &ctx));
            assert_eq!(sorted, reversed);
        }
    }
}
