/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The canonical order of types, which unions and intersections keep their members in.
//!
//! The canonical order compares classes by name and then by the position of their definition.
//! That is definition data, which only a `TypeEnv` can read once `Class` stops carrying it. So the
//! canonical order is `TypeOrd`, a derived order like `Ord` whose context holds the env. It falls
//! back to `Ord` to break ties between classes, so it is a total order as long as the env reads the
//! same data for a class throughout one sort. While `Class` carries its data, `TypeOrd` is `Ord`.

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
    use crate::class::ClassDefIndex;
    use crate::class::ClassType;
    use crate::lit_int::LitInt;
    use crate::literal::Lit;
    use crate::literal::LitEnum;
    use crate::test_type_env::TestTypeEnv;
    use crate::types::AnyStyle;
    use crate::types::NeverStyle;
    use crate::types::TArgs;

    /// A class `name` whose name starts at `position` in `module`, as the `index`-th class there.
    fn class(module: &str, index: u32, name: &str, position: u32) -> Class {
        Class::new(
            ClassDefIndex(index),
            Identifier::new(Name::new(name), TextRange::empty(TextSize::new(position))),
            NestingContext::toplevel(),
            Module::new(
                ModuleName::from_str(module),
                ModulePath::namespace(PathBuf::from(module)),
                Arc::new(String::new()),
            ),
            false,
            false,
            None,
        )
    }

    #[test]
    fn test_classes_order_by_name_then_position_then_identity() {
        let env = TestTypeEnv::new();
        let ctx = TypeOrdCtx::new(&env);
        let foo_zed = class("foo", 0, "Zed", 0);
        let bar_abc = class("bar", 0, "Abc", 9);
        let bar_zed = class("bar", 1, "Zed", 0);
        let bar_zed_later = class("bar", 2, "Zed", 5);
        let bar_zed_again = class("bar", 3, "Zed", 0);
        // The name comes first.
        assert_eq!(foo_zed.type_cmp(&bar_abc, &ctx), Ordering::Greater);
        // Then the position, before the module.
        assert_eq!(foo_zed.type_cmp(&bar_zed_later, &ctx), Ordering::Less);
        // Then `Ord`: the module, then the index.
        assert_eq!(bar_zed.type_cmp(&foo_zed, &ctx), Ordering::Less);
        assert_eq!(bar_zed.type_cmp(&bar_zed_again, &ctx), Ordering::Less);
        assert_eq!(foo_zed.type_cmp(&foo_zed.clone(), &ctx), Ordering::Equal);
        assert_eq!(
            Type::ClassDef(bar_abc).type_cmp(&Type::None, &ctx),
            Type::ClassDef(bar_zed).type_cmp(&Type::None, &ctx),
            "types of different variants are ordered by variant"
        );
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
        let class = |rng: &mut Rng| classes[rng.below(classes.len())].clone();
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

    #[test]
    fn test_type_ord_agrees_with_ord_while_classes_carry_their_data() {
        let env = TestTypeEnv::new();
        let ctx = TypeOrdCtx::new(&env);
        let mut classes = Vec::new();
        for module in ["aaa", "zzz"] {
            for (index, (name, position)) in [("Abc", 0), ("Abc", 4), ("Zed", 0), ("Zed", 4)]
                .into_iter()
                .enumerate()
            {
                classes.push(class(module, index as u32, name, position));
            }
        }
        let mut rng = Rng(0x2545F4914F6CDD1D);
        for _ in 0..200 {
            let mut members: Vec<Type> = (0..rng.below(12))
                .map(|_| random_type(&mut rng, &env, &classes, 2))
                .collect();
            for x in &members {
                for y in &members {
                    assert_eq!(x.type_cmp(y, &ctx), x.cmp(y), "{x:?} vs {y:?}");
                }
            }
            // The order that union simplification sorts members in, without an env.
            let mut expected = members.clone();
            expected.sort_by(|x, y| {
                x.ignoring_truthiness_marker()
                    .cmp(y.ignoring_truthiness_marker())
                    .then_with(|| x.cmp(y))
            });
            members.sort_by(|x, y| cmp_union_members(x, y, &ctx));
            assert_eq!(members, expected);
        }
    }
}
