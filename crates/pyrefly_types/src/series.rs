/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Element dtypes attached to otherwise-opaque Series instances.

use pyrefly_derive::TypeEq;
use pyrefly_derive::TypeOrd;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;

use crate::class::ClassType;
use crate::polars_dtype::PolarsDType;
use crate::types::Type;

/// A Series instance carrying its element dtype.
#[derive(
    Debug, PartialOrd, Ord, Clone, Eq, PartialEq, Hash, Visit, VisitMut, TypeEq, TypeOrd
)]
pub struct SeriesSchema {
    pub underlying: ClassType,
    pub dtype: PolarsDType,
}

impl SeriesSchema {
    pub fn to_type(self) -> Type {
        Type::Series(Box::new(self))
    }

    pub fn underlying_type(&self) -> Type {
        Type::ClassType(self.underlying.clone())
    }
}

#[cfg(test)]
mod tests {
    use pyrefly_util::display::DisplayWithCtx;

    use super::*;
    use crate::class::ClassType;
    use crate::test_type_env::TestTypeEnv;
    use crate::types::TArgs;

    fn underlying_class(env: &TestTypeEnv) -> ClassType {
        ClassType::new(env.class("polars", "Series"), TArgs::default())
    }

    fn series(env: &TestTypeEnv, dtype: PolarsDType) -> SeriesSchema {
        SeriesSchema {
            underlying: underlying_class(env),
            dtype,
        }
    }

    #[test]
    fn display_shows_the_element_dtype() {
        let env = TestTypeEnv::new();
        let s = series(&env, PolarsDType::Int64).to_type();
        assert_eq!(s.display_with(&env).to_string(), "Series[Int64]");
    }

    #[test]
    fn strip_replaces_the_schema_with_its_class() {
        let env = TestTypeEnv::new();
        let underlying = Type::ClassType(underlying_class(&env));
        assert_eq!(
            series(&env, PolarsDType::Int64)
                .to_type()
                .strip_library_schemas(),
            underlying
        );
        let optional = Type::optional(series(&env, PolarsDType::String).to_type());
        assert_eq!(optional.strip_library_schemas(), Type::optional(underlying));
    }

    #[test]
    fn traversal_preserves_underlying() {
        let env = TestTypeEnv::new();
        let s = series(&env, PolarsDType::Int64).to_type();
        let Type::Series(schema) = s else {
            unreachable!("to_type produces the Series variant")
        };
        assert_eq!(schema.underlying, underlying_class(&env));
        assert_eq!(schema.dtype, PolarsDType::Int64);
    }
}
