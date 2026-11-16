/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Column schemas attached to otherwise-opaque DataFrame instances.

use pyrefly_derive::TypeEq;
use pyrefly_derive::TypeOrd;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use ruff_python_ast::name::Name;

use crate::class::ClassType;
use crate::polars_dtype::PolarsDType;
use crate::types::Type;

/// Whether `columns` is exhaustive.
#[derive(
    Debug, PartialOrd, Ord, Clone, Copy, Eq, PartialEq, Hash, Visit, VisitMut, TypeEq, TypeOrd
)]
pub enum SchemaCompleteness {
    Complete,
    Partial,
}

impl SchemaCompleteness {
    pub fn combine(self, other: Self) -> Self {
        match (self, other) {
            (Self::Complete, Self::Complete) => Self::Complete,
            _ => Self::Partial,
        }
    }
}

#[derive(
    Debug, PartialOrd, Ord, Clone, Copy, Eq, PartialEq, Hash, Visit, VisitMut, TypeEq, TypeOrd
)]
pub enum DataFrameKind {
    Polars,
    Narwhals,
    Pandas,
}

impl DataFrameKind {
    pub fn is_polars_api(self) -> bool {
        matches!(self, Self::Polars | Self::Narwhals)
    }
}

/// Whether a schema is inferred or declared by an explicit annotation.
#[derive(
    Debug, PartialOrd, Ord, Clone, Copy, Eq, PartialEq, Hash, Visit, VisitMut, TypeEq, TypeOrd
)]
pub enum SchemaRole {
    Inferred,
    Contract,
}

/// A DataFrame with ordered column information.
#[derive(
    Debug, PartialOrd, Ord, Clone, Eq, PartialEq, Hash, Visit, VisitMut, TypeEq, TypeOrd
)]
pub struct DataFrameSchema {
    pub underlying: ClassType,
    pub columns: Vec<(Name, PolarsDType)>,
    pub completeness: SchemaCompleteness,
    pub kind: DataFrameKind,
    pub role: SchemaRole,
}

impl DataFrameSchema {
    pub fn to_type(self) -> Type {
        Type::DataFrame(Box::new(self))
    }

    pub fn underlying_type(&self) -> Type {
        Type::ClassType(self.underlying.clone())
    }

    pub fn is_complete(&self) -> bool {
        self.completeness == SchemaCompleteness::Complete
    }

    pub fn is_contract(&self) -> bool {
        self.role == SchemaRole::Contract
    }

    pub fn has_column(&self, name: &Name) -> bool {
        self.columns.iter().any(|(c, _)| c == name)
    }

    /// Whether this schema guarantees the contract described by `target`.
    pub fn satisfies(&self, target: &Self) -> bool {
        if self.kind != target.kind {
            return false;
        }
        match target.completeness {
            SchemaCompleteness::Complete => {
                self.completeness == SchemaCompleteness::Complete && self.columns == target.columns
            }
            SchemaCompleteness::Partial => target.columns.iter().all(|required| {
                self.columns
                    .iter()
                    .any(|actual| actual.0 == required.0 && actual.1 == required.1)
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hash;
    use std::hash::Hasher;

    use pyrefly_util::display::DisplayWithCtx;
    use ruff_python_ast::name::Name;

    use super::*;
    use crate::class::ClassType;
    use crate::equality::TypeEq;
    use crate::equality::TypeEqCtx;
    use crate::test_type_env::TestTypeEnv;
    use crate::types::TArgs;

    fn underlying_class(env: &TestTypeEnv) -> ClassType {
        ClassType::new(env.class("polars", "DataFrame"), TArgs::default())
    }

    fn col(name: &str, dtype: PolarsDType) -> (Name, PolarsDType) {
        (Name::new(name), dtype)
    }

    fn schema(
        env: &TestTypeEnv,
        columns: Vec<(Name, PolarsDType)>,
        completeness: SchemaCompleteness,
    ) -> DataFrameSchema {
        DataFrameSchema {
            underlying: underlying_class(env),
            columns,
            completeness,
            kind: DataFrameKind::Polars,
            role: SchemaRole::Inferred,
        }
    }

    fn hash_of(schema: &DataFrameSchema) -> u64 {
        let mut hasher = DefaultHasher::new();
        schema.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn partial_schema_display_shows_trailing_marker() {
        let env = TestTypeEnv::new();
        let df = schema(
            &env,
            vec![col("a", PolarsDType::Int64)],
            SchemaCompleteness::Partial,
        )
        .to_type();
        assert_eq!(
            df.display_with(&env).to_string(),
            "DataFrame[a: Int64, ...]",
            "a Partial schema shows known columns plus a trailing marker for the unknown rest"
        );
    }

    #[test]
    fn column_order_is_part_of_identity() {
        let env = TestTypeEnv::new();
        let ab = schema(
            &env,
            vec![col("a", PolarsDType::Int64), col("b", PolarsDType::String)],
            SchemaCompleteness::Complete,
        );
        let ba = schema(
            &env,
            vec![col("b", PolarsDType::String), col("a", PolarsDType::Int64)],
            SchemaCompleteness::Complete,
        );

        assert_ne!(ab, ba);
        assert_ne!(hash_of(&ab), hash_of(&ba));
        assert_ne!(ab.cmp(&ba), Ordering::Equal);
        assert!(!ab.type_eq(&ba, &mut TypeEqCtx::default()));

        let ab2 = schema(
            &env,
            vec![col("a", PolarsDType::Int64), col("b", PolarsDType::String)],
            SchemaCompleteness::Complete,
        );
        assert_eq!(ab, ab2);
        assert_eq!(hash_of(&ab), hash_of(&ab2));
        assert_eq!(ab.cmp(&ab2), Ordering::Equal);
        assert!(ab.type_eq(&ab2, &mut TypeEqCtx::default()));
    }

    #[test]
    fn completeness_is_part_of_identity() {
        let env = TestTypeEnv::new();
        let complete = schema(
            &env,
            vec![col("a", PolarsDType::Int64)],
            SchemaCompleteness::Complete,
        );
        let partial = schema(
            &env,
            vec![col("a", PolarsDType::Int64)],
            SchemaCompleteness::Partial,
        );
        assert_ne!(complete, partial);
        assert!(!complete.type_eq(&partial, &mut TypeEqCtx::default()));
    }

    #[test]
    fn role_is_part_of_identity() {
        let env = TestTypeEnv::new();
        let inferred = schema(
            &env,
            vec![col("a", PolarsDType::Int64)],
            SchemaCompleteness::Complete,
        );
        let contract = DataFrameSchema {
            role: SchemaRole::Contract,
            ..inferred.clone()
        };
        assert_ne!(inferred, contract);
        assert!(!inferred.type_eq(&contract, &mut TypeEqCtx::default()));
    }

    #[test]
    fn combining_completeness_requires_both_schemas_to_be_complete() {
        assert_eq!(
            SchemaCompleteness::Complete.combine(SchemaCompleteness::Complete),
            SchemaCompleteness::Complete
        );
        assert_eq!(
            SchemaCompleteness::Complete.combine(SchemaCompleteness::Partial),
            SchemaCompleteness::Partial
        );
        assert_eq!(
            SchemaCompleteness::Partial.combine(SchemaCompleteness::Complete),
            SchemaCompleteness::Partial
        );
        assert_eq!(
            SchemaCompleteness::Partial.combine(SchemaCompleteness::Partial),
            SchemaCompleteness::Partial
        );
    }

    #[test]
    fn exact_schema_requires_identical_complete_columns() {
        let env = TestTypeEnv::new();
        let target = schema(
            &env,
            vec![col("a", PolarsDType::Int64), col("b", PolarsDType::String)],
            SchemaCompleteness::Complete,
        );
        assert!(target.satisfies(&target));
        assert!(
            !schema(
                &env,
                vec![col("b", PolarsDType::String), col("a", PolarsDType::Int64)],
                SchemaCompleteness::Complete,
            )
            .satisfies(&target)
        );
        assert!(
            !schema(
                &env,
                vec![col("a", PolarsDType::Int64), col("b", PolarsDType::String)],
                SchemaCompleteness::Partial,
            )
            .satisfies(&target)
        );
        assert!(
            !schema(
                &env,
                vec![
                    col("a", PolarsDType::Int64),
                    col("b", PolarsDType::String),
                    col("c", PolarsDType::Boolean),
                ],
                SchemaCompleteness::Complete,
            )
            .satisfies(&target)
        );
    }

    #[test]
    fn open_schema_requires_known_columns_by_name_and_dtype() {
        let env = TestTypeEnv::new();
        let target = schema(
            &env,
            vec![col("a", PolarsDType::Int64), col("b", PolarsDType::String)],
            SchemaCompleteness::Partial,
        );
        for completeness in [SchemaCompleteness::Complete, SchemaCompleteness::Partial] {
            assert!(
                schema(
                    &env,
                    vec![
                        col("extra", PolarsDType::Boolean),
                        col("b", PolarsDType::String),
                        col("a", PolarsDType::Int64),
                    ],
                    completeness,
                )
                .satisfies(&target)
            );
        }
        assert!(
            !schema(
                &env,
                vec![col("a", PolarsDType::Int64)],
                SchemaCompleteness::Complete,
            )
            .satisfies(&target)
        );
        assert!(
            !schema(
                &env,
                vec![col("a", PolarsDType::Int64), col("b", PolarsDType::Unknown)],
                SchemaCompleteness::Partial,
            )
            .satisfies(&target)
        );
    }

    #[test]
    fn kind_is_part_of_identity() {
        let env = TestTypeEnv::new();
        let cols = || vec![col("a", PolarsDType::Int64)];
        let polars = schema(&env, cols(), SchemaCompleteness::Complete);
        let pandas = DataFrameSchema {
            kind: DataFrameKind::Pandas,
            ..schema(&env, cols(), SchemaCompleteness::Complete)
        };
        assert_ne!(polars, pandas);
        assert!(!polars.type_eq(&pandas, &mut TypeEqCtx::default()));
    }

    #[test]
    fn strip_library_schemas_replaces_every_schema_with_its_class() {
        let env = TestTypeEnv::new();
        let underlying = Type::ClassType(underlying_class(&env));

        for (columns, completeness) in [
            (
                vec![col("a", PolarsDType::Int64)],
                SchemaCompleteness::Complete,
            ),
            (vec![], SchemaCompleteness::Complete),
            (
                vec![col("a", PolarsDType::Int64)],
                SchemaCompleteness::Partial,
            ),
        ] {
            let stripped = schema(&env, columns, completeness)
                .to_type()
                .strip_library_schemas();
            assert_eq!(stripped, underlying);
        }

        let pandas = DataFrameSchema {
            kind: DataFrameKind::Pandas,
            ..schema(
                &env,
                vec![col("a", PolarsDType::Int64)],
                SchemaCompleteness::Complete,
            )
        }
        .to_type();
        assert_eq!(pandas.strip_library_schemas(), underlying);

        let optional = Type::optional(
            schema(
                &env,
                vec![col("a", PolarsDType::Int64)],
                SchemaCompleteness::Complete,
            )
            .to_type(),
        );
        assert_eq!(
            optional.strip_library_schemas(),
            Type::optional(underlying.clone())
        );
        assert_eq!(underlying.clone().strip_library_schemas(), underlying);
    }

    #[test]
    fn traversal_preserves_underlying() {
        let env = TestTypeEnv::new();
        let df = schema(
            &env,
            vec![col("a", PolarsDType::Int64)],
            SchemaCompleteness::Complete,
        )
        .to_type();
        let Type::DataFrame(s) = df else {
            unreachable!("to_type produces the DataFrame variant")
        };
        assert_eq!(s.underlying, underlying_class(&env));
        assert_eq!(s.columns[0].1, PolarsDType::Int64);
    }
}
