/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cmp::Ordering;
use std::fmt;
use std::fmt::Display;
use std::hash::Hash;
use std::hash::Hasher;

use pyrefly_derive::TypeEq;
use pyrefly_derive::Visit;
use pyrefly_derive::VisitMut;
use pyrefly_util::display::commas_iter;

use crate::class::ClassType;
use crate::dimension::ShapeError;
use crate::dimension::SymInt;
use crate::dimension::canonicalize;
use crate::dimension::gradual_size;
use crate::dimension::is_gradual_size;
use crate::lit_int::LitInt;
use crate::literal::Lit;
use crate::quantified::QuantifiedKind;
use crate::tuple::Tuple;
use crate::types::Type;

// ============================================================================
// Shaped Array Types
// ============================================================================

/// Whether a shaped-array type was constructed using native (`Tensor[N, M]`) or
/// jaxtyping (`Float[Tensor, "N M"]`) syntax. Controls display rendering and
/// enables diagnostic checks (e.g., mixing both syntaxes in one function).
///
/// Transparent to equality, hashing, and ordering — syntax does not affect
/// type identity. Two shaped-array types with different syntax but identical base
/// class and shape are considered equal.
#[derive(Debug, Clone, Copy, Default)]
#[derive(Visit, VisitMut)]
pub enum ShapedArraySyntax {
    #[default]
    Native,
    Jaxtyping,
}

// Syntax is a display/diagnostic concern, not a type identity concern.
// All trait impls treat every ShapedArraySyntax value as equal.

impl PartialEq for ShapedArraySyntax {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Eq for ShapedArraySyntax {}

impl Hash for ShapedArraySyntax {
    fn hash<H: Hasher>(&self, _state: &mut H) {}
}

impl PartialOrd for ShapedArraySyntax {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ShapedArraySyntax {
    fn cmp(&self, _other: &Self) -> Ordering {
        Ordering::Equal
    }
}

impl crate::equality::TypeEq for ShapedArraySyntax {
    fn type_eq(&self, _other: &Self, _ctx: &mut crate::equality::TypeEqCtx) -> bool {
        true
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
enum ShapedArrayShapeStorage {
    Inline(SymIntTuple),
    TupleCarrier { index: usize },
}

/// A class instance with shape information.
/// Example: Tensor[[2, 3]] represents a 2x3 tensor
/// Example: Tensor (no brackets) represents a shapeless tensor (`SymIntTuple`)
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct ShapedArrayType {
    /// Base shaped-array class (e.g., torch.Tensor)
    pub base_class: ClassType,
    shape: ShapedArrayShapeStorage,
    /// Whether this type was constructed from native or jaxtyping syntax.
    pub syntax: ShapedArraySyntax,
}

impl ShapedArrayType {
    /// Create a shaped-array type with shape information (defaults to Native syntax).
    pub fn new(base_class: ClassType, shape: SymIntTuple) -> Self {
        Self {
            base_class,
            shape: ShapedArrayShapeStorage::Inline(shape),
            syntax: ShapedArraySyntax::Native,
        }
    }

    /// Create a shapeless shaped-array type (compatible with any shape).
    pub fn shapeless(base_class: ClassType) -> Self {
        Self {
            base_class,
            shape: ShapedArrayShapeStorage::Inline(SymIntTuple::shapeless()),
            syntax: ShapedArraySyntax::Native,
        }
    }

    /// Set the syntax for this shaped-array type.
    pub fn with_syntax(mut self, syntax: ShapedArraySyntax) -> Self {
        self.syntax = syntax;
        self
    }

    pub fn with_tuple_carrier_shape_arg(mut self, index: usize) -> Self {
        self.shape = ShapedArrayShapeStorage::TupleCarrier { index };
        self
    }

    pub fn to_type(self) -> Type {
        Type::ShapedArray(Box::new(self))
    }

    pub fn tuple_carrier_shape_arg_index(&self) -> Option<usize> {
        match self.shape {
            ShapedArrayShapeStorage::Inline(_) => None,
            ShapedArrayShapeStorage::TupleCarrier { index } => Some(index),
        }
    }

    pub fn set_tuple_carrier_shape_arg(&mut self, index: usize) {
        self.shape = ShapedArrayShapeStorage::TupleCarrier { index };
    }

    pub fn shape(&self) -> SymIntTuple {
        match &self.shape {
            ShapedArrayShapeStorage::Inline(shape) => shape.clone(),
            ShapedArrayShapeStorage::TupleCarrier { index } => {
                let shape_arg = self
                    .base_class
                    .targs()
                    .as_slice()
                    .get(*index)
                    .expect("shape argument index should point to a class type argument");
                SymIntTuple::from_shape_arg_type(shape_arg)
                    .or_else(|| tuple_carrier_to_shape(shape_arg))
                    .expect("registered shaped-array shape argument should project to SymIntTuple")
            }
        }
    }

    pub fn set_shape(&mut self, shape: SymIntTuple) {
        match &mut self.shape {
            ShapedArrayShapeStorage::Inline(stored_shape) => *stored_shape = shape,
            ShapedArrayShapeStorage::TupleCarrier { index } => {
                let shape_arg = self
                    .base_class
                    .targs_mut()
                    .as_mut()
                    .get_mut(*index)
                    .expect("shape argument index should point to a class type argument");
                *shape_arg = shape.to_shape_arg_type();
            }
        }
    }

    /// Returns rank if shape is concrete, None for variadic/shapeless
    pub fn rank(&self) -> Option<usize> {
        match self.shape().view() {
            SymIntTupleView::Concrete(dims) => Some(dims.len()),
            SymIntTupleView::Gradual | SymIntTupleView::Unpacked { .. } => None,
        }
    }

    /// Returns true if the shaped array has no shape information.
    /// (represented as a gradual `SymIntTuple`)
    pub fn is_shapeless(&self) -> bool {
        is_shapeless(&self.shape())
    }
}

impl Display for ShapedArrayType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.syntax {
            ShapedArraySyntax::Native => {
                let shape = self.shape();
                if is_shapeless(&shape) {
                    write!(f, "{}", self.base_class.name())
                } else if self.tuple_carrier_shape_arg_index().is_some() {
                    write!(
                        f,
                        "{}[{}]",
                        self.base_class.name(),
                        fmt_tuple_carrier(&shape)
                    )
                } else {
                    write!(f, "{}[{}]", self.base_class.name(), shape)
                }
            }
            ShapedArraySyntax::Jaxtyping => {
                let shape = self.shape();
                write!(
                    f,
                    "Shaped[{}, \"{}\"]",
                    self.base_class.name(),
                    shape.fmt_jaxtyping()
                )
            }
        }
    }
}

/// Shape of a shaped array.
///
/// The storage is deliberately not a `Tuple`: fixed dimensions are always
/// canonical `SymInt`s, while variadic middles carry the original tuple/type
/// variable shape carrier.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
pub struct SymIntTuple(SymIntTupleRepr);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Visit, VisitMut, TypeEq)]
enum SymIntTupleRepr {
    Concrete(Vec<SymInt>),
    Gradual,
    Unpacked {
        prefix: Vec<SymInt>,
        middle: Box<Type>,
        suffix: Vec<SymInt>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymIntTupleView<'a> {
    Concrete(&'a [SymInt]),
    Gradual,
    Unpacked {
        prefix: &'a [SymInt],
        middle: &'a Type,
        suffix: &'a [SymInt],
    },
}

impl SymIntTuple {
    pub fn new(dims: Vec<SymInt>) -> Self {
        Self::from_symints(dims)
    }

    /// Create from Vec<Type> directly (for when dims are already wrapped)
    /// Automatically normalizes dimensions to canonical form:
    /// - Canonicalizes `SymInt` expressions (e.g., 2+3 -> 5, N+0 -> N)
    /// - Wraps scalar symbolic dimensions in `SymInt::Symbolic`
    pub fn from_types(dims: Vec<Type>) -> Self {
        Self::from_symints(dims.into_iter().map(type_to_dim_recover).collect())
    }

    fn from_symints(dims: Vec<SymInt>) -> Self {
        Self(SymIntTupleRepr::Concrete(
            dims.into_iter().map(canonicalize_symint_dim).collect(),
        ))
    }

    /// Rebuild this shape through the canonical constructors.
    pub fn normalize(&self) -> Self {
        match &self.0 {
            SymIntTupleRepr::Concrete(dims) => Self::from_symints(dims.clone()),
            SymIntTupleRepr::Gradual => Self::shapeless(),
            SymIntTupleRepr::Unpacked {
                prefix,
                middle,
                suffix,
            } => Self::unpacked(
                dims_to_types(prefix),
                middle.as_ref().clone(),
                dims_to_types(suffix),
            ),
        }
    }

    fn unpacked_from_parts(prefix: Vec<SymInt>, middle: Type, suffix: Vec<SymInt>) -> Self {
        if prefix.is_empty() && suffix.is_empty() && is_gradual_shape_middle(&middle) {
            Self::shapeless()
        } else {
            Self(SymIntTupleRepr::Unpacked {
                prefix,
                middle: Box::new(middle),
                suffix,
            })
        }
    }

    pub fn from_tuple(tuple: Tuple) -> Self {
        match tuple {
            Tuple::Concrete(dims) => Self::from_types(dims),
            Tuple::Unpacked(unpacked) => {
                let (prefix, middle, suffix) = *unpacked;
                Self::unpacked(prefix, middle, suffix)
            }
            Tuple::Unbounded(elt) if elt.is_any() || is_gradual_size(&elt) => Self::shapeless(),
            Tuple::Unbounded(elt) => {
                Self::unpacked(Vec::new(), Type::Tuple(Tuple::Unbounded(elt)), Vec::new())
            }
        }
    }

    pub fn shapeless() -> Self {
        Self(SymIntTupleRepr::Gradual)
    }

    pub fn is_shapeless(&self) -> bool {
        is_shapeless(self)
    }

    pub fn view(&self) -> SymIntTupleView<'_> {
        match &self.0 {
            SymIntTupleRepr::Concrete(dims) => SymIntTupleView::Concrete(dims),
            SymIntTupleRepr::Gradual => SymIntTupleView::Gradual,
            SymIntTupleRepr::Unpacked {
                prefix,
                middle,
                suffix,
            } => SymIntTupleView::Unpacked {
                prefix,
                middle,
                suffix,
            },
        }
    }

    /// Project this shape to the ordinary tuple type it denotes.
    pub fn to_tuple_type(&self) -> Type {
        match &self.0 {
            SymIntTupleRepr::Concrete(dims) => Type::Tuple(Tuple::Concrete(dims_to_types(dims))),
            SymIntTupleRepr::Gradual => Type::Tuple(Tuple::Unbounded(Box::new(gradual_size()))),
            SymIntTupleRepr::Unpacked {
                prefix,
                middle,
                suffix,
            } => {
                let middle = match middle.as_ref() {
                    Type::SymIntTuple(shape) => shape.to_tuple_type(),
                    middle if is_tuple_carrier_shape_middle(middle) => {
                        Type::Tuple(Tuple::Unbounded(Box::new(gradual_size())))
                    }
                    middle => middle.clone(),
                };
                Type::Tuple(Tuple::Unpacked(Box::new((
                    dims_to_types(prefix),
                    middle,
                    dims_to_types(suffix),
                ))))
            }
        }
    }

    pub fn to_tuple(&self) -> Tuple {
        let Type::Tuple(tuple) = self.to_tuple_type() else {
            unreachable!("SymIntTuple always projects to a tuple")
        };
        tuple
    }

    pub fn to_shape_arg_type(&self) -> Type {
        Type::SymIntTuple(Box::new(self.clone()))
    }

    pub fn from_shape_arg_type(arg: &Type) -> Option<Self> {
        match arg {
            Type::SymIntTuple(shape) => Some(shape.normalize()),
            _ => None,
        }
    }

    /// Create variadic shape with unpacked TypeVarTuple: Tensor[2, *Shape, 4]
    pub fn unpacked(prefix: Vec<Type>, middle: Type, suffix: Vec<Type>) -> Self {
        // Canonicalize all dimensions
        let mut prefix: Vec<SymInt> = prefix.into_iter().map(type_to_dim_recover).collect();
        let mut suffix: Vec<SymInt> = suffix.into_iter().map(type_to_dim_recover).collect();

        if let Type::SymIntTuple(shape) = &middle {
            match shape.view() {
                SymIntTupleView::Concrete(dims) => {
                    prefix.extend_from_slice(dims);
                    prefix.extend(suffix);
                    return Self::from_symints(prefix);
                }
                SymIntTupleView::Gradual => {
                    return Self::unpacked_from_parts(prefix, gradual_shape_middle(), suffix);
                }
                SymIntTupleView::Unpacked {
                    prefix: inner_prefix,
                    middle: inner_middle,
                    suffix: inner_suffix,
                } => {
                    prefix.extend_from_slice(inner_prefix);
                    let mut combined_suffix = inner_suffix.to_vec();
                    combined_suffix.append(&mut suffix);
                    return Self::unpacked(
                        dims_to_types(&prefix),
                        inner_middle.clone(),
                        dims_to_types(&combined_suffix),
                    );
                }
            }
        }

        if let Type::Tuple(Tuple::Concrete(dims)) = &middle {
            let dims = dims.iter().map(carrier_element_to_dim_recover);
            prefix.extend(dims);
            prefix.extend(suffix);
            return Self::from_symints(prefix);
        }

        if let Type::Tuple(Tuple::Unpacked(unpacked)) = &middle {
            let (inner_prefix, inner_middle, inner_suffix) = &**unpacked;
            prefix.extend(inner_prefix.iter().map(carrier_element_to_dim_recover));
            let mut combined_suffix: Vec<SymInt> = inner_suffix
                .iter()
                .map(carrier_element_to_dim_recover)
                .collect();
            combined_suffix.append(&mut suffix);
            return Self::unpacked(
                dims_to_types(&prefix),
                inner_middle.clone(),
                dims_to_types(&combined_suffix),
            );
        }

        match middle {
            Type::Tuple(Tuple::Unbounded(elt)) => {
                let dim = unbounded_middle_element_to_dim(&elt);
                if matches!(dim, SymInt::Int) {
                    Self::unpacked_from_parts(prefix, gradual_shape_middle(), suffix)
                } else {
                    let middle = Type::Tuple(Tuple::Unbounded(Box::new(dim_to_type(&dim))));
                    Self::unpacked_from_parts(prefix, middle, suffix)
                }
            }
            Type::Any(_) => Self::unpacked_from_parts(prefix, gradual_shape_middle(), suffix),
            middle if is_unresolved_shape_middle(&middle) => {
                Self::unpacked_from_parts(prefix, middle, suffix)
            }
            _ => Self::shapeless(),
        }
    }

    pub fn rank(&self) -> usize {
        match &self.0 {
            SymIntTupleRepr::Concrete(dims) => dims.len(),
            SymIntTupleRepr::Gradual | SymIntTupleRepr::Unpacked { .. } => {
                // For unpacked shapes, rank is unknown at parse time
                // This should not be called for variadic shapes
                panic!("Cannot determine rank of variadic tensor shape")
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        match &self.0 {
            SymIntTupleRepr::Concrete(dims) => dims.is_empty(),
            SymIntTupleRepr::Gradual | SymIntTupleRepr::Unpacked { .. } => false,
        }
    }

    /// Get a slice of dimensions (only valid for concrete shapes)
    pub fn dims_slice(&self) -> &[SymInt] {
        match &self.0 {
            SymIntTupleRepr::Concrete(dims) => dims,
            SymIntTupleRepr::Gradual | SymIntTupleRepr::Unpacked { .. } => {
                panic!("Cannot get dims_slice for variadic tensor shape")
            }
        }
    }

    /// Get the concrete dims if this is a concrete shape.
    pub fn as_concrete(&self) -> Option<&[SymInt]> {
        match &self.0 {
            SymIntTupleRepr::Concrete(dims) => Some(dims),
            SymIntTupleRepr::Gradual | SymIntTupleRepr::Unpacked { .. } => None,
        }
    }

    /// Get a mutable reference to concrete dims (for meta-shape operations)
    /// Panics if called on Unpacked shape
    pub fn dims_mut(&mut self) -> &mut Vec<SymInt> {
        match &mut self.0 {
            SymIntTupleRepr::Concrete(dims) => dims,
            SymIntTupleRepr::Gradual | SymIntTupleRepr::Unpacked { .. } => {
                panic!("Cannot get mutable dims for variadic tensor shape")
            }
        }
    }

    /// Get dims as a Vec for concrete shapes, panics for unpacked
    /// This is used by meta-shape code that doesn't support variadic shapes yet
    pub fn dims(&self) -> &Vec<SymInt> {
        match &self.0 {
            SymIntTupleRepr::Concrete(dims) => dims,
            SymIntTupleRepr::Gradual | SymIntTupleRepr::Unpacked { .. } => {
                panic!("Meta-shape operations do not yet support variadic tensor shapes")
            }
        }
    }

    /// Check if all dimensions are literal (concrete integers)
    /// Returns false for variadic shapes
    pub fn all_literal(&self) -> bool {
        match &self.0 {
            SymIntTupleRepr::Concrete(dims) => {
                dims.iter().all(|dim| matches!(dim, SymInt::Literal(_)))
            }
            SymIntTupleRepr::Gradual | SymIntTupleRepr::Unpacked { .. } => false,
        }
    }

    /// Extract literal dimension values if all are literal
    /// Returns None for variadic shapes
    pub fn as_literals(&self) -> Option<Vec<i64>> {
        match &self.0 {
            SymIntTupleRepr::Concrete(dims) if self.all_literal() => Some(
                dims.iter()
                    .map(|dim| match dim {
                        SymInt::Literal(n) => *n,
                        _ => unreachable!("all_literal checked every concrete dimension"),
                    })
                    .collect(),
            ),
            _ => None,
        }
    }

    /// Get a dimension by index (only for concrete shapes)
    pub fn get_dim(&self, index: usize) -> Type {
        match &self.0 {
            SymIntTupleRepr::Concrete(dims) => dim_to_type(dims.get(index).unwrap()),
            SymIntTupleRepr::Gradual | SymIntTupleRepr::Unpacked { .. } => {
                panic!("Cannot get dimension by index for variadic tensor shape")
            }
        }
    }

    /// Normalize a dimension index to handle negative indices
    ///
    /// Negative indices count from the end: -1 is the last dimension, -2 is second-to-last, etc.
    /// Returns an error if the index is out of range.
    pub fn normalize_dim(&self, dim: i64) -> Result<usize, ShapeError> {
        // Check for variadic shape first - cannot normalize dims for unpacked shapes
        if !matches!(self.0, SymIntTupleRepr::Concrete(_)) {
            return Err(ShapeError::InvalidDimension {
                value: dim,
                reason: "Cannot normalize dimension index for variadic tensor shape".to_owned(),
            });
        }

        let rank = self.rank() as i64;

        if rank == 0 {
            return Err(ShapeError::InvalidDimension {
                value: dim,
                reason: "Cannot normalize dimension for scalar tensor (rank 0)".to_owned(),
            });
        }

        let normalized = if dim < 0 { rank + dim } else { dim };

        if normalized < 0 || normalized >= rank {
            return Err(ShapeError::InvalidDimension {
                value: dim,
                reason: format!(
                    "Dimension {} out of range for tensor with rank {} (valid range: {} to {})",
                    dim,
                    rank,
                    -rank,
                    rank - 1
                ),
            });
        }

        Ok(normalized as usize)
    }

    /// Format the shape using jaxtyping syntax (space-separated, no parens for scalar).
    ///
    /// Handles all jaxtyping dimension types:
    /// - `Type::Any` → `_` (anonymous dim)
    /// - `Type::SymInt(Literal(n))` → `n`
    /// - `Type::SymInt(Add/Sub)` → `a+b` / `a-b` (no parens, no spaces)
    /// - `Type::Quantified` → dim name
    /// - Unpacked with gradual `SymIntTuple` middle → `...` (ellipsis)
    /// - Unpacked with TypeVarTuple middle → `*name`
    pub fn fmt_jaxtyping(&self) -> String {
        match &self.0 {
            SymIntTupleRepr::Concrete(dims) => {
                if dims.is_empty() {
                    String::new() // Scalar: empty string inside quotes
                } else {
                    dims.iter()
                        .map(fmt_jaxtyping_symint)
                        .collect::<Vec<_>>()
                        .join(" ")
                }
            }
            SymIntTupleRepr::Gradual => "...".to_owned(),
            SymIntTupleRepr::Unpacked {
                prefix,
                middle,
                suffix,
            } => {
                let mut parts: Vec<String> = prefix.iter().map(fmt_jaxtyping_symint).collect();

                // Ellipsis: a gradual `SymIntTuple` middle renders as "..."
                // Named TypeVarTuple renders as "*name"
                if is_gradual_shape_middle(middle) {
                    parts.push("...".to_owned());
                } else {
                    parts.push(format!("*{middle}"));
                }

                parts.extend(suffix.iter().map(fmt_jaxtyping_symint));
                parts.join(" ")
            }
        }
    }
}

pub(crate) fn fmt_shape_dim(d: &SymInt) -> String {
    format!("{d}")
}

/// Format a `SymInt` in jaxtyping syntax (no parens, no spaces around operators).
fn fmt_jaxtyping_symint(expr: &SymInt) -> String {
    match expr {
        SymInt::Literal(n) => n.to_string(),
        SymInt::Int => "_".to_owned(),
        SymInt::Symbolic(ty) => format!("{ty}"),
        SymInt::Add(left, right) => {
            // After canonicalization, Sub(a,b) becomes Add(Literal(-b), a).
            // Detect this and render as subtraction: Add(-n, x) → x-n
            if let SymInt::Literal(n) = left.as_ref()
                && *n < 0
            {
                return format!("{}-{}", fmt_jaxtyping_symint(right), n.wrapping_neg());
            }
            format!(
                "{}+{}",
                fmt_jaxtyping_symint(left),
                fmt_jaxtyping_symint(right)
            )
        }
        SymInt::Sub(left, right) => {
            format!(
                "{}-{}",
                fmt_jaxtyping_symint(left),
                fmt_jaxtyping_symint(right)
            )
        }
        // Mul/FloorDiv fall back to default `SymInt` display (rare in jaxtyping)
        _ => format!("{expr}"),
    }
}

impl Display for SymIntTuple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            SymIntTupleRepr::Concrete(dims) => {
                if dims.is_empty() {
                    write!(f, "()") // Scalar tensor: Tensor[()]
                } else {
                    write!(f, "{}", commas_iter(|| dims.iter().map(fmt_shape_dim)))
                }
            }
            SymIntTupleRepr::Gradual => write!(f, "*SymIntTuple"),
            SymIntTupleRepr::Unpacked {
                prefix,
                middle,
                suffix,
            } => {
                let prefix_str = if prefix.is_empty() {
                    "".to_owned()
                } else {
                    format!("{}, ", commas_iter(|| prefix.iter().map(fmt_shape_dim)))
                };
                let suffix_str = if suffix.is_empty() {
                    "".to_owned()
                } else {
                    format!(", {}", commas_iter(|| suffix.iter().map(fmt_shape_dim)))
                };
                write!(
                    f,
                    "{}*{}{}",
                    prefix_str,
                    fmt_unpacked_middle(middle),
                    suffix_str
                )
            }
        }
    }
}

fn fmt_unpacked_middle(middle: &Type) -> String {
    match middle {
        Type::SymIntTuple(shape) if shape.is_shapeless() => "tuple[int, ...]".to_owned(),
        Type::Tuple(Tuple::Unbounded(elt)) if elt.is_any() => "tuple[int, ...]".to_owned(),
        Type::Tuple(Tuple::Unbounded(elt)) if is_gradual_size(elt) => "tuple[int, ...]".to_owned(),
        middle if is_tuple_carrier_shape_middle(middle) => format!("Elements[{middle}]"),
        _ => format!("{middle}"),
    }
}

fn fmt_tuple_carrier(shape: &SymIntTuple) -> String {
    match shape.view() {
        SymIntTupleView::Concrete(dims) => {
            format!("[{}]", commas_iter(|| dims.iter().map(fmt_shape_dim)))
        }
        SymIntTupleView::Gradual => {
            // No unbounded shape reaches here: the only caller (`Display`) handles
            // the shapeless `SymIntTuple` case before calling this.
            unreachable!("shapeless SymIntTuple is handled before fmt_tuple_carrier")
        }
        SymIntTupleView::Unpacked {
            prefix,
            middle,
            suffix,
        } => {
            if prefix.is_empty() && suffix.is_empty() && is_tuple_carrier_shape_middle(middle) {
                return middle.to_string();
            }
            let mut parts: Vec<String> = prefix.iter().map(fmt_shape_dim).collect();
            parts.push(format!("*{}", fmt_unpacked_middle(middle)));
            parts.extend(suffix.iter().map(fmt_shape_dim));
            format!("[{}]", parts.join(", "))
        }
    }
}

// ============================================================================
// Tuple-carrier conversion
// ============================================================================
//
// A "tuple carrier" is the user-facing spelling of a shape that NumPy-style
// syntax such as `ndarray[[3, 4, 5], DType]` or
// `ndarray[tuple[Literal[3], Literal[4], Literal[5]], DType]` produces, where
// each dimension is written as `Literal[n]` or `SymInt[x]`. Internally we store
// scalar dimensions as `Type::SymInt`, while variadic middles keep their carrier
// type. These helpers canonicalize between the two representations so the rest
// of the type checker only ever deals with the internal form.

fn canonicalize_symint_dim(dim: SymInt) -> SymInt {
    match canonicalize(Type::SymInt(dim)) {
        Type::SymInt(dim) => dim,
        _ => unreachable!("canonicalizing a SymInt dimension should produce a SymInt"),
    }
}

fn type_to_dim_recover(dim: Type) -> SymInt {
    SymInt::from_type(&dim)
        .map(canonicalize_symint_dim)
        .unwrap_or(SymInt::Int)
}

fn dim_to_type(dim: &SymInt) -> Type {
    Type::SymInt(dim.clone())
}

fn dims_to_types(dims: &[SymInt]) -> Vec<Type> {
    dims.iter().map(dim_to_type).collect()
}

/// Convert an internal shape dimension into its tuple-carrier element.
/// Literal dimensions become `Literal[n]`; other dimensions remain in their
/// canonical internal representation.
fn dim_to_carrier_element(dim: &SymInt) -> Type {
    match dim {
        SymInt::Literal(n) => LitInt::new(*n).to_explicit_type(),
        _ => dim_to_type(dim),
    }
}

fn is_valid_internal_dim(dim: &Type) -> bool {
    match dim {
        Type::SymInt(expr) => is_valid_internal_symint(expr),
        Type::Quantified(q) => q.kind == QuantifiedKind::SymIntVar,
        Type::TypeVar(tv) => tv.kind() == QuantifiedKind::SymIntVar,
        Type::Var(_) | Type::Any(_) => true,
        _ => false,
    }
}

fn is_valid_internal_symint(expr: &SymInt) -> bool {
    match expr {
        SymInt::Literal(_) | SymInt::Int => true,
        SymInt::Symbolic(ty) => is_valid_internal_dim(ty),
        SymInt::Add(left, right)
        | SymInt::Sub(left, right)
        | SymInt::Mul(left, right)
        | SymInt::FloorDiv(left, right)
        | SymInt::Pow(left, right) => {
            is_valid_internal_symint(left) && is_valid_internal_symint(right)
        }
    }
}

/// Convert a single tuple-carrier element into an internal shape dimension.
///
/// Returns `None` for elements that are not valid dimensions (e.g. non-int
/// literals or arbitrary class types) so that conversion fails cleanly instead
/// of silently treating an unrelated type as a dimension.
fn carrier_element_to_dim(carrier: &Type) -> Option<SymInt> {
    match carrier {
        // `Literal[n]` (int) -> internal literal dimension.
        Type::Literal(lit) => match &lit.value {
            Lit::Int(i) => i.as_i64().map(SymInt::Literal),
            _ => None,
        },
        // Dimensions already in internal form pass through unchanged.
        Type::SymInt(expr) if is_valid_internal_symint(expr) => {
            Some(canonicalize_symint_dim(expr.clone()))
        }
        Type::Quantified(q) if q.kind == QuantifiedKind::SymIntVar => {
            Some(type_to_dim_recover(carrier.clone()))
        }
        Type::TypeVar(tv) if tv.kind() == QuantifiedKind::SymIntVar => {
            Some(type_to_dim_recover(carrier.clone()))
        }
        Type::Var(_) => Some(SymInt::Int),
        Type::Any(_) => Some(SymInt::Int),
        Type::ClassType(cls) if cls.is_builtin("int") => Some(SymInt::Int),
        _ => None,
    }
}

fn carrier_element_to_dim_recover(carrier: &Type) -> SymInt {
    carrier_element_to_dim(carrier).unwrap_or(SymInt::Int)
}

/// Convert a `SymIntTuple` into the equivalent tuple-carrier `Type`.
pub fn shape_to_tuple_carrier(shape: &SymIntTuple) -> Type {
    match shape.view() {
        SymIntTupleView::Concrete(dims) => Type::Tuple(Tuple::Concrete(
            dims.iter().map(dim_to_carrier_element).collect(),
        )),
        SymIntTupleView::Gradual => Type::any_tuple(),
        SymIntTupleView::Unpacked {
            prefix,
            middle,
            suffix,
        } => {
            let middle = match middle {
                Type::SymIntTuple(shape) => shape_to_tuple_carrier(shape),
                _ => middle.clone(),
            };
            Type::Tuple(Tuple::Unpacked(Box::new((
                prefix.iter().map(dim_to_carrier_element).collect(),
                middle,
                suffix.iter().map(dim_to_carrier_element).collect(),
            ))))
        }
    }
}

/// Detects a tuple-carrier shape variable occupying the variadic middle of an
/// unpacked shape.
pub fn is_tuple_carrier_shape_middle(ty: &Type) -> bool {
    // An ordinary TypeVar is legal here only as a whole-shape carrier from
    // tuple-carrier syntax, e.g. `Array[S, DType]` -> `Unpacked([], S, [])`.
    // Scalar symbolic dimensions use `SymInt`/`SymIntVar` and must not reach
    // this fallback as bare TypeVars.
    matches!(ty, Type::Var(_))
        || matches!(ty, Type::TypeVar(tv) if tv.kind() == QuantifiedKind::TypeVar)
        || matches!(ty, Type::Quantified(q) if q.kind == QuantifiedKind::TypeVar)
}

fn is_unresolved_shape_middle(ty: &Type) -> bool {
    // Tuple-carrier shape variables are scalar TypeVars syntactically, but in an
    // unpacked shape middle they stand for an unresolved shape tuple.
    matches!(ty, Type::Var(_) | Type::TypeVarTuple(_))
        || matches!(ty, Type::Quantified(q) if q.kind == QuantifiedKind::TypeVarTuple)
        || is_tuple_carrier_shape_middle(ty)
}

fn unbounded_middle_element_to_dim(elt: &Type) -> SymInt {
    if elt.is_any() || matches!(elt, Type::ClassType(cls) if cls.is_builtin("int")) {
        SymInt::Int
    } else {
        carrier_element_to_dim(elt).unwrap_or(SymInt::Int)
    }
}

/// Convert a projected tuple-carrier shape back to the class type argument.
///
/// A tuple-carrier `TypeVar` represents the whole shape tuple, so `ndarray[S,
/// DType]` projects to `Unpacked([], S, [])` but must round-trip back to `S`,
/// not `tuple[*S]`.
pub fn shape_to_tuple_carrier_arg(shape: &SymIntTuple) -> Type {
    match shape.view() {
        SymIntTupleView::Unpacked {
            prefix,
            middle,
            suffix,
        } => {
            if prefix.is_empty() && suffix.is_empty() && is_tuple_carrier_shape_middle(middle) {
                middle.clone()
            } else {
                shape_to_tuple_carrier(shape)
            }
        }
        _ => shape_to_tuple_carrier(shape),
    }
}

/// Convert a tuple-carrier `Type` into a `SymIntTuple`.
///
/// Returns `None` when the carrier is not a tuple or contains an element that is
/// not a valid dimension.
///
/// `tuple[T, ...]` (including `tuple[int, ...]` and `tuple[Any, ...]`)
/// intentionally canonicalizes to the shapeless / unknown-rank shape: an
/// unbounded carrier conveys no recoverable per-dimension information.
pub fn tuple_carrier_to_shape(carrier: &Type) -> Option<SymIntTuple> {
    match carrier {
        Type::Tuple(Tuple::Concrete(elts)) => {
            let dims = elts
                .iter()
                .map(carrier_element_to_dim)
                .collect::<Option<Vec<_>>>()?;
            Some(SymIntTuple::from_symints(dims))
        }
        Type::Tuple(Tuple::Unpacked(unpacked)) => {
            let (prefix, middle, suffix) = &**unpacked;
            let prefix = prefix
                .iter()
                .map(carrier_element_to_dim)
                .collect::<Option<Vec<_>>>()?;
            let suffix = suffix
                .iter()
                .map(carrier_element_to_dim)
                .collect::<Option<Vec<_>>>()?;
            if matches!(middle, Type::Tuple(Tuple::Unbounded(_))) {
                return Some(SymIntTuple::unpacked(
                    dims_to_types(&prefix),
                    gradual_shape_middle(),
                    dims_to_types(&suffix),
                ));
            }
            validate_tuple_carrier_unpacked_middle(middle)?;
            let middle = recover_unbounded_tuple_carrier_middle(middle.clone());
            Some(SymIntTuple::unpacked(
                dims_to_types(&prefix),
                middle,
                dims_to_types(&suffix),
            ))
        }
        Type::Tuple(Tuple::Unbounded(_)) => Some(shapeless_shape()),
        _ if is_tuple_carrier_shape_middle(carrier) => Some(SymIntTuple::unpacked(
            Vec::new(),
            carrier.clone(),
            Vec::new(),
        )),
        _ => None,
    }
}

fn recover_unbounded_tuple_carrier_middle(middle: Type) -> Type {
    match middle {
        Type::Tuple(Tuple::Unpacked(unpacked)) => {
            let (prefix, middle, suffix) = *unpacked;
            Type::Tuple(Tuple::Unpacked(Box::new((
                prefix,
                recover_unbounded_tuple_carrier_middle(middle),
                suffix,
            ))))
        }
        Type::Tuple(Tuple::Unbounded(_)) => gradual_shape_middle(),
        middle => middle,
    }
}

fn validate_tuple_carrier_unpacked_middle(middle: &Type) -> Option<()> {
    match middle {
        Type::Tuple(Tuple::Concrete(elts)) => {
            elts.iter()
                .map(carrier_element_to_dim)
                .collect::<Option<Vec<_>>>()?;
            Some(())
        }
        Type::Tuple(Tuple::Unpacked(unpacked)) => {
            let (prefix, middle, suffix) = &**unpacked;
            prefix
                .iter()
                .map(carrier_element_to_dim)
                .collect::<Option<Vec<_>>>()?;
            suffix
                .iter()
                .map(carrier_element_to_dim)
                .collect::<Option<Vec<_>>>()?;
            validate_tuple_carrier_unpacked_middle(middle)
        }
        Type::Tuple(Tuple::Unbounded(_)) => Some(()),
        middle if is_unresolved_shape_middle(middle) => Some(()),
        Type::SymIntTuple(_) => Some(()),
        _ => None,
    }
}

fn gradual_shape_middle() -> Type {
    SymIntTuple::shapeless().to_shape_arg_type()
}

fn is_gradual_shape_middle(middle: &Type) -> bool {
    match middle {
        Type::SymIntTuple(shape) => shape.is_shapeless(),
        Type::Tuple(Tuple::Unbounded(elt)) => elt.is_any() || is_gradual_size(elt),
        _ => false,
    }
}

/// Check if a shape is shapeless: gradual `SymIntTuple`.
fn is_shapeless(shape: &SymIntTuple) -> bool {
    matches!(shape.view(), SymIntTupleView::Gradual)
}

/// Compute the broadcasted shape of two tensor shapes following NumPy/PyTorch broadcasting rules:
/// - Dimensions are aligned from right to left
/// - Each dimension must either match or one of them must be 1
/// - Missing dimensions are treated as 1
///
/// For shapes with variadic middles (Unpacked), the algorithm:
/// 1. Consume concrete suffix dims from both sides, right-to-left, broadcasting each pair.
///    Stop when either side runs out of concrete dims (hits a middle or exhausts its dims).
/// 2. Analyze what remains after suffix consumption:
///    - empty + anything → result is the other side
///    - concrete + unpacked(p, m, []) → shapeless if m is gradual; error if m is TypeVarTuple
///    - unpacked + unpacked → if same TypeVarTuple with no extra suffix, broadcast prefixes;
///      if either is gradual, shapeless; otherwise error
/// 3. Assemble result from step 2 output + broadcast suffix.
pub fn broadcast_shapes(a: &SymIntTuple, b: &SymIntTuple) -> Result<SymIntTuple, ShapeError> {
    match (a.view(), b.view()) {
        (SymIntTupleView::Concrete(a_dims), SymIntTupleView::Concrete(b_dims)) => {
            broadcast_concrete(a_dims, b_dims)
        }
        (SymIntTupleView::Concrete(concrete), SymIntTupleView::Gradual)
        | (SymIntTupleView::Gradual, SymIntTupleView::Concrete(concrete)) => {
            broadcast_concrete_with_unpacked(concrete, &[], &gradual_shape_middle(), &[])
        }
        (
            SymIntTupleView::Concrete(concrete),
            SymIntTupleView::Unpacked {
                prefix,
                middle,
                suffix,
            },
        )
        | (
            SymIntTupleView::Unpacked {
                prefix,
                middle,
                suffix,
            },
            SymIntTupleView::Concrete(concrete),
        ) => broadcast_concrete_with_unpacked(concrete, prefix, middle, suffix),
        (SymIntTupleView::Gradual, SymIntTupleView::Gradual) => Ok(SymIntTuple::shapeless()),
        (
            SymIntTupleView::Unpacked {
                prefix,
                middle,
                suffix,
            },
            SymIntTupleView::Gradual,
        ) => broadcast_unpacked_with_unpacked(
            prefix,
            middle,
            suffix,
            &[],
            &gradual_shape_middle(),
            &[],
        ),
        (
            SymIntTupleView::Gradual,
            SymIntTupleView::Unpacked {
                prefix,
                middle,
                suffix,
            },
        ) => broadcast_unpacked_with_unpacked(
            &[],
            &gradual_shape_middle(),
            &[],
            prefix,
            middle,
            suffix,
        ),
        (
            SymIntTupleView::Unpacked {
                prefix: ap,
                middle: am,
                suffix: a_suf,
            },
            SymIntTupleView::Unpacked {
                prefix: bp,
                middle: bm,
                suffix: b_suf,
            },
        ) => broadcast_unpacked_with_unpacked(ap, am, a_suf, bp, bm, b_suf),
    }
}

/// Broadcast a Concrete shape with an Unpacked shape.
///
/// Right-aligns concrete dims against the Unpacked's suffix, broadcasting pairwise.
/// After suffix consumption:
/// - If no concrete dims remain: preserve the Unpacked's prefix + middle.
/// - If concrete dims remain and middle is gradual: result middle is gradual `SymIntTuple`.
/// - If concrete dims remain and middle is TypeVarTuple: error.
fn broadcast_concrete_with_unpacked(
    concrete: &[SymInt],
    prefix: &[SymInt],
    middle: &Type,
    suffix: &[SymInt],
) -> Result<SymIntTuple, ShapeError> {
    let matched = concrete.len().min(suffix.len());

    // Build result suffix: unmatched suffix dims on the left pass through,
    // then broadcast the matched pairs (right-aligned).
    let mut result_suffix = suffix[..suffix.len() - matched].to_vec();
    for i in 0..matched {
        let c_idx = concrete.len() - matched + i;
        let s_idx = suffix.len() - matched + i;
        result_suffix.push(broadcast_dim(&concrete[c_idx], &suffix[s_idx], s_idx)?);
    }

    // Remaining concrete dims not consumed by suffix matching
    let remaining = &concrete[..concrete.len() - matched];

    if remaining.is_empty() {
        // All concrete dims consumed → preserve prefix + middle
        Ok(SymIntTuple::unpacked_from_parts(
            prefix.to_vec(),
            middle.clone(),
            result_suffix,
        ))
    } else if is_gradual_shape_middle(middle) {
        // Can't align remaining concrete with gradual shapeless middle.
        Ok(SymIntTuple::unpacked_from_parts(
            vec![],
            gradual_shape_middle(),
            result_suffix,
        ))
    } else {
        Err(ShapeError::ShapeComputation {
            message: "Cannot broadcast concrete dims with variadic shape: alignment is ambiguous"
                .to_owned(),
        })
    }
}

/// Broadcast two Unpacked shapes.
///
/// Right-aligns suffixes, broadcasting matched pairs. Then analyzes the middles:
/// - Same TypeVarTuple with no extra suffix dims: cancel middles, broadcast prefixes.
/// - Either middle is gradual: result is shapeless + broadcast suffix.
/// - Otherwise: error.
fn broadcast_unpacked_with_unpacked(
    ap: &[SymInt],
    am: &Type,
    a_suf: &[SymInt],
    bp: &[SymInt],
    bm: &Type,
    b_suf: &[SymInt],
) -> Result<SymIntTuple, ShapeError> {
    let matched = a_suf.len().min(b_suf.len());

    // Broadcast matched suffix pairs (right-aligned)
    let mut result_suffix = Vec::new();
    for i in 0..matched {
        let a_idx = a_suf.len() - matched + i;
        let b_idx = b_suf.len() - matched + i;
        result_suffix.push(broadcast_dim(&a_suf[a_idx], &b_suf[b_idx], 0)?);
    }

    // Unmatched suffix dims (at most one side has them)
    let a_extra = &a_suf[..a_suf.len() - matched];
    let b_extra = &b_suf[..b_suf.len() - matched];
    let has_extra = !a_extra.is_empty() || !b_extra.is_empty();

    let am_canon = canonicalize(am.clone());
    let bm_canon = canonicalize(bm.clone());

    if !has_extra && am_canon == bm_canon && !is_gradual_shape_middle(am) {
        // Same TypeVarTuple, no extra suffix → cancel middles, broadcast prefixes
        let prefix = broadcast_concrete(ap, bp)?
            .as_concrete()
            .expect("broadcast_concrete returns a concrete shape")
            .to_vec();
        Ok(SymIntTuple::unpacked_from_parts(
            prefix,
            am.clone(),
            result_suffix,
        ))
    } else if is_gradual_shape_middle(am) || is_gradual_shape_middle(bm) {
        // At least one gradual shapeless middle → can't determine alignment.
        Ok(SymIntTuple::unpacked_from_parts(
            vec![],
            gradual_shape_middle(),
            result_suffix,
        ))
    } else {
        // Different TypeVarTuples or structural mismatch — degrade to shapeless
        // batch dims rather than producing a hard error. At runtime the middles
        // are often identical (e.g. two Linear.forward calls on the same batch)
        // but the checker can't prove it.
        Ok(SymIntTuple::unpacked_from_parts(
            vec![],
            gradual_shape_middle(),
            result_suffix,
        ))
    }
}

/// Broadcast two concrete dimension lists following NumPy/PyTorch rules.
/// Returns a Concrete SymIntTuple.
fn broadcast_concrete(a_dims: &[SymInt], b_dims: &[SymInt]) -> Result<SymIntTuple, ShapeError> {
    let max_rank = a_dims.len().max(b_dims.len());
    let mut result_dims = Vec::with_capacity(max_rank);

    // Iterate from right to left
    for i in 0..max_rank {
        let a_idx = a_dims.len().wrapping_sub(i + 1);
        let b_idx = b_dims.len().wrapping_sub(i + 1);

        let a_dim = if a_idx < a_dims.len() {
            Some(&a_dims[a_idx])
        } else {
            None // Treat as implicit 1
        };

        let b_dim = if b_idx < b_dims.len() {
            Some(&b_dims[b_idx])
        } else {
            None // Treat as implicit 1
        };

        let result_dim = match (a_dim, b_dim) {
            (Some(a_ty), Some(b_ty)) => broadcast_dim(a_ty, b_ty, max_rank - i - 1)?,
            // One shape ran out of dimensions, use the other
            (Some(dim), None) | (None, Some(dim)) => dim.clone(),
            (None, None) => unreachable!(),
        };

        result_dims.push(result_dim);
    }

    // Reverse to get left-to-right order
    result_dims.reverse();
    Ok(SymIntTuple::from_symints(result_dims))
}

/// Broadcast a single pair of dimensions.
/// Canonicalizes both sides so symbolic expressions that reduce to literals are caught.
fn broadcast_dim(a_ty: &SymInt, b_ty: &SymInt, position: usize) -> Result<SymInt, ShapeError> {
    let a_ty = canonicalize_symint_dim(a_ty.clone());
    let b_ty = canonicalize_symint_dim(b_ty.clone());
    match (&a_ty, &b_ty) {
        // Gradual SymInt is compatible with anything; prefer the more precise side.
        (SymInt::Int, _) => Ok(b_ty.clone()),
        (_, SymInt::Int) => Ok(a_ty.clone()),
        // Equal dimensions (after canonicalization): compatible
        _ if a_ty == b_ty => Ok(a_ty.clone()),
        // SymInt(1) broadcasts to anything
        (SymInt::Literal(1), _) => Ok(b_ty.clone()),
        (_, SymInt::Literal(1)) => Ok(a_ty.clone()),
        // Different non-broadcastable types: incompatible
        _ => Err(ShapeError::ShapeComputation {
            message: format!(
                "Cannot broadcast dimension {} with dimension {} at position {}",
                dim_to_type(&a_ty),
                dim_to_type(&b_ty),
                position
            ),
        }),
    }
}

// ============================================================================
// Shaped Array Indexing / Slicing
// ============================================================================

/// A single index operation, pre-classified by the type checker.
/// The type checker resolves Expr nodes into these before calling shape functions.
pub enum IndexOp {
    /// Integer index: removes the dimension
    Int,
    /// Slice: replaces dimension with (stop - start) / step.
    /// `start` defaults to 0, `stop` defaults to the dimension size.
    /// `step` defaults to 1 (no stride).
    Slice {
        start: Option<Type>,
        stop: Option<Type>,
        /// Step/stride for the slice. `None` means step=1 (default).
        /// Can be a literal `SymInt(Literal(n))`, a symbolic `SymInt(Var(...))`,
        /// a `SymInt[S]`, or a `Quantified` type variable.
        step: Option<Type>,
    },
    /// Tensor index: replaces dimension with the index tensor's dims
    ShapedArrayIndex(Vec<Type>),
    /// Tuple/list fancy index: dimension becomes known size or unknown.
    /// `Some(n)` for concrete tuple of length n, `None` for list/unknown.
    Fancy(Option<i64>),
    /// None/np.newaxis index: inserts a new dimension of size 1.
    /// Does not consume a shape dimension.
    NewAxis,
}

/// Apply a single integer index — removes first dimension.
/// E.g. `Tensor[10, 20][i]` -> `Tensor[20]`
pub fn index_shape_int(shape: &SymIntTuple) -> Result<SymIntTuple, ShapeError> {
    match shape.view() {
        SymIntTupleView::Concrete(dims) => {
            if dims.is_empty() {
                return Err(ShapeError::ScalarIndex);
            }
            Ok(SymIntTuple::from_symints(dims[1..].to_vec()))
        }
        SymIntTupleView::Unpacked {
            prefix,
            middle,
            suffix,
        } if !prefix.is_empty() => Ok(SymIntTuple::unpacked_from_parts(
            prefix[1..].to_vec(),
            middle.clone(),
            suffix.to_vec(),
        )),
        // First dim is in variadic middle; can't determine result
        SymIntTupleView::Gradual | SymIntTupleView::Unpacked { .. } => Ok(shapeless_shape()),
    }
}

/// Apply a single slice to first dimension.
/// E.g. `Tensor[10, 20][2:5]` -> `Tensor[3, 20]`
/// With step: `Tensor[100][::2]` -> `Tensor[50]` (ceil_div(100, 2))
pub fn index_shape_slice(
    shape: &SymIntTuple,
    start: Option<Type>,
    stop: Option<Type>,
    step: Option<Type>,
) -> Result<SymIntTuple, ShapeError> {
    match shape.view() {
        SymIntTupleView::Concrete(dims) => {
            if dims.is_empty() {
                return Err(ShapeError::ScalarIndex);
            }
            let start = adjust_negative(
                start.unwrap_or_else(|| Type::SymInt(SymInt::Literal(0))),
                &dim_to_type(&dims[0]),
            );
            let stop = adjust_negative(
                stop.unwrap_or_else(|| dim_to_type(&dims[0])),
                &dim_to_type(&dims[0]),
            );
            let range_dim = sub_dim(stop, start);
            let new_first_dim = apply_step(range_dim, step);
            let mut new_dims = vec![new_first_dim];
            new_dims.extend(dims[1..].iter().map(dim_to_type));
            Ok(SymIntTuple::from_types(new_dims))
        }
        SymIntTupleView::Unpacked {
            prefix,
            middle,
            suffix,
        } if !prefix.is_empty() => {
            let start = adjust_negative(
                start.unwrap_or_else(|| Type::SymInt(SymInt::Literal(0))),
                &dim_to_type(&prefix[0]),
            );
            let stop = adjust_negative(
                stop.unwrap_or_else(|| dim_to_type(&prefix[0])),
                &dim_to_type(&prefix[0]),
            );
            let range_dim = sub_dim(stop, start);
            let new_first_dim = apply_step(range_dim, step);
            let mut new_prefix = vec![new_first_dim];
            new_prefix.extend(prefix[1..].iter().map(dim_to_type));
            Ok(SymIntTuple::unpacked(
                new_prefix,
                middle.clone(),
                dims_to_types(suffix),
            ))
        }
        // Empty prefix: dim0 is hidden in the variadic middle
        SymIntTupleView::Gradual | SymIntTupleView::Unpacked { .. } => Ok(shapeless_shape()),
    }
}

/// Apply tensor-as-index — replaces first dim with index tensor's dims.
/// E.g. `Tensor[B, D1, D2][Tensor[T]]` -> `Tensor[T, D1, D2]`
pub fn index_shape_tensor(
    shape: &SymIntTuple,
    idx_dims: &[Type],
) -> Result<SymIntTuple, ShapeError> {
    match shape.view() {
        SymIntTupleView::Concrete(dims) => {
            if dims.is_empty() {
                return Err(ShapeError::ScalarIndex);
            }
            let mut new_dims = idx_dims.to_vec();
            new_dims.extend(dims[1..].iter().map(dim_to_type));
            Ok(SymIntTuple::from_types(new_dims))
        }
        SymIntTupleView::Unpacked {
            prefix,
            middle,
            suffix,
        } if !prefix.is_empty() => {
            let mut new_prefix = idx_dims.to_vec();
            new_prefix.extend(prefix[1..].iter().map(dim_to_type));
            Ok(SymIntTuple::unpacked(
                new_prefix,
                middle.clone(),
                dims_to_types(suffix),
            ))
        }
        // First dim is in variadic middle; can't determine result
        SymIntTupleView::Gradual | SymIntTupleView::Unpacked { .. } => Ok(shapeless_shape()),
    }
}

/// Count how many shape dimensions a sequence of ops consumes.
/// `NewAxis` ops don't consume a dimension; all others consume one.
fn ops_dims_consumed(ops: &[IndexOp]) -> usize {
    ops.iter()
        .filter(|op| !matches!(op, IndexOp::NewAxis))
        .count()
}

/// Apply multi-axis indexing with optional ellipsis.
/// `pre_ops` are applied left-to-right from dim 0.
/// `post_ops` are applied from the end (only when `has_ellipsis` is true).
/// Dims between pre and post (the ellipsis range) are preserved.
pub fn index_shape_multi(
    shape: &SymIntTuple,
    pre_ops: &[IndexOp],
    post_ops: &[IndexOp],
    has_ellipsis: bool,
) -> Result<SymIntTuple, ShapeError> {
    match shape.view() {
        SymIntTupleView::Concrete(shape_dims) => {
            let pre_consumed = ops_dims_consumed(pre_ops);
            let post_consumed = ops_dims_consumed(post_ops);
            let total_consumed = pre_consumed + post_consumed;
            if total_consumed > shape_dims.len() {
                return Err(ShapeError::TooManyIndices {
                    got: total_consumed,
                    max: shape_dims.len(),
                });
            }

            let pre_dims = dims_to_types(&shape_dims[..pre_consumed]);
            let (pre_result, _) = apply_ops_to_dims(pre_ops, &pre_dims)?;

            let post_start = if has_ellipsis {
                shape_dims.len() - post_consumed
            } else {
                pre_consumed
            };
            let post_dims = dims_to_types(&shape_dims[post_start..]);
            let (post_result, _) = apply_ops_to_dims(post_ops, &post_dims)?;

            let mut new_dims = pre_result;
            if has_ellipsis {
                // Preserve ellipsis-covered dims
                new_dims.extend(shape_dims[pre_consumed..post_start].iter().map(dim_to_type));
            } else {
                // No ellipsis: append remaining unindexed dims
                new_dims.extend(shape_dims[pre_consumed..].iter().map(dim_to_type));
            }
            new_dims.extend(post_result);

            Ok(SymIntTuple::from_types(new_dims))
        }
        SymIntTupleView::Gradual => {
            let pre_consumed = ops_dims_consumed(pre_ops);
            let post_consumed = ops_dims_consumed(post_ops);
            if pre_consumed > 0 || post_consumed > 0 {
                return Ok(shapeless_shape());
            }

            let (pre_result, _) = apply_ops_to_dims(pre_ops, &[])?;
            let (post_result, _) = apply_ops_to_dims(post_ops, &[])?;
            let middle = SymIntTuple::shapeless().to_shape_arg_type();
            Ok(SymIntTuple::unpacked(pre_result, middle, post_result))
        }
        SymIntTupleView::Unpacked {
            prefix,
            middle,
            suffix,
        } => {
            let pre_consumed = ops_dims_consumed(pre_ops);
            let post_consumed = ops_dims_consumed(post_ops);
            if pre_consumed > prefix.len() || post_consumed > suffix.len() {
                return Ok(shapeless_shape());
            }

            let pre_dims = dims_to_types(&prefix[..pre_consumed]);
            let (pre_result, _) = apply_ops_to_dims(pre_ops, &pre_dims)?;

            let post_suffix_start = suffix.len() - post_consumed;
            let post_dims = dims_to_types(&suffix[post_suffix_start..]);
            let (post_result, _) = apply_ops_to_dims(post_ops, &post_dims)?;

            let remaining_prefix = &prefix[pre_consumed..];
            let remaining_suffix = &suffix[..post_suffix_start];

            let mut result_prefix = pre_result;
            result_prefix.extend(remaining_prefix.iter().map(dim_to_type));
            let mut result_suffix = dims_to_types(remaining_suffix);
            result_suffix.extend(post_result);

            Ok(SymIntTuple::unpacked(
                result_prefix,
                middle.clone(),
                result_suffix,
            ))
        }
    }
}

/// Create a shapeless shape (compatible with any shape).
fn shapeless_shape() -> SymIntTuple {
    SymIntTuple::shapeless()
}

/// Adjust a negative slice bound by adding dim size (Python negative index semantics).
/// E.g. -1 on dim N becomes N + (-1) = N - 1.
/// Also handles symbolic negation: -1 * X (from unary `-` on a Dim/SymInt expression)
/// becomes dim_size + (-1 * X) = dim_size - X.
fn adjust_negative(bound: Type, dim_size: &Type) -> Type {
    let is_negative = match &bound {
        // Literal negative: -1, -2, etc.
        Type::SymInt(SymInt::Literal(v)) => *v < 0,
        // Symbolic negation: (-1 * X), (-2 * X), etc. from unary negation
        Type::SymInt(SymInt::Mul(left, _)) if let SymInt::Literal(v) = left.as_ref() => *v < 0,
        _ => false,
    };
    if is_negative {
        Type::SymInt(SymInt::add(dim_size.clone(), bound))
    } else {
        bound
    }
}

/// Compute stop - start, simplifying x - 0 to x.
fn sub_dim(stop: Type, start: Type) -> Type {
    match &start {
        Type::SymInt(SymInt::Literal(0)) => stop,
        _ => Type::SymInt(SymInt::sub(stop, start)),
    }
}

/// Apply step (stride) to a range dimension: ceil_div(range, step).
/// step=None or step=Literal(1) is identity. For literal range and step,
/// computes the exact integer ceiling division. For symbolic steps (SymInt,
/// Quantified), builds a symbolic ceil_div expression.
fn apply_step(range_dim: Type, step: Option<Type>) -> Type {
    let step = match step {
        None => return range_dim,
        Some(s) => s,
    };
    match &step {
        // Literal step: exact arithmetic
        Type::SymInt(SymInt::Literal(1)) => range_dim,
        Type::SymInt(SymInt::Literal(s)) if *s > 1 => {
            let s = *s;
            if let Type::SymInt(SymInt::Literal(n)) = &range_dim {
                Type::SymInt(SymInt::Literal((*n + s - 1) / s))
            } else {
                // Symbolic range, literal step: ceil_div(range, step)
                let step_minus_1 = Type::SymInt(SymInt::Literal(s - 1));
                let numerator = Type::SymInt(SymInt::add(range_dim, step_minus_1));
                Type::SymInt(SymInt::floor_div(
                    numerator,
                    Type::SymInt(SymInt::Literal(s)),
                ))
            }
        }
        Type::SymInt(SymInt::Literal(s)) if *s <= 0 => {
            // Negative or zero step: degenerate, return unknown
            Type::any_implicit()
        }
        // Symbolic step (SymInt var, Quantified): build ceil_div(range, step) symbolically
        _ => {
            // ceil_div(range, step) = (range + step - 1) // step
            let step_minus_1 =
                Type::SymInt(SymInt::sub(step.clone(), Type::SymInt(SymInt::Literal(1))));
            let numerator = Type::SymInt(SymInt::add(range_dim, step_minus_1));
            Type::SymInt(SymInt::floor_div(numerator, step))
        }
    }
}

/// Apply a single `IndexOp` to a known dimension.
/// Returns `Some(new_dim)` for ops that keep the dim, `None` for `Int` (dim removed).
/// Must not be called with `NewAxis` — that is handled by `apply_ops_to_dims`.
fn apply_index_op(op: &IndexOp, dim: &Type) -> Option<Type> {
    match op {
        IndexOp::Int => None,
        IndexOp::Slice { start, stop, step } => {
            let start = adjust_negative(
                start
                    .clone()
                    .unwrap_or_else(|| Type::SymInt(SymInt::Literal(0))),
                dim,
            );
            let stop = adjust_negative(stop.clone().unwrap_or_else(|| dim.clone()), dim);
            let range_dim = sub_dim(stop, start);
            Some(apply_step(range_dim, step.clone()))
        }
        IndexOp::ShapedArrayIndex(idx_dims) => {
            // Multi-axis tensor indexing: this case shouldn't appear in apply_index_op
            // since tensor indexing replaces dims entirely. Treat as fancy.
            if idx_dims.is_empty() {
                None
            } else {
                // Return the first dim of the index tensor; the rest are handled
                // at a higher level. For multi-axis, this degrades to unknown.
                Some(Type::any_implicit())
            }
        }
        IndexOp::Fancy(Some(n)) => Some(Type::SymInt(SymInt::Literal(*n))),
        IndexOp::Fancy(None) => Some(Type::any_implicit()),
        IndexOp::NewAxis => unreachable!("NewAxis handled by apply_ops_to_dims"),
    }
}

/// Apply a sequence of `IndexOp`s to a slice of dimensions.
/// `NewAxis` ops insert a dim of size 1 without consuming a shape dimension.
/// `ShapedArrayIndex` ops broadcast together: the first emits the index dims,
/// subsequent ones with the same shape consume a dim without emitting.
/// Other ops consume one shape dimension each.
/// Returns (result_dims, number_of_shape_dims_consumed).
fn apply_ops_to_dims(ops: &[IndexOp], dims: &[Type]) -> Result<(Vec<Type>, usize), ShapeError> {
    let mut new_dims = Vec::new();
    let mut dim_idx = 0;
    let mut tensor_index_emitted = false;
    for op in ops {
        match op {
            IndexOp::NewAxis => {
                new_dims.push(Type::SymInt(SymInt::Literal(1)));
            }
            IndexOp::ShapedArrayIndex(idx_dims) => {
                if dim_idx >= dims.len() {
                    return Err(ShapeError::TooManyIndices {
                        got: dim_idx + 1,
                        max: dims.len(),
                    });
                }
                // First tensor index in a group emits the broadcast shape.
                // Subsequent tensor indices consume a dim without emitting
                // (they participate in the same broadcast group).
                if !tensor_index_emitted {
                    new_dims.extend(idx_dims.iter().cloned());
                    tensor_index_emitted = true;
                }
                dim_idx += 1;
            }
            _ => {
                if dim_idx >= dims.len() {
                    return Err(ShapeError::TooManyIndices {
                        got: dim_idx + 1,
                        max: dims.len(),
                    });
                }
                if let Some(new_dim) = apply_index_op(op, &dims[dim_idx]) {
                    new_dims.push(new_dim);
                }
                dim_idx += 1;
            }
        }
    }
    Ok((new_dims, dim_idx))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use pyrefly_python::module::Module;
    use pyrefly_python::module_name::ModuleName;
    use pyrefly_python::module_path::ModulePath;
    use pyrefly_python::nesting_context::NestingContext;
    use ruff_python_ast::Identifier;
    use ruff_python_ast::name::Name;
    use ruff_text_size::TextRange;
    use ruff_text_size::TextSize;

    use crate::class::Class;
    use crate::class::ClassDefIndex;
    use crate::class::ClassType;
    use crate::dimension::SymInt;
    use crate::dimension::gradual_size;
    use crate::lit_int::LitInt;
    use crate::literal::Lit;
    use crate::literal::LitStyle;
    use crate::literal::Literal;
    use crate::quantified::AnchorIndex;
    use crate::quantified::Quantified;
    use crate::quantified::QuantifiedIdentity;
    use crate::quantified::QuantifiedKind;
    use crate::quantified::QuantifiedOrigin;
    use crate::shaped_array::ShapedArrayType;
    use crate::shaped_array::SymIntTuple;
    use crate::shaped_array::SymIntTupleRepr;
    use crate::shaped_array::SymIntTupleView;
    use crate::shaped_array::broadcast_shapes;
    use crate::shaped_array::gradual_shape_middle;
    use crate::shaped_array::is_tuple_carrier_shape_middle;
    use crate::shaped_array::shape_to_tuple_carrier;
    use crate::shaped_array::shape_to_tuple_carrier_arg;
    use crate::shaped_array::tuple_carrier_to_shape;
    use crate::tuple::Tuple;
    use crate::type_var::PreInferenceVariance;
    use crate::type_var::Restriction;
    use crate::type_var::TypeVar;
    use crate::type_var_tuple::TypeVarTuple;
    use crate::types::AnyStyle;
    use crate::types::TArgs;
    use crate::types::TParams;
    use crate::types::Type;
    use crate::types::Var;

    /// Internal literal dimension `n` (`Type::SymInt(SymInt::Literal(n))`).
    fn size(n: i64) -> Type {
        Type::SymInt(SymInt::Literal(n))
    }

    fn dim(n: i64) -> SymInt {
        SymInt::Literal(n)
    }

    /// User-facing `Literal[n]` carrier element.
    fn literal(n: i64) -> Type {
        LitInt::new(n).to_explicit_type()
    }

    fn concrete_carrier(elts: Vec<Type>) -> Type {
        Type::Tuple(Tuple::Concrete(elts))
    }

    fn fake_module(module: &str) -> Module {
        Module::new(
            ModuleName::from_str(module),
            ModulePath::filesystem(PathBuf::from(module)),
            Arc::new("fake module contents".to_owned()),
        )
    }

    fn fake_class_type(module: &str, name: &str) -> ClassType {
        let module = fake_module(module);
        ClassType::new(
            Class::new(
                ClassDefIndex(0),
                Identifier::new(Name::new(name), TextRange::empty(TextSize::new(0))),
                NestingContext::toplevel(),
                module,
                None,
            ),
            TArgs::default(),
        )
    }

    fn fake_type_var(name: &str, kind: QuantifiedKind) -> TypeVar {
        TypeVar::new_with_kind(
            Identifier::new(Name::new(name), TextRange::empty(TextSize::new(0))),
            fake_module("__test__"),
            kind,
            Restriction::Unrestricted,
            None,
            PreInferenceVariance::Invariant,
        )
    }

    fn fake_type_var_tuple(name: &str) -> TypeVarTuple {
        TypeVarTuple::new(
            Identifier::new(Name::new(name), TextRange::empty(TextSize::new(0))),
            fake_module("__test__"),
            None,
        )
    }

    fn fake_tparam(name: &str, kind: QuantifiedKind) -> Quantified {
        Quantified::new(
            QuantifiedIdentity::new(
                ModuleName::from_str("__test__"),
                AnchorIndex::first(TextRange::default()),
                QuantifiedOrigin::Pep695,
            ),
            Name::new(name),
            kind,
            None,
            Restriction::Unrestricted,
            PreInferenceVariance::Invariant,
        )
    }

    fn registered_array_shape_arg(shape_arg: Type) -> ShapedArrayType {
        let shape_param = fake_tparam("Shape", QuantifiedKind::TypeVar);
        let class = fake_class_type("arrays", "Array").class_object().clone();
        ShapedArrayType::new(
            ClassType::new(
                class,
                TArgs::new(Arc::new(TParams::new(vec![shape_param])), vec![shape_arg]),
            ),
            SymIntTuple::shapeless(),
        )
        .with_tuple_carrier_shape_arg(0)
    }

    fn registered_array_shape_arg_at(
        shape_arg_index: usize,
        shape_args: Vec<Type>,
    ) -> ShapedArrayType {
        let tparams = (0..shape_args.len())
            .map(|i| fake_tparam(&format!("Shape{i}"), QuantifiedKind::TypeVar))
            .collect();
        let class = fake_class_type("arrays", "Array").class_object().clone();
        ShapedArrayType::new(
            ClassType::new(
                class,
                TArgs::new(Arc::new(TParams::new(tparams)), shape_args),
            ),
            SymIntTuple::shapeless(),
        )
        .with_tuple_carrier_shape_arg(shape_arg_index)
    }

    #[test]
    fn concrete_shape_to_tuple_carrier() {
        let shape = SymIntTuple::from_types(vec![size(3), size(4), size(5)]);
        assert_eq!(
            shape_to_tuple_carrier(&shape),
            concrete_carrier(vec![literal(3), literal(4), literal(5)])
        );
    }

    #[test]
    fn shapeless_projects_to_gradual_symint_tuple() {
        let shape = SymIntTuple::shapeless();
        assert!(shape.is_shapeless());
        assert_eq!(
            shape.to_tuple_type(),
            Type::Tuple(Tuple::Unbounded(Box::new(gradual_size())))
        );
        assert_eq!(shape.to_tuple(), Tuple::Unbounded(Box::new(gradual_size())));
    }

    #[test]
    fn symint_tuple_view_borrows_shape_structure() {
        let concrete = SymIntTuple::from_types(vec![size(2), size(3)]);
        match concrete.view() {
            SymIntTupleView::Concrete(dims) => assert_eq!(dims, &[dim(2), dim(3)]),
            _ => panic!("expected concrete shape view"),
        }

        let shapeless = SymIntTuple::shapeless();
        assert!(shapeless.is_shapeless());
        assert!(matches!(shapeless.view(), SymIntTupleView::Gradual));

        let middle = Type::Var(Var::ZERO);
        let unpacked = SymIntTuple::unpacked(vec![size(1)], middle.clone(), vec![size(4)]);
        match unpacked.view() {
            SymIntTupleView::Unpacked {
                prefix,
                middle: view_middle,
                suffix,
            } => {
                assert_eq!(prefix, &[dim(1)]);
                assert_eq!(view_middle, &middle);
                assert_eq!(suffix, &[dim(4)]);
            }
            _ => panic!("expected unpacked shape view"),
        }
    }

    #[test]
    fn from_tuple_wraps_valid_unbounded_shape_middle() {
        let elt = literal(5);
        let shape = SymIntTuple::from_tuple(Tuple::Unbounded(Box::new(elt.clone())));

        assert_eq!(
            shape,
            SymIntTuple::unpacked(
                Vec::new(),
                Type::Tuple(Tuple::Unbounded(Box::new(size(5)))),
                Vec::new(),
            )
        );
        assert_eq!(
            shape.to_tuple_type(),
            Type::Tuple(Tuple::Unpacked(Box::new((
                Vec::new(),
                Type::Tuple(Tuple::Unbounded(Box::new(size(5)))),
                Vec::new(),
            ))))
        );
    }

    #[test]
    fn unpacked_nested_shapeless_syminttuple_middle_flattens() {
        assert_eq!(
            SymIntTuple::unpacked(
                Vec::new(),
                SymIntTuple::shapeless().to_shape_arg_type(),
                Vec::new(),
            ),
            SymIntTuple::shapeless()
        );
    }

    #[test]
    fn unpacked_invalid_unbounded_middle_recovers_to_gradual() {
        let elt = Type::ClassType(fake_class_type("torch", "Materialization"));
        let shape = SymIntTuple::from_tuple(Tuple::Unbounded(Box::new(elt)));

        assert_eq!(shape, SymIntTuple::shapeless());
    }

    #[test]
    fn from_tuple_valid_unbounded_middle_projects_as_tuple_carrier() {
        let shape = SymIntTuple::from_tuple(Tuple::Unbounded(Box::new(size(5))));

        assert_eq!(
            shape,
            SymIntTuple::unpacked(
                Vec::new(),
                Type::Tuple(Tuple::Unbounded(Box::new(size(5)))),
                Vec::new(),
            )
        );
    }

    #[test]
    fn broadcast_accepts_raw_gradual_tuple_middle() {
        let unnormalized = SymIntTuple(SymIntTupleRepr::Unpacked {
            prefix: vec![dim(2)],
            middle: Box::new(Type::Tuple(Tuple::Unbounded(Box::new(gradual_size())))),
            suffix: vec![dim(3)],
        });
        let concrete = SymIntTuple::from_types(vec![size(4), size(3)]);

        assert_eq!(
            broadcast_shapes(&concrete, &unnormalized).unwrap(),
            SymIntTuple::unpacked(Vec::new(), gradual_shape_middle(), vec![size(3)],)
        );
    }

    #[test]
    fn concrete_projects_to_symint_dims_not_literals() {
        let n = Type::Quantified(Box::new(Quantified::new(
            QuantifiedIdentity::new(
                ModuleName::from_str("__test__"),
                AnchorIndex::first(TextRange::default()),
                QuantifiedOrigin::Pep695,
            ),
            Name::new("N"),
            QuantifiedKind::SymIntVar,
            None,
            Restriction::Unrestricted,
            PreInferenceVariance::Invariant,
        )));
        let projected = SymIntTuple::from_types(vec![size(2), n.clone()]).to_tuple_type();

        assert_eq!(
            projected,
            Type::Tuple(Tuple::Concrete(vec![
                size(2),
                Type::SymInt(SymInt::Symbolic(Box::new(n))),
            ]))
        );
        assert_eq!(projected.to_string(), "tuple[SymInt[2], SymInt[N]]");
    }

    #[test]
    fn unpacked_projection_preserves_shape() {
        let projected =
            SymIntTuple::unpacked(vec![size(2)], Type::any_tuple(), vec![size(3)]).to_tuple_type();

        assert_eq!(
            projected,
            Type::Tuple(Tuple::Unpacked(Box::new((
                vec![size(2)],
                Type::Tuple(Tuple::Unbounded(Box::new(gradual_size()))),
                vec![size(3)],
            ))))
        );
    }

    #[test]
    fn unpacked_typevartuple_middle_projects_as_variadic() {
        let s = Type::Quantified(Box::new(Quantified::new(
            QuantifiedIdentity::new(
                ModuleName::from_str("__test__"),
                AnchorIndex::first(TextRange::default()),
                QuantifiedOrigin::Pep695,
            ),
            Name::new("S"),
            QuantifiedKind::TypeVarTuple,
            None,
            Restriction::Unrestricted,
            PreInferenceVariance::Invariant,
        )));
        let projected =
            SymIntTuple::unpacked(vec![size(2)], s.clone(), vec![size(3)]).to_tuple_type();

        assert_eq!(
            projected,
            Type::Tuple(Tuple::Unpacked(Box::new(
                (vec![size(2)], s, vec![size(3)],)
            )))
        );
    }

    #[test]
    fn affixed_tuple_carrier_middle_projects_to_gradual_tuple_boundary() {
        let middle = Type::Var(Var::ZERO);
        let shape = SymIntTuple::unpacked(vec![size(1)], middle.clone(), vec![size(2)]);

        assert_eq!(
            shape.to_tuple_type(),
            Type::Tuple(Tuple::Unpacked(Box::new((
                vec![size(1)],
                Type::Tuple(Tuple::Unbounded(Box::new(gradual_size()))),
                vec![size(2)],
            ))))
        );
        assert_eq!(
            SymIntTuple::from_tuple(shape.to_tuple()),
            SymIntTuple::unpacked(vec![size(1)], gradual_shape_middle(), vec![size(2)])
        );
        assert_eq!(
            shape_to_tuple_carrier_arg(&shape),
            shape_to_tuple_carrier(&shape)
        );

        let whole_shape = SymIntTuple::unpacked(Vec::new(), middle.clone(), Vec::new());
        assert_eq!(shape_to_tuple_carrier_arg(&whole_shape), middle);
    }

    #[test]
    fn affixed_tuple_carrier_middle_displays_as_elements_unpack() {
        let middle = Type::Quantified(Box::new(Quantified::new(
            QuantifiedIdentity::new(
                ModuleName::from_str("__test__"),
                AnchorIndex::first(TextRange::default()),
                QuantifiedOrigin::Pep695,
            ),
            Name::new("S"),
            QuantifiedKind::TypeVar,
            None,
            Restriction::Unrestricted,
            PreInferenceVariance::Invariant,
        )));

        assert_eq!(
            SymIntTuple::unpacked(vec![size(1)], middle, vec![size(2)]).to_string(),
            "1, *Elements[S], 2"
        );
        assert_eq!(SymIntTuple::shapeless().to_string(), "*SymIntTuple");
    }

    #[test]
    fn literal_carrier_to_concrete_shape() {
        let carrier = concrete_carrier(vec![literal(3), literal(4), literal(5)]);
        assert_eq!(
            tuple_carrier_to_shape(&carrier),
            Some(SymIntTuple::from_types(vec![size(3), size(4), size(5)]))
        );
    }

    #[test]
    fn concrete_round_trip_both_directions() {
        let shape = SymIntTuple::from_types(vec![size(2), size(3)]);
        let carrier = shape_to_tuple_carrier(&shape);
        assert_eq!(tuple_carrier_to_shape(&carrier), Some(shape.clone()));

        let carrier = concrete_carrier(vec![literal(2), literal(3)]);
        let shape = tuple_carrier_to_shape(&carrier).unwrap();
        assert_eq!(shape_to_tuple_carrier(&shape), carrier);
    }

    #[test]
    fn explicit_any_internal_dimension_becomes_gradual_symint_carrier() {
        let shape = SymIntTuple::from_types(vec![Type::Any(AnyStyle::Explicit)]);
        assert_eq!(
            shape_to_tuple_carrier(&shape),
            concrete_carrier(vec![gradual_size()])
        );
    }

    #[test]
    fn error_any_internal_dimension_becomes_gradual_symint_carrier() {
        let shape = SymIntTuple::from_types(vec![Type::Any(AnyStyle::Error)]);
        assert_eq!(
            shape_to_tuple_carrier(&shape),
            concrete_carrier(vec![gradual_size()])
        );
    }

    #[test]
    fn symbolic_internal_dimension_round_trips_through_size_carrier() {
        let var = Type::Var(Var::ZERO);
        let shape = SymIntTuple::from_types(vec![var.clone()]);
        let carrier = concrete_carrier(vec![Type::SymInt(SymInt::Symbolic(Box::new(var)))]);

        assert_eq!(shape_to_tuple_carrier(&shape), carrier);
        assert_eq!(tuple_carrier_to_shape(&carrier), Some(shape));
    }

    #[test]
    fn raw_internal_symintvar_carrier_elements_pass_through() {
        let quantified = Type::Quantified(Box::new(fake_tparam("N", QuantifiedKind::SymIntVar)));
        let dims = vec![size(8), Type::Any(AnyStyle::Explicit), quantified];

        assert_eq!(
            tuple_carrier_to_shape(&concrete_carrier(dims.clone())),
            Some(SymIntTuple::from_types(dims))
        );
    }

    #[test]
    fn symint_carriers_with_internal_operands_pass_through() {
        let quantified = Type::Quantified(Box::new(fake_tparam("N", QuantifiedKind::SymIntVar)));
        let symint = Type::SymInt(SymInt::add(quantified.clone(), size(1)));

        assert_eq!(
            tuple_carrier_to_shape(&concrete_carrier(vec![symint.clone()])),
            Some(SymIntTuple::from_types(vec![symint.clone()]))
        );
    }

    #[test]
    fn raw_typevar_carrier_projects_to_variadic_shape() {
        let carrier = Type::Quantified(Box::new(Quantified::new(
            QuantifiedIdentity::new(
                ModuleName::from_str("__test__"),
                AnchorIndex::first(TextRange::default()),
                QuantifiedOrigin::Pep695,
            ),
            Name::new("Shape"),
            QuantifiedKind::TypeVar,
            None,
            Restriction::Unrestricted,
            PreInferenceVariance::Invariant,
        )));
        let shape = SymIntTuple::unpacked(Vec::new(), carrier.clone(), Vec::new());

        assert_eq!(tuple_carrier_to_shape(&carrier), Some(shape.clone()));
        assert_eq!(shape_to_tuple_carrier_arg(&shape), carrier);

        let carrier = Type::Var(Var::ZERO);
        let shape = SymIntTuple::unpacked(Vec::new(), carrier.clone(), Vec::new());
        assert_eq!(shape_to_tuple_carrier_arg(&shape), carrier);
    }

    #[test]
    fn shape_arg_type_is_first_class_symint_tuple() {
        let shape = SymIntTuple::from_types(vec![size(2), size(3)]);
        let shape_arg = shape.to_shape_arg_type();
        assert_eq!(shape_arg, Type::SymIntTuple(Box::new(shape.clone())));
        assert_eq!(SymIntTuple::from_shape_arg_type(&shape_arg), Some(shape));
    }

    #[test]
    fn normalize_rebuilds_raw_representations_and_preserves_middle() {
        let raw_concrete = SymIntTuple(SymIntTupleRepr::Concrete(vec![SymInt::add(
            size(1),
            size(2),
        )]));
        let concrete = SymIntTuple::from_types(vec![size(3)]);
        assert_ne!(raw_concrete, concrete);
        assert_eq!(raw_concrete.normalize(), concrete);
        assert_eq!(
            SymIntTuple::from_shape_arg_type(&Type::SymIntTuple(Box::new(raw_concrete))),
            Some(concrete.clone())
        );

        let gradual = SymIntTuple(SymIntTupleRepr::Gradual).normalize();
        assert_eq!(gradual, SymIntTuple::shapeless());

        let middle = Type::Var(Var::ZERO);
        let raw_unpacked = SymIntTuple(SymIntTupleRepr::Unpacked {
            prefix: vec![SymInt::add(size(1), size(2))],
            middle: Box::new(middle.clone()),
            suffix: vec![SymInt::add(size(3), size(4))],
        });
        let unpacked = raw_unpacked.normalize();
        assert_eq!(
            unpacked,
            SymIntTuple::unpacked(vec![size(3)], middle.clone(), vec![size(7)])
        );
        assert!(matches!(
            unpacked.view(),
            SymIntTupleView::Unpacked { middle: stored, .. } if stored == &middle
        ));

        for normalized in [concrete, gradual, unpacked] {
            assert_eq!(normalized.normalize(), normalized);
        }
    }

    #[test]
    fn registered_shape_projects_from_first_class_shape_arg() {
        let projected = SymIntTuple::from_types(vec![size(2), size(3)]);
        let tensor = registered_array_shape_arg(projected.to_shape_arg_type());

        assert_eq!(tensor.shape(), projected);
    }

    #[test]
    fn registered_shape_projects_from_legacy_tuple_carrier() {
        let projected = SymIntTuple::from_types(vec![size(6)]);
        let tensor = registered_array_shape_arg(concrete_carrier(vec![literal(6)]));

        assert_eq!(tensor.shape(), projected);
    }

    #[test]
    fn set_shape_updates_registered_shape_arg() {
        let old_shape = SymIntTuple::from_types(vec![size(2)]);
        let new_shape = SymIntTuple::from_types(vec![size(4), size(5)]);
        let mut tensor = registered_array_shape_arg(old_shape.to_shape_arg_type());

        tensor.set_shape(new_shape.clone());

        assert_eq!(tensor.shape(), new_shape.clone());
        assert_eq!(
            tensor.base_class.targs().as_slice()[0],
            new_shape.to_shape_arg_type()
        );
    }

    #[test]
    fn tuple_carrier_shape_arg_index_participates_in_identity() {
        let shape = SymIntTuple::from_types(vec![size(2)]);
        let shape_args = vec![shape.to_shape_arg_type(), shape.to_shape_arg_type()];
        let first_arg_shape = registered_array_shape_arg_at(0, shape_args.clone());
        let second_arg_shape = registered_array_shape_arg_at(1, shape_args);

        assert_eq!(first_arg_shape.shape(), second_arg_shape.shape());
        assert_ne!(first_arg_shape, second_arg_shape);
    }

    #[test]
    fn registered_shape_display_uses_projected_shape_arg() {
        let projected = SymIntTuple::from_types(vec![size(2), size(3)]);
        let tensor = registered_array_shape_arg(projected.to_shape_arg_type());
        assert_eq!(tensor.to_string(), "Array[[2, 3]]");

        let tensor = registered_array_shape_arg(SymIntTuple::shapeless().to_shape_arg_type());
        assert_eq!(tensor.to_string(), "Array");
    }

    #[test]
    #[should_panic(
        expected = "registered shaped-array shape argument should project to SymIntTuple"
    )]
    fn registered_shape_with_invalid_carrier_panics() {
        let tensor = registered_array_shape_arg(Type::None);

        let _ = tensor.shape();
    }

    #[test]
    fn unpacked_first_class_symint_tuple_middle_flattens() {
        let middle = SymIntTuple::from_types(vec![size(3), size(4)]).to_shape_arg_type();
        assert_eq!(
            SymIntTuple::unpacked(vec![size(2)], middle, vec![size(5)]),
            SymIntTuple::from_types(vec![size(2), size(3), size(4), size(5)])
        );
    }

    #[test]
    fn from_shape_arg_type_normalizes_raw_nested_symint_tuple() {
        let inner = SymIntTuple::from_types(vec![size(2), size(3)]);
        let nested = SymIntTuple(SymIntTupleRepr::Unpacked {
            prefix: Vec::new(),
            middle: Box::new(inner.to_shape_arg_type()),
            suffix: Vec::new(),
        })
        .to_shape_arg_type();
        assert_eq!(SymIntTuple::from_shape_arg_type(&nested), Some(inner));
    }

    #[test]
    fn finite_tuple_unpack_flattens() {
        let middle = Type::Tuple(Tuple::Concrete(vec![
            size(2),
            LitInt::new(3).to_explicit_type(),
        ]));
        assert_eq!(
            SymIntTuple::unpacked(vec![size(1)], middle, vec![size(4)]),
            SymIntTuple::from_types(vec![size(1), size(2), size(3), size(4)])
        );
    }

    #[test]
    fn invalid_concrete_tuple_middle_recovers_with_gradual_dims() {
        let middle = Type::Tuple(Tuple::Concrete(vec![
            literal(2),
            Type::ClassType(fake_class_type("builtins", "str")),
            size(3),
            bool_literal(),
            Type::Quantified(Box::new(fake_tparam("T", QuantifiedKind::TypeVar))),
            Type::Quantified(Box::new(fake_tparam("P", QuantifiedKind::ParamSpec))),
            Type::Quantified(Box::new(fake_tparam("Ts", QuantifiedKind::TypeVarTuple))),
        ]));
        let shape = SymIntTuple::unpacked(vec![size(1)], middle, vec![size(4)]);
        let expected = SymIntTuple::from_types(vec![
            size(1),
            size(2),
            gradual_size(),
            size(3),
            gradual_size(),
            gradual_size(),
            gradual_size(),
            gradual_size(),
            size(4),
        ]);

        assert!(matches!(shape.view(), SymIntTupleView::Concrete(_)));
        assert_eq!(shape, expected);
        assert_eq!(shape.to_tuple_type(), expected.to_tuple_type());
        assert_eq!(SymIntTuple::from_tuple(shape.to_tuple()), shape);
    }

    #[test]
    fn invalid_non_tuple_middle_recovers_as_shapeless() {
        let middle = Type::ClassType(fake_class_type("builtins", "str"));
        let shape = SymIntTuple::unpacked(vec![size(1)], middle, vec![size(2)]);

        assert_eq!(shape, SymIntTuple::shapeless());
        assert_eq!(
            shape.to_tuple_type(),
            Type::Tuple(Tuple::Unbounded(Box::new(gradual_size())))
        );
        assert_eq!(SymIntTuple::from_tuple(shape.to_tuple()), shape);
    }

    #[test]
    fn any_middle_recovers_with_gradual_unbounded_middle() {
        let shapeless =
            SymIntTuple::unpacked(Vec::new(), Type::Any(AnyStyle::Implicit), Vec::new());
        assert_eq!(shapeless, SymIntTuple::shapeless());

        let shape =
            SymIntTuple::unpacked(vec![size(1)], Type::Any(AnyStyle::Implicit), vec![size(2)]);

        assert_eq!(
            shape,
            SymIntTuple::unpacked(
                vec![size(1)],
                SymIntTuple::shapeless().to_shape_arg_type(),
                vec![size(2)],
            )
        );
        assert_eq!(SymIntTuple::from_tuple(shape.to_tuple()), shape);
    }

    #[test]
    fn invalid_unbounded_tuple_middle_element_recovers_to_gradual() {
        let middle = Type::Tuple(Tuple::Unbounded(Box::new(Type::ClassType(
            fake_class_type("builtins", "str"),
        ))));
        let shape = SymIntTuple::unpacked(vec![size(1)], middle, vec![size(2)]);

        assert_eq!(
            shape,
            SymIntTuple::unpacked(
                vec![size(1)],
                SymIntTuple::shapeless().to_shape_arg_type(),
                vec![size(2)],
            )
        );
        assert_eq!(SymIntTuple::from_tuple(shape.to_tuple()), shape);
    }

    #[test]
    fn invalid_quantified_unbounded_middle_elements_recover_to_gradual() {
        let invalid_elements = [
            Type::Quantified(Box::new(fake_tparam("T", QuantifiedKind::TypeVar))),
            Type::Quantified(Box::new(fake_tparam("P", QuantifiedKind::ParamSpec))),
            Type::Quantified(Box::new(fake_tparam("Ts", QuantifiedKind::TypeVarTuple))),
            Type::TypeVarTuple(fake_type_var_tuple("Ts")),
        ];
        for elt in invalid_elements {
            let middle = Type::Tuple(Tuple::Unbounded(Box::new(elt)));
            let shape = SymIntTuple::unpacked(vec![size(1)], middle, vec![size(2)]);

            assert_eq!(
                shape,
                SymIntTuple::unpacked(
                    vec![size(1)],
                    SymIntTuple::shapeless().to_shape_arg_type(),
                    vec![size(2)],
                )
            );
            assert_eq!(SymIntTuple::from_tuple(shape.to_tuple()), shape);
        }
    }

    #[test]
    fn valid_unbounded_middle_elements_are_canonicalized() {
        let quantified = Type::Quantified(Box::new(fake_tparam("N", QuantifiedKind::SymIntVar)));
        let type_var = Type::TypeVar(fake_type_var("N", QuantifiedKind::SymIntVar));
        let int_type = Type::ClassType(fake_class_type("builtins", "int"));
        for (elt, expected) in [
            (literal(5), size(5)),
            (Type::Any(AnyStyle::Explicit), gradual_size()),
            (int_type, gradual_size()),
            (
                quantified.clone(),
                Type::SymInt(SymInt::Symbolic(Box::new(quantified))),
            ),
            (
                type_var.clone(),
                Type::SymInt(SymInt::Symbolic(Box::new(type_var))),
            ),
            (
                Type::SymInt(SymInt::add(size(1), size(2))),
                Type::SymInt(SymInt::Literal(3)),
            ),
        ] {
            let middle = Type::Tuple(Tuple::Unbounded(Box::new(elt)));
            let shape = SymIntTuple::unpacked(vec![size(1)], middle, vec![size(2)]);

            assert_eq!(
                shape,
                SymIntTuple::unpacked(
                    vec![size(1)],
                    Type::Tuple(Tuple::Unbounded(Box::new(expected))),
                    vec![size(2)],
                )
            );
            assert_eq!(SymIntTuple::from_tuple(shape.to_tuple()), shape);
        }
    }

    #[test]
    fn ordinary_var_unbounded_middle_element_recovers_to_gradual() {
        let middle = Type::Tuple(Tuple::Unbounded(Box::new(Type::Var(Var::ZERO))));
        let shape = SymIntTuple::unpacked(vec![size(1)], middle, vec![size(2)]);

        assert_eq!(
            shape,
            SymIntTuple::unpacked(
                vec![size(1)],
                SymIntTuple::shapeless().to_shape_arg_type(),
                vec![size(2)],
            )
        );
        assert_eq!(SymIntTuple::from_tuple(shape.to_tuple()), shape);
    }

    #[test]
    fn invalid_quantified_middle_kinds_recover_as_shapeless() {
        for kind in [QuantifiedKind::SymIntVar, QuantifiedKind::ParamSpec] {
            let middle = Type::Quantified(Box::new(fake_tparam("Invalid", kind)));
            let shape = SymIntTuple::unpacked(vec![size(1)], middle, vec![size(2)]);

            assert_eq!(shape, SymIntTuple::shapeless());
            assert_eq!(SymIntTuple::from_tuple(shape.to_tuple()), shape);
        }
    }

    #[test]
    fn scalar_typevar_middle_is_preserved_as_tuple_carrier() {
        let quantified = Type::Quantified(Box::new(fake_tparam("Shape", QuantifiedKind::TypeVar)));
        let whole_shape = SymIntTuple::unpacked(Vec::new(), quantified.clone(), Vec::new());
        assert_eq!(
            whole_shape,
            SymIntTuple::unpacked(Vec::new(), quantified.clone(), Vec::new(),)
        );

        let affixed = SymIntTuple::unpacked(vec![size(1)], quantified.clone(), vec![size(2)]);
        assert_eq!(
            affixed,
            SymIntTuple::unpacked(vec![size(1)], quantified, vec![size(2)],)
        );

        let direct = Type::TypeVar(fake_type_var("Shape", QuantifiedKind::TypeVar));
        let whole_shape = SymIntTuple::unpacked(Vec::new(), direct.clone(), Vec::new());
        assert_eq!(
            whole_shape,
            SymIntTuple::unpacked(Vec::new(), direct.clone(), Vec::new(),)
        );
        let affixed = SymIntTuple::unpacked(vec![size(1)], direct.clone(), vec![size(2)]);
        assert_eq!(
            affixed,
            SymIntTuple::unpacked(vec![size(1)], direct, vec![size(2)],)
        );
    }

    #[test]
    fn true_unresolved_variadic_middles_are_preserved() {
        for middle in [
            Type::Quantified(Box::new(fake_tparam("Shape", QuantifiedKind::TypeVarTuple))),
            Type::TypeVarTuple(fake_type_var_tuple("Shape")),
            Type::Var(Var::ZERO),
        ] {
            let shape = SymIntTuple::unpacked(vec![size(1)], middle.clone(), vec![size(2)]);

            assert_eq!(
                shape,
                SymIntTuple::unpacked(vec![size(1)], middle, vec![size(2)],)
            );
            if is_tuple_carrier_shape_middle(match shape.view() {
                SymIntTupleView::Unpacked { middle, .. } => middle,
                _ => unreachable!("test constructs unpacked shapes"),
            }) {
                assert_eq!(
                    SymIntTuple::from_tuple(shape.to_tuple()),
                    SymIntTuple::unpacked(vec![size(1)], gradual_shape_middle(), vec![size(2)])
                );
            } else {
                assert_eq!(SymIntTuple::from_tuple(shape.to_tuple()), shape);
            }
        }
    }

    #[test]
    fn unpacked_carrier_round_trip() {
        // tuple[Literal[2], *Ts, Literal[3]] <-> Unpacked([2], Ts, [3]).
        let middle = Type::Var(Var::ZERO);
        let shape = SymIntTuple::unpacked(vec![size(2)], middle.clone(), vec![size(3)]);
        let carrier = shape_to_tuple_carrier(&shape);
        assert_eq!(
            carrier,
            Type::Tuple(Tuple::Unpacked(Box::new((
                vec![literal(2)],
                middle,
                vec![literal(3)],
            ))))
        );
        assert_eq!(tuple_carrier_to_shape(&carrier), Some(shape));
    }

    #[test]
    fn unbounded_carriers_canonicalize_to_shapeless() {
        // Unbounded carriers have no recoverable rank or per-dimension values,
        // regardless of their element type.
        let any_unbounded = Type::any_tuple();
        let internal_unbounded = Type::Tuple(Tuple::Unbounded(Box::new(size(5))));
        let int_unbounded = Type::Tuple(Tuple::Unbounded(Box::new(Type::ClassType(
            fake_class_type("builtins", "int"),
        ))));
        let shapeless = SymIntTuple::shapeless();
        assert_eq!(
            tuple_carrier_to_shape(&any_unbounded),
            Some(shapeless.clone())
        );
        assert_eq!(
            tuple_carrier_to_shape(&internal_unbounded),
            Some(shapeless.clone())
        );
        assert_eq!(tuple_carrier_to_shape(&int_unbounded), Some(shapeless));
    }

    #[test]
    fn unpacked_tuple_carrier_middle_is_validated_strictly() {
        let valid = Type::Tuple(Tuple::Unpacked(Box::new((
            vec![literal(1)],
            Type::Tuple(Tuple::Concrete(vec![literal(3)])),
            vec![literal(2)],
        ))));
        assert_eq!(
            tuple_carrier_to_shape(&valid),
            Some(SymIntTuple::from_types(vec![size(1), size(3), size(2)]))
        );

        let invalid = Type::Tuple(Tuple::Unpacked(Box::new((
            vec![literal(1)],
            Type::Tuple(Tuple::Concrete(vec![Type::ClassType(fake_class_type(
                "builtins", "str",
            ))])),
            vec![literal(2)],
        ))));
        assert_eq!(tuple_carrier_to_shape(&invalid), None);
    }

    #[test]
    fn nested_concrete_tuple_carrier_middle_is_recursively_flattened() {
        let carrier = Type::Tuple(Tuple::Unpacked(Box::new((
            vec![literal(1)],
            Type::Tuple(Tuple::Unpacked(Box::new((
                vec![literal(2)],
                Type::Tuple(Tuple::Concrete(vec![literal(3)])),
                vec![literal(4)],
            )))),
            vec![literal(5)],
        ))));

        assert_eq!(
            tuple_carrier_to_shape(&carrier),
            Some(SymIntTuple::from_types(vec![
                size(1),
                size(2),
                size(3),
                size(4),
                size(5),
            ]))
        );
    }

    #[test]
    fn nested_unbounded_tuple_carrier_middle_recovers_to_gradual() {
        let carrier = Type::Tuple(Tuple::Unpacked(Box::new((
            vec![literal(1)],
            Type::Tuple(Tuple::Unpacked(Box::new((
                vec![literal(2)],
                Type::Tuple(Tuple::Unbounded(Box::new(literal(3)))),
                vec![literal(4)],
            )))),
            vec![literal(5)],
        ))));

        assert_eq!(
            tuple_carrier_to_shape(&carrier),
            Some(SymIntTuple::unpacked(
                vec![size(1), size(2)],
                gradual_shape_middle(),
                vec![size(4), size(5)],
            ))
        );
    }

    #[test]
    fn unpacked_tuple_carrier_unbounded_middle_recovers_to_gradual() {
        let carrier = Type::Tuple(Tuple::Unpacked(Box::new((
            vec![literal(1)],
            Type::Tuple(Tuple::Unbounded(Box::new(literal(5)))),
            vec![literal(2)],
        ))));

        assert_eq!(
            tuple_carrier_to_shape(&carrier),
            Some(SymIntTuple::unpacked(
                vec![size(1)],
                gradual_shape_middle(),
                vec![size(2)],
            ))
        );

        let invalid_prefix = Type::Tuple(Tuple::Unpacked(Box::new((
            vec![Type::ClassType(fake_class_type("builtins", "str"))],
            Type::Tuple(Tuple::Unbounded(Box::new(literal(5)))),
            vec![literal(2)],
        ))));
        assert_eq!(tuple_carrier_to_shape(&invalid_prefix), None);
    }

    #[test]
    fn bare_var_internal_tuple_middle_scalar_position_recovers_to_gradual_dim() {
        let middle = Type::Tuple(Tuple::Concrete(vec![Type::Var(Var::ZERO)]));
        let shape = SymIntTuple::unpacked(vec![size(1)], middle, vec![size(2)]);

        assert_eq!(
            shape,
            SymIntTuple::from_types(vec![size(1), gradual_size(), size(2)])
        );
    }

    #[test]
    fn bare_var_tuple_carrier_scalar_position_recovers_to_gradual_dim() {
        assert_eq!(
            tuple_carrier_to_shape(&concrete_carrier(vec![Type::Var(Var::ZERO)])),
            Some(SymIntTuple::from_types(vec![gradual_size()]))
        );
    }

    #[test]
    fn unsupported_carrier_elements_fail() {
        // A non-int literal element is not a valid dimension.
        let carrier = concrete_carrier(vec![LitInt::new(0).to_explicit_type(), bool_literal()]);
        assert_eq!(tuple_carrier_to_shape(&carrier), None);
        // A non-tuple carrier is not convertible at all.
        assert_eq!(tuple_carrier_to_shape(&literal(3)), None);
    }

    #[test]
    fn invalid_quantified_carrier_elements_fail_front_door() {
        for elt in [
            Type::Quantified(Box::new(fake_tparam("T", QuantifiedKind::TypeVar))),
            Type::Quantified(Box::new(fake_tparam("P", QuantifiedKind::ParamSpec))),
            Type::Quantified(Box::new(fake_tparam("Ts", QuantifiedKind::TypeVarTuple))),
            Type::TypeVar(fake_type_var("T", QuantifiedKind::TypeVar)),
            Type::TypeVarTuple(fake_type_var_tuple("Ts")),
        ] {
            assert_eq!(tuple_carrier_to_shape(&concrete_carrier(vec![elt])), None);
        }
    }

    #[test]
    fn unsupported_symint_operands_fail() {
        let invalid_symint = Type::SymInt(SymInt::Symbolic(Box::new(literal(1))));
        assert_eq!(
            tuple_carrier_to_shape(&concrete_carrier(vec![invalid_symint.clone()])),
            None
        );
    }

    fn bool_literal() -> Type {
        Type::Literal(Box::new(Literal {
            value: Lit::Bool(true),
            style: LitStyle::Explicit,
        }))
    }
}
