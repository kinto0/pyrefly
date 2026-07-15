/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Dimension types and operations for tensor shape inference.
//!
//! This module provides:
//! - `SymInt`: Symbolic integer expressions (literals, arithmetic operations)
//! - Simplification: Algebraic simplification of dimension expressions
//! - Canonicalization: Normalization to unique canonical forms for comparison

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::fmt::Display;

use crate::equality::TypeEq;
use crate::literal::Lit;
use crate::quantified::QuantifiedKind;
use crate::types::AnyStyle;
use crate::types::Type;

/// A dimension expression in a tensor shape.
///
/// Dimensions can be:
/// - Concrete literals: `Tensor[2, 3]`
/// - Symbolic expressions: `Tensor[N, N+1]`, `Tensor[N*M]`
///
/// Type variables and solver variables are represented as symbolic leaves.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SymInt {
    /// Concrete dimension: Tensor[2, 3]
    /// Only positive integers are allowed
    Literal(i64),

    /// The gradual integer size: bare `SymInt`, `SymInt[int]`, or `Any` in a shape
    /// context. It is consistent with concrete sizes without becoming arbitrary
    /// `Any`.
    Int,

    /// A symbolic dimension leaf, typically a quantified type parameter or a
    /// solver variable standing for one.
    Symbolic(Box<Type>),

    /// Addition: N + M (for concat, etc.)
    Add(Box<SymInt>, Box<SymInt>),

    /// Subtraction: N - M
    Sub(Box<SymInt>, Box<SymInt>),

    /// Multiplication: N * M (for reshape, etc.)
    Mul(Box<SymInt>, Box<SymInt>),

    /// Floor division: N // M
    FloorDiv(Box<SymInt>, Box<SymInt>),

    /// Exponentiation: N ** M (for geometric progressions)
    Pow(Box<SymInt>, Box<SymInt>),
}

impl SymInt {
    pub fn literal(value: i64) -> Self {
        Self::Literal(value)
    }

    pub fn as_literal(&self) -> Option<i64> {
        match self {
            Self::Literal(n) => Some(*n),
            _ => None,
        }
    }

    pub fn is_literal(&self) -> bool {
        matches!(self, Self::Literal(_))
    }

    /// Helper constructors for expressions.
    /// Take Type arguments to support type variables in expressions.
    pub fn add(left: Type, right: Type) -> Self {
        Self::Add(
            Box::new(Self::from_type_or_legacy_symbolic(left)),
            Box::new(Self::from_type_or_legacy_symbolic(right)),
        )
    }

    pub fn sub(left: Type, right: Type) -> Self {
        Self::Sub(
            Box::new(Self::from_type_or_legacy_symbolic(left)),
            Box::new(Self::from_type_or_legacy_symbolic(right)),
        )
    }

    pub fn mul(left: Type, right: Type) -> Self {
        Self::Mul(
            Box::new(Self::from_type_or_legacy_symbolic(left)),
            Box::new(Self::from_type_or_legacy_symbolic(right)),
        )
    }

    pub fn floor_div(left: Type, right: Type) -> Self {
        Self::FloorDiv(
            Box::new(Self::from_type_or_legacy_symbolic(left)),
            Box::new(Self::from_type_or_legacy_symbolic(right)),
        )
    }

    pub fn pow(left: Type, right: Type) -> Self {
        Self::Pow(
            Box::new(Self::from_type_or_legacy_symbolic(left)),
            Box::new(Self::from_type_or_legacy_symbolic(right)),
        )
    }

    /// Convert a `Type` to a `SymInt`.
    pub fn from_type(ty: &Type) -> Option<SymInt> {
        match ty {
            Type::SymInt(dim) => Some(dim.clone()),
            Type::Literal(lit) if let Lit::Int(i) = &lit.value => i.as_i64().map(SymInt::Literal),
            Type::Quantified(q) if q.kind() == QuantifiedKind::SymVar => {
                Some(SymInt::Symbolic(Box::new(ty.clone())))
            }
            Type::Var(_) => Some(SymInt::Symbolic(Box::new(ty.clone()))),
            _ => None,
        }
    }

    fn from_type_or_legacy_symbolic(ty: Type) -> SymInt {
        Self::from_type(&ty).unwrap_or_else(|| Self::Symbolic(Box::new(ty)))
    }

    fn into_type(self) -> Type {
        Type::SymInt(self)
    }
}

/// The gradual size type: the internal representation of bare `SymInt` and
/// `SymInt[int]`.
pub fn gradual_size() -> Type {
    Type::SymInt(SymInt::Int)
}

/// Whether `ty` is the gradual size type.
pub fn is_gradual_size(ty: &Type) -> bool {
    matches!(ty, Type::SymInt(SymInt::Int))
}

impl Display for SymInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Literal(n) => write!(f, "{}", n),
            Self::Int => write!(f, "int"),
            Self::Symbolic(ty) => write!(f, "{}", ty),
            Self::Add(left, right) => write!(f, "({} + {})", left, right),
            Self::Sub(left, right) => write!(f, "({} - {})", left, right),
            Self::Mul(left, right) => {
                // Simplify display: (1 * x) -> x, (x * 1) -> x
                match (left.as_ref(), right.as_ref()) {
                    (SymInt::Literal(1), _) => write!(f, "{}", right),
                    (_, SymInt::Literal(1)) => write!(f, "{}", left),
                    _ => write!(f, "({} * {})", left, right),
                }
            }
            Self::FloorDiv(left, right) => {
                // Simplify display: (x // 1) -> x
                if matches!(right.as_ref(), SymInt::Literal(1)) {
                    write!(f, "{}", left)
                } else {
                    write!(f, "({} // {})", left, right)
                }
            }
            Self::Pow(left, right) => {
                write!(f, "({} ** {})", left, right)
            }
        }
    }
}

// ============================================================================
// Canonicalization
// ============================================================================

/// Canonicalize a dimension expression to a unique normal form.
///
/// This transforms dimension expressions into a canonical form where:
/// - Like terms are combined (e.g., 4*N + 2*N = 6*N)
/// - Divisions are flattened (e.g., (N // M) // K = N // (M*K))
/// - Factors are GCD-reduced (e.g., (4*N) // (6*M) = (2*N) // (3*M))
/// - Expressions are ordered consistently
/// - Gradual sizes propagate through the entire expression
///
/// This enables structural equality checking after canonicalization.
pub fn canonicalize(ty: Type) -> Type {
    // Normalize and canonicalize based on type
    match ty {
        Type::SymInt(dim) => {
            if symint_is_gradual(&dim) {
                return gradual_size();
            }
            canonicalize_symint(dim)
        }
        // Quantified, Var, Any, and Literal are already canonical
        other => other,
    }
}

/// Inner canonicalization that skips the Any check.
/// Called after the top-level `canonicalize` has already verified no Any is present.
fn canonicalize_inner(ty: Type) -> Type {
    match ty {
        Type::SymInt(dim) => canonicalize_symint(dim),
        other => other,
    }
}

/// Whether a type is gradual for size purposes.
///
/// For a `Size` type this is equivalent to `is_gradual_size(&canonicalize(ty))`
/// but avoids allocating a canonicalized copy, so it is cheap to call on a hot
/// path before deciding whether canonicalization is needed at all.
pub fn type_is_gradual(ty: &Type) -> bool {
    match ty {
        Type::SymInt(dim) => symint_is_gradual(dim),
        Type::Any(AnyStyle::Implicit | AnyStyle::Explicit) => true,
        _ => false,
    }
}

/// Whether a symbolic integer expression contains a gradual leaf.
fn symint_is_gradual(dim: &SymInt) -> bool {
    match dim {
        SymInt::Int => true,
        SymInt::Symbolic(ty) => symbolic_is_gradual(ty),
        SymInt::Add(left, right)
        | SymInt::Sub(left, right)
        | SymInt::Mul(left, right)
        | SymInt::FloorDiv(left, right)
        | SymInt::Pow(left, right) => symint_is_gradual(left) || symint_is_gradual(right),
        SymInt::Literal(_) => false,
    }
}

/// Whether canonicalizing a `Symbolic` leaf yields the gradual size. A builtin
/// `int` leaf canonicalizes to the gradual size via `canonicalize_symbolic`, so
/// it must be reported as gradual here even though it is not itself a `Size`.
/// Every other leaf falls back to `type_is_gradual`, which already matches how
/// `canonicalize` short-circuits gradual (`Any`) and nested `Size` leaves. This
/// keeps `type_is_gradual` equivalent to
/// `is_gradual_size(&canonicalize(..))` without allocating a canonical copy.
fn symbolic_is_gradual(ty: &Type) -> bool {
    match ty {
        Type::ClassType(cls) if cls.is_builtin("int") => true,
        _ => type_is_gradual(ty),
    }
}

/// Main canonicalization function for symbolic integer expressions.
fn canonicalize_symint(dim: SymInt) -> Type {
    match dim {
        SymInt::Literal(_) | SymInt::Int => Type::SymInt(dim),
        SymInt::Symbolic(ty) => canonicalize_symbolic(*ty),
        SymInt::Add(left, right) => canonicalize_sum(left.into_type(), right.into_type()),
        SymInt::Sub(left, right) => {
            // Normalize: a - b → a + (-1) * b
            let neg_one = Type::SymInt(SymInt::Literal(-1));
            let neg_right = Type::SymInt(SymInt::mul(neg_one, right.into_type()));
            canonicalize_sum(left.into_type(), neg_right)
        }
        SymInt::Mul(left, right) => canonicalize_product(left.into_type(), right.into_type()),
        SymInt::FloorDiv(left, right) => canonicalize_division(left.into_type(), right.into_type()),
        SymInt::Pow(left, right) => canonicalize_pow(left.into_type(), right.into_type()),
    }
}

fn canonicalize_symbolic(ty: Type) -> Type {
    match ty {
        Type::SymInt(dim) => canonicalize_symint(dim),
        Type::Literal(lit) => {
            let n = match &lit.value {
                Lit::Int(i) => i.as_i64(),
                _ => unreachable!(
                    "only integer literals can be converted into symbolic size expressions"
                ),
            };
            n.map(|n| Type::SymInt(SymInt::Literal(n)))
                .unwrap_or_else(|| Type::SymInt(SymInt::Symbolic(Box::new(Type::Literal(lit)))))
        }
        Type::ClassType(cls) if cls.is_builtin("int") => gradual_size(),
        other => Type::SymInt(SymInt::Symbolic(Box::new(other))),
    }
}

/// Canonicalize a sum expression
fn canonicalize_sum(left: Type, right: Type) -> Type {
    // Step 1: Recursively canonicalize operands
    let left_canon = canonicalize_inner(left);
    let right_canon = canonicalize_inner(right);

    // Step 2: Flatten to list of terms
    let mut terms = Vec::new();
    collect_terms(left_canon, &mut terms);
    collect_terms(right_canon, &mut terms);

    // Step 3: Combine like terms by extracting coefficients
    #[allow(clippy::mutable_key_type)]
    let mut term_map: HashMap<Type, i64> = HashMap::new();

    for term in terms {
        let (coeff, non_literal_part) = extract_coefficient(term);
        *term_map.entry(non_literal_part).or_insert(0) += coeff;
    }

    // Step 4: Rebuild terms, filtering out zero coefficients
    let mut new_terms = Vec::new();
    for (part, coeff) in term_map {
        if coeff == 0 {
            continue;
        }

        if matches!(part, Type::SymInt(SymInt::Literal(1))) {
            // Coefficient only (no non-literal part)
            new_terms.push(Type::SymInt(SymInt::Literal(coeff)));
        } else if coeff == 1 {
            // Coefficient is 1, just use the part
            new_terms.push(part);
        } else {
            // General case: coeff * part
            let coeff_ty = Type::SymInt(SymInt::Literal(coeff));
            new_terms.push(Type::SymInt(SymInt::mul(coeff_ty, part)));
        }
    }

    // Step 5: Sort terms by canonical order
    new_terms.sort_by(compare_type);

    // Step 6: Build result
    rebuild_sum(new_terms)
}

/// Generic function to collect operands from a binary symbolic integer expression.
fn collect_operands(
    ty: Type,
    items: &mut Vec<Type>,
    extract: fn(&SymInt) -> Option<(&SymInt, &SymInt)>,
) {
    match &ty {
        Type::SymInt(dim) => {
            if let Some((left, right)) = extract(dim) {
                collect_operands(left.clone().into_type(), items, extract);
                collect_operands(right.clone().into_type(), items, extract);
            } else {
                items.push(ty);
            }
        }
        _ => items.push(ty),
    }
}

fn extract_add_operands(dim: &SymInt) -> Option<(&SymInt, &SymInt)> {
    match dim {
        SymInt::Add(l, r) => Some((l.as_ref(), r.as_ref())),
        _ => None,
    }
}

fn extract_mul_operands(dim: &SymInt) -> Option<(&SymInt, &SymInt)> {
    match dim {
        SymInt::Mul(l, r) => Some((l.as_ref(), r.as_ref())),
        _ => None,
    }
}

fn collect_terms(ty: Type, terms: &mut Vec<Type>) {
    collect_operands(ty, terms, extract_add_operands);
}

/// Rebuild a sum expression from a list of terms.
fn rebuild_sum(terms: Vec<Type>) -> Type {
    if terms.is_empty() {
        Type::SymInt(SymInt::Literal(0))
    } else if terms.len() == 1 {
        terms.into_iter().next().unwrap()
    } else {
        let mut iter = terms.into_iter();
        let first = iter.next().unwrap();
        iter.fold(first, |acc, term| Type::SymInt(SymInt::add(acc, term)))
    }
}

/// Separate literal factors from non-literal factors, computing their product.
fn separate_literal_factors(factors: Vec<Type>) -> (i64, Vec<Type>) {
    let literal_product: i64 = factors
        .iter()
        .filter_map(|f| f.as_shape_literal())
        .product();

    let non_literal: Vec<Type> = factors
        .into_iter()
        .filter(|f| f.as_shape_literal().is_none())
        .collect();

    (literal_product, non_literal)
}

/// Extract coefficient and non-literal part from a term
fn extract_coefficient(term: Type) -> (i64, Type) {
    match term {
        Type::SymInt(SymInt::Literal(n)) => (n, Type::SymInt(SymInt::Literal(1))),
        Type::SymInt(SymInt::Mul(_, _)) => {
            // Collect all factors
            let mut factors = Vec::new();
            collect_factors(term, &mut factors);

            // Separate literal from non-literal factors
            let (coeff, non_literal_factors) = separate_literal_factors(factors);

            let non_literal_part = if non_literal_factors.is_empty() {
                Type::SymInt(SymInt::Literal(1))
            } else {
                rebuild_product(non_literal_factors)
            };

            (coeff, non_literal_part)
        }
        other => (1, other),
    }
}

/// Canonicalize a product expression
fn canonicalize_product(left: Type, right: Type) -> Type {
    // Step 1: Recursively canonicalize operands
    let left_canon = canonicalize_inner(left);
    let right_canon = canonicalize_inner(right);

    // Step 2: Flatten to list of factors
    let mut factors = Vec::new();
    collect_factors(left_canon, &mut factors);
    collect_factors(right_canon, &mut factors);

    // Step 3: Check for zero
    if factors
        .iter()
        .any(|f| matches!(f, Type::SymInt(SymInt::Literal(0))))
    {
        return Type::SymInt(SymInt::Literal(0));
    }

    // Step 4: Separate literals from non-literals
    let (mut literal_product, mut non_literal_factors) = separate_literal_factors(factors);

    // Step 4b: Group same-base Pow factors and absorb matching literals.
    // For example: 2 * 2**(I-1) → 2**(I-1+1) → 2**I
    // Literal factors that equal a Pow base are converted to base**1 and merged.
    if non_literal_factors
        .iter()
        .any(|f| matches!(f, Type::SymInt(SymInt::Pow(_, _))))
    {
        #[allow(clippy::mutable_key_type)]
        let mut pow_groups: HashMap<Type, Vec<Type>> = HashMap::new();
        let mut remaining = Vec::new();

        for factor in non_literal_factors.drain(..) {
            if let Type::SymInt(SymInt::Pow(base, exp)) = factor {
                pow_groups
                    .entry(base.into_type())
                    .or_default()
                    .push(exp.into_type());
            } else {
                remaining.push(factor);
            }
        }

        // Check if literal_product matches any Pow base
        for (base, exponents) in &mut pow_groups {
            if let Some(base_val) = base.as_shape_literal()
                && literal_product != 1
                && base_val != 0
            {
                let (k, remainder) = extract_base_power(literal_product, base_val);
                if k > 0 {
                    exponents.push(Type::SymInt(SymInt::Literal(k)));
                    literal_product = remainder;
                }
            }
        }

        // Rebuild: combine each group into base ** sum(exponents)
        non_literal_factors = remaining;
        for (base, exponents) in pow_groups {
            // Build raw sum of exponents; canonicalize_pow will canonicalize it
            // via canonicalize_inner on the exponent
            let exp_sum = exponents
                .into_iter()
                .reduce(|acc, e| Type::SymInt(SymInt::add(acc, e)))
                .unwrap();
            let combined = canonicalize_pow(base, exp_sum);
            match &combined {
                Type::SymInt(SymInt::Literal(n)) => {
                    literal_product *= n;
                }
                _ => {
                    non_literal_factors.push(combined);
                }
            }
        }
    }

    // Step 5: Distributive law — coeff * (a + b) → coeff*a + coeff*b
    // When any factor (literal, symbolic, or both) multiplies a sum, distribute
    // across the sum terms. This enables like-term cancellation at the caller's
    // sum level. For example:
    //   4 * (N + 2)       → 4*N + 8           (literal coefficient)
    //   GR * (I + (-1))   → GR*I + (-1)*GR    (symbolic coefficient)
    //   2 * GR * (I + 3)  → 2*GR*I + 6*GR     (mixed coefficient)
    if let Some(sum_idx) = non_literal_factors
        .iter()
        .position(|f| matches!(f, Type::SymInt(SymInt::Add(_, _))))
    {
        // Only distribute if there's at least one other factor to distribute
        let has_other_factors = literal_product != 1 || non_literal_factors.len() > 1;
        if has_other_factors {
            let sum = non_literal_factors.remove(sum_idx);

            // Build coefficient from literal and remaining non-literal factors
            let mut coeff_factors = Vec::new();
            if literal_product != 1 {
                coeff_factors.push(Type::SymInt(SymInt::Literal(literal_product)));
            }
            coeff_factors.extend(non_literal_factors);
            let coeff = rebuild_product(coeff_factors);

            // Distribute coefficient across each sum term
            let mut terms = Vec::new();
            collect_terms(sum, &mut terms);
            let distributed_terms: Vec<Type> = terms
                .into_iter()
                .map(|term| {
                    let product = Type::SymInt(SymInt::mul(coeff.clone(), term));
                    canonicalize_inner(product)
                })
                .collect();
            return rebuild_sum(distributed_terms);
        }
    }

    // Step 6: Sort factors by canonical order
    non_literal_factors.sort_by(compare_type);

    // Step 7: Add literal coefficient if not 1
    let mut all_factors = Vec::new();
    if literal_product != 1 {
        all_factors.push(Type::SymInt(SymInt::Literal(literal_product)));
    }
    all_factors.extend(non_literal_factors);

    // Step 8: Build result
    if all_factors.is_empty() {
        Type::SymInt(SymInt::Literal(1))
    } else {
        rebuild_product(all_factors)
    }
}

fn collect_factors(ty: Type, factors: &mut Vec<Type>) {
    collect_operands(ty, factors, extract_mul_operands);
}

fn rebuild_product(factors: Vec<Type>) -> Type {
    if factors.is_empty() {
        Type::SymInt(SymInt::Literal(1))
    } else if factors.len() == 1 {
        factors.into_iter().next().unwrap()
    } else {
        let mut iter = factors.into_iter();
        let first = iter.next().unwrap();
        iter.fold(first, |acc, f| Type::SymInt(SymInt::mul(acc, f)))
    }
}

/// Canonicalize a floor division expression
fn canonicalize_division(num: Type, den: Type) -> Type {
    // Step 1: Canonicalize the numerator
    let canonical_num = canonicalize_inner(num);

    // Step 2: Check if numerator is a division - if so, flatten
    if let Type::SymInt(SymInt::FloorDiv(inner_num, inner_den)) = canonical_num {
        // Apply composition law: (a // b) // c = a // (b * c)
        let new_den = Type::SymInt(SymInt::mul(inner_den.into_type(), den));
        return canonicalize_division(inner_num.into_type(), new_den);
    }

    // Step 3: Now canonicalize the denominator
    let canonical_den = canonicalize_inner(den);

    // Step 4: Apply simplifications
    match (&canonical_num, &canonical_den) {
        // 0 // a = 0
        (Type::SymInt(SymInt::Literal(0)), _) => Type::SymInt(SymInt::Literal(0)),

        // a // 1 = a
        (_, Type::SymInt(SymInt::Literal(1))) => canonical_num,

        // Both literals: compute
        (Type::SymInt(SymInt::Literal(n)), Type::SymInt(SymInt::Literal(d))) if *d != 0 => {
            Type::SymInt(SymInt::Literal(n / d))
        }

        // Literal term extraction from sum numerator:
        // (a + k*d + b) // d  →  k + (a + b) // d
        // Sound because (k*d + r) // d = k + r // d for all integers k, d, r (d ≠ 0).
        // Enables: (H - 2) // 2 + 1  →  -1 + H // 2 + 1  →  H // 2
        (Type::SymInt(SymInt::Add(_, _)), Type::SymInt(SymInt::Literal(d))) if *d != 0 => {
            let d = *d;
            let mut terms = Vec::new();
            collect_terms(canonical_num, &mut terms);
            let original_count = terms.len();

            // Partition into extractable (literal multiples of d) and remaining
            let mut extracted_sum: i64 = 0;
            let mut remaining = Vec::new();
            for term in terms {
                if let Type::SymInt(SymInt::Literal(n)) = &term
                    && n % d == 0
                {
                    extracted_sum += n / d;
                    continue;
                }
                remaining.push(term);
            }

            if extracted_sum == 0 && remaining.len() == original_count {
                // Nothing extracted — fall through to cancellation
                let (new_num, new_den) =
                    try_cancel_common_factors(rebuild_sum(remaining), canonical_den);
                if matches!(new_den, Type::SymInt(SymInt::Literal(1))) {
                    new_num
                } else {
                    Type::SymInt(SymInt::floor_div(new_num, new_den))
                }
            } else if remaining.is_empty() {
                // All terms extracted — result is just the extracted literal
                Type::SymInt(SymInt::Literal(extracted_sum))
            } else {
                // Some terms extracted: extracted_sum + remaining // d
                let remainder_div = Type::SymInt(SymInt::FloorDiv(
                    Box::new(SymInt::from_type_or_legacy_symbolic(rebuild_sum(remaining))),
                    Box::new(SymInt::Literal(d)),
                ));
                if extracted_sum == 0 {
                    remainder_div
                } else {
                    canonicalize_sum(Type::SymInt(SymInt::Literal(extracted_sum)), remainder_div)
                }
            }
        }

        // Sum numerator, non-literal denominator: try un-distributing the sum.
        // The distributive law in canonicalize_product expands B*(2*A-1) into
        // -B + 2*A*B. When this sum is divided by (2*A-1), we need to factor
        // the common factor B back out to recover B*(2*A-1) and cancel.
        (Type::SymInt(SymInt::Add(_, _)), _) => {
            if let Some(result) = try_factor_sum_and_cancel(&canonical_num, &canonical_den) {
                result
            } else {
                let (new_num, new_den) = try_cancel_common_factors(canonical_num, canonical_den);
                if matches!(new_den, Type::SymInt(SymInt::Literal(1))) {
                    new_num
                } else {
                    Type::SymInt(SymInt::floor_div(new_num, new_den))
                }
            }
        }

        // Try cancellation
        _ => {
            let (new_num, new_den) = try_cancel_common_factors(canonical_num, canonical_den);

            // If denominator is 1 after cancellation, return numerator
            if matches!(new_den, Type::SymInt(SymInt::Literal(1))) {
                new_num
            } else {
                Type::SymInt(SymInt::floor_div(new_num, new_den))
            }
        }
    }
}

/// Canonicalize an exponentiation expression.
///
/// Rules (checked in this order):
/// 1. Exponent 0 → 1 (for any base)
/// 2. Exponent 1 → base (avoids allocation, reuses canon_base)
/// 3. Both concrete → compute the literal (e.g., 2**3 → 8), with overflow check
/// 4. Nested Pow: (a**b)**c → a**(b*c)
/// 5. Otherwise: Pow(canon_base, canon_exponent)
fn canonicalize_pow(base: Type, exp: Type) -> Type {
    let canon_base = canonicalize_inner(base);
    let canon_exp = canonicalize_inner(exp);

    match (&canon_base, &canon_exp) {
        // a ** 0 = 1
        (_, Type::SymInt(SymInt::Literal(0))) => Type::SymInt(SymInt::Literal(1)),

        // a ** 1 = a
        (_, Type::SymInt(SymInt::Literal(1))) => canon_base,

        // Both literals: compute base^exp with overflow protection
        (Type::SymInt(SymInt::Literal(b)), Type::SymInt(SymInt::Literal(e))) => {
            if *e >= 0 && *e <= 63 {
                match b.checked_pow(*e as u32) {
                    Some(result) => Type::SymInt(SymInt::Literal(result)),
                    None => {
                        // Overflow: keep symbolic
                        Type::SymInt(SymInt::pow(canon_base, canon_exp))
                    }
                }
            } else {
                // Negative exponent: not meaningful for integer dimensions
                Type::SymInt(SymInt::pow(canon_base, canon_exp))
            }
        }

        // (a ** b) ** c = a ** (b * c)
        (Type::SymInt(SymInt::Pow(inner_base, inner_exp)), _) => {
            let new_exp = Type::SymInt(SymInt::mul(inner_exp.clone().into_type(), canon_exp));
            canonicalize_pow(inner_base.clone().into_type(), new_exp)
        }

        _ => Type::SymInt(SymInt::pow(canon_base, canon_exp)),
    }
}

/// Try to cancel common factors between numerator and denominator
fn try_cancel_common_factors(num: Type, den: Type) -> (Type, Type) {
    // Extract factors from numerator and denominator
    let mut num_factors = Vec::new();
    let mut den_factors = Vec::new();
    collect_factors(num, &mut num_factors);
    collect_factors(den, &mut den_factors);

    // Step 1: Separate literals from non-literals
    let (num_literal, mut num_factors) = separate_literal_factors(num_factors);
    let (den_literal, mut den_factors) = separate_literal_factors(den_factors);

    // Step 2: Apply GCD to literals
    let g = gcd(num_literal.abs(), den_literal.abs());
    let new_num_literal = num_literal / g;
    let new_den_literal = den_literal / g;

    // Step 3: Find and remove structurally equal non-literal factors
    let mut i = 0;
    while i < num_factors.len() {
        if let Some(pos) = den_factors.iter().position(|df| num_factors[i] == *df) {
            // Found a common factor - cancel it
            num_factors.remove(i);
            den_factors.remove(pos);
            // Don't increment i, check the same position again
        } else {
            i += 1;
        }
    }

    // Step 4: Rebuild numerator
    if new_num_literal != 1 {
        num_factors.insert(0, Type::SymInt(SymInt::Literal(new_num_literal)));
    }
    let new_num = rebuild_product(num_factors);

    // Step 5: Rebuild denominator
    if new_den_literal != 1 {
        den_factors.insert(0, Type::SymInt(SymInt::Literal(new_den_literal)));
    }
    let new_den = rebuild_product(den_factors);

    (new_num, new_den)
}

/// Try to factor a common factor out of a sum numerator and cancel with the denominator.
///
/// When canonicalize_product distributes B*(2*A-1) into -B + 2*A*B, this function
/// reverses the expansion inside division context:
///   (-B + 2*A*B) // (-1 + 2*A)
///   → terms: [-1*B, 2*A*B], common non-literal factor: B
///   → B * (-1 + 2*A) // (-1 + 2*A) → B
///
/// Only simplifies when ALL sum terms share the common factor (exact divisibility).
fn try_factor_sum_and_cancel(num: &Type, den: &Type) -> Option<Type> {
    let mut terms = Vec::new();
    collect_terms(num.clone(), &mut terms);

    if terms.len() < 2 {
        return None;
    }

    // For each term, extract literal coefficient and non-literal factors.
    let term_factorizations: Vec<(i64, Vec<Type>)> = terms
        .iter()
        .map(|term| {
            let mut factors = Vec::new();
            collect_factors(term.clone(), &mut factors);
            separate_literal_factors(factors)
        })
        .collect();

    // Find common non-literal factors across ALL terms (set intersection).
    let mut common_factors: Vec<Type> = term_factorizations[0].1.clone();
    for (_, non_lit_factors) in &term_factorizations[1..] {
        let mut remaining = non_lit_factors.clone();
        common_factors.retain(|cf| {
            if let Some(pos) = remaining.iter().position(|f| f == cf) {
                remaining.remove(pos);
                true
            } else {
                false
            }
        });
    }

    if common_factors.is_empty() {
        return None;
    }

    // Factor out common factors from each term, rebuilding quotient terms.
    let quotient_terms: Vec<Type> = term_factorizations
        .iter()
        .map(|(coeff, non_lit_factors)| {
            let mut remaining = non_lit_factors.clone();
            for cf in &common_factors {
                if let Some(pos) = remaining.iter().position(|f| f == cf) {
                    remaining.remove(pos);
                }
            }
            let mut all_factors = Vec::new();
            if *coeff != 1 {
                all_factors.push(Type::SymInt(SymInt::Literal(*coeff)));
            }
            all_factors.extend(remaining);
            rebuild_product(all_factors)
        })
        .collect();

    // Canonicalize the quotient sum so it can be compared structurally with the denominator.
    let quotient_sum = rebuild_sum(quotient_terms);
    let canonical_quotient = canonicalize_inner(quotient_sum);

    // Numerator = common_factors * canonical_quotient (as a product).
    // Try cancelling this product with the denominator.
    let mut all_num_factors = common_factors;
    all_num_factors.push(canonical_quotient);
    let factored_num = rebuild_product(all_num_factors);

    let (new_num, new_den) = try_cancel_common_factors(factored_num, den.clone());
    if matches!(new_den, Type::SymInt(SymInt::Literal(1))) {
        Some(new_num)
    } else {
        None
    }
}

/// Decompose `value` as `base^k * remainder` where k is maximized.
/// Returns (k, remainder). For example: extract_base_power(8, 2) = (3, 1),
/// extract_base_power(12, 2) = (2, 3), extract_base_power(7, 2) = (0, 7).
fn extract_base_power(mut value: i64, base: i64) -> (i64, i64) {
    if base.abs() <= 1 {
        return (0, value);
    }
    let mut k = 0;
    while value != 0 && value % base == 0 {
        value /= base;
        k += 1;
    }
    (k, value)
}

fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

/// Compare types for canonical ordering.
/// Ordering: Literal < Int < Quantified < Var < SymInt(Symbolic) < SymInt(FloorDiv) < SymInt(Mul) < SymInt(Add) < SymInt(Sub)
fn compare_type(a: &Type, b: &Type) -> Ordering {
    match (a, b) {
        // Literals: compare numerically
        (Type::SymInt(SymInt::Literal(n1)), Type::SymInt(SymInt::Literal(n2))) => n1.cmp(n2),

        // Literals come first
        (Type::SymInt(SymInt::Literal(_)), _) => Ordering::Less,
        (_, Type::SymInt(SymInt::Literal(_))) => Ordering::Greater,

        (Type::SymInt(SymInt::Int), Type::SymInt(SymInt::Int)) => Ordering::Equal,
        (Type::SymInt(SymInt::Int), _) => Ordering::Less,
        (_, Type::SymInt(SymInt::Int)) => Ordering::Greater,

        // Quantified (type parameters)
        (Type::Quantified(q1), Type::Quantified(q2)) => q1.cmp(q2),
        (Type::Quantified(_), _) => Ordering::Less,
        (_, Type::Quantified(_)) => Ordering::Greater,

        // Var (solver variables, created during generic instantiation)
        (Type::Var(v1), Type::Var(v2)) => v1.cmp(v2),
        (Type::Var(_), _) => Ordering::Less,
        (_, Type::Var(_)) => Ordering::Greater,

        // SymInt variants
        (Type::SymInt(d1), Type::SymInt(d2)) => compare_symint(d1, d2),

        // Symbolic integer expressions come after non-symbolic-int types.
        (Type::SymInt(_), _) => Ordering::Greater,
        (_, Type::SymInt(_)) => Ordering::Less,

        // Fallback: types that shouldn't appear in dimension expressions
        _ => Ordering::Equal,
    }
}

fn compare_symint(a: &SymInt, b: &SymInt) -> Ordering {
    use SymInt::*;
    match (a, b) {
        (Literal(n1), Literal(n2)) => n1.cmp(n2),
        (Int, Int) => Ordering::Equal,
        (Symbolic(t1), Symbolic(t2)) => compare_type(t1, t2),

        // Type ordering: Literal < Int < Symbolic < FloorDiv < Pow < Mul < Add < Sub
        (Literal(_), _) => Ordering::Less,
        (_, Literal(_)) => Ordering::Greater,

        (Int, _) => Ordering::Less,
        (_, Int) => Ordering::Greater,

        (Symbolic(_), _) => Ordering::Less,
        (_, Symbolic(_)) => Ordering::Greater,

        (FloorDiv(_, _), Pow(_, _) | Mul(_, _) | Add(_, _) | Sub(_, _)) => Ordering::Less,
        (Pow(_, _) | Mul(_, _) | Add(_, _) | Sub(_, _), FloorDiv(_, _)) => Ordering::Greater,

        (Pow(_, _), Mul(_, _) | Add(_, _) | Sub(_, _)) => Ordering::Less,
        (Mul(_, _) | Add(_, _) | Sub(_, _), Pow(_, _)) => Ordering::Greater,

        (Mul(_, _), Add(_, _) | Sub(_, _)) => Ordering::Less,
        (Add(_, _) | Sub(_, _), Mul(_, _)) => Ordering::Greater,

        (Add(_, _), Sub(_, _)) => Ordering::Less,
        (Sub(_, _), Add(_, _)) => Ordering::Greater,

        // Same variant: compare lexicographically
        (FloorDiv(n1, d1), FloorDiv(n2, d2))
        | (Pow(n1, d1), Pow(n2, d2))
        | (Mul(n1, d1), Mul(n2, d2))
        | (Add(n1, d1), Add(n2, d2))
        | (Sub(n1, d1), Sub(n2, d2)) => match compare_symint(n1, n2) {
            Ordering::Equal => compare_symint(d1, d2),
            other => other,
        },
    }
}

// ============================================================================
// Trait Implementations
// ============================================================================

impl pyrefly_util::visit::Visit<Type> for SymInt {
    fn recurse<'a>(&'a self, f: &mut dyn FnMut(&'a Type)) {
        match self {
            SymInt::Literal(_) | SymInt::Int => {}
            SymInt::Symbolic(ty) => f(ty),
            SymInt::Add(left, right)
            | SymInt::Sub(left, right)
            | SymInt::Mul(left, right)
            | SymInt::FloorDiv(left, right)
            | SymInt::Pow(left, right) => {
                left.recurse(f);
                right.recurse(f);
            }
        }
    }
}

impl pyrefly_util::visit::VisitMut<Type> for SymInt {
    fn recurse_mut(&mut self, f: &mut dyn FnMut(&mut Type)) {
        match self {
            SymInt::Literal(_) | SymInt::Int => {}
            SymInt::Symbolic(ty) => f(ty),
            SymInt::Add(left, right)
            | SymInt::Sub(left, right)
            | SymInt::Mul(left, right)
            | SymInt::FloorDiv(left, right)
            | SymInt::Pow(left, right) => {
                left.recurse_mut(f);
                right.recurse_mut(f);
            }
        }
    }
}

impl TypeEq for SymInt {}

// ============================================================================
// Shape Errors
// ============================================================================

/// Errors that can occur during shape/dimension checking
#[derive(Debug, Clone)]
pub enum ShapeError {
    /// Tensor ranks don't match
    RankMismatch { got: usize, want: usize },

    /// Invalid dimension value (e.g., negative or zero).
    /// `value` is the offending dimension index; `reason` explains why it's invalid.
    InvalidDimension { value: i64, reason: String },

    /// General shape computation error from a meta-shape function or broadcasting.
    /// The message is self-contained (no "Invalid dimension value N:" prefix).
    ShapeComputation { message: String },

    /// Structural mismatch between dimension types
    StructuralMismatch {
        got: String,
        got_canonical: String,
        want: String,
        want_canonical: String,
    },

    /// Type variable in nested position cannot be inferred
    /// For example: passing Dim[(A * B) // 2] to parameter Dim[X // 2]
    /// X appears in a nested position (inside // 2) and cannot be inferred
    NestedTypeVarNotInferred,

    /// Cannot index a scalar (rank-0) tensor
    ScalarIndex,

    /// Too many indices for tensor rank
    TooManyIndices { got: usize, max: usize },

    /// Operation not supported on variadic shapes.
    /// Triggers fixture fallback instead of a user-visible error.
    Unsupported { message: String },
}

impl Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RankMismatch { got, want } => {
                write!(
                    f,
                    "Tensor rank mismatch: expected {} dimensions, got {} dimensions",
                    want, got
                )
            }
            Self::InvalidDimension { value, reason } => {
                write!(f, "Invalid dimension value {}: {}", value, reason)
            }
            Self::ShapeComputation { message } => {
                write!(f, "{}", message)
            }
            Self::StructuralMismatch {
                got: _,
                got_canonical,
                want: _,
                want_canonical,
            } => {
                write!(
                    f,
                    "Size mismatch: expected {}, got {}",
                    want_canonical, got_canonical
                )
            }
            Self::NestedTypeVarNotInferred => {
                write!(f, "Type variable cannot be inferred from a nested position")
            }
            Self::ScalarIndex => {
                write!(f, "Cannot index scalar tensor (rank 0)")
            }
            Self::TooManyIndices { got, max } => {
                write!(
                    f,
                    "Too many indices for tensor: got {}, expected at most {}",
                    got, max
                )
            }
            Self::Unsupported { message } => {
                write!(f, "Unsupported: {}", message)
            }
        }
    }
}

impl ShapeError {
    pub fn rank_mismatch(got: usize, want: usize) -> Self {
        Self::RankMismatch { got, want }
    }

    pub fn invalid_dimension(value: i64, reason: impl Into<String>) -> Self {
        Self::InvalidDimension {
            value,
            reason: reason.into(),
        }
    }

    pub fn structural_mismatch(
        got: impl Into<String>,
        got_canonical: impl Into<String>,
        want: impl Into<String>,
        want_canonical: impl Into<String>,
    ) -> Self {
        Self::StructuralMismatch {
            got: got.into(),
            got_canonical: got_canonical.into(),
            want: want.into(),
            want_canonical: want_canonical.into(),
        }
    }

    pub fn nested_type_var_not_inferred() -> Self {
        Self::NestedTypeVarNotInferred
    }
}

/// Check if a dimension type contains a solver Var anywhere in its structure.
/// This is used to detect when a type variable in a nested position cannot be inferred.
pub fn contains_var_in_type(ty: &Type) -> bool {
    match ty {
        Type::Var(_) => true,
        Type::SymInt(dim) => contains_var_in_symint(dim, false),
        _ => false,
    }
}

fn contains_var_in_symint(dim: &SymInt, nested: bool) -> bool {
    match dim {
        SymInt::Symbolic(ty) => contains_var_in_symbolic_type(ty, nested),
        SymInt::Add(left, right)
        | SymInt::Sub(left, right)
        | SymInt::Mul(left, right)
        | SymInt::FloorDiv(left, right)
        | SymInt::Pow(left, right) => {
            contains_var_in_symint(left, true) || contains_var_in_symint(right, true)
        }
        _ => false,
    }
}

fn contains_var_in_symbolic_type(ty: &Type, nested: bool) -> bool {
    match ty {
        Type::Var(_) => nested,
        Type::SymInt(dim) => contains_var_in_symint(dim, true),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use ruff_python_ast::Int;

    use super::*;
    use crate::lit_int::LitInt;
    use crate::literal::Lit;
    use crate::literal::LitStyle;
    use crate::literal::Literal;
    use crate::types::AnyStyle;
    use crate::types::Var;

    fn symint_literal(n: i64) -> Type {
        Type::SymInt(SymInt::Literal(n))
    }

    #[test]
    fn canonicalize_preserves_any_in_symint() {
        let error_any = Type::SymInt(SymInt::Symbolic(Box::new(Type::Any(AnyStyle::Error))));
        assert_eq!(
            canonicalize(error_any),
            Type::SymInt(SymInt::Symbolic(Box::new(Type::Any(AnyStyle::Error))))
        );

        let gradual_symint = Type::SymInt(SymInt::add(gradual_size(), symint_literal(1)));
        assert_eq!(canonicalize(gradual_symint), gradual_size());
    }

    #[test]
    fn contains_var_detects_nested_var_inside_symbolic_symint() {
        let top_level_symbolic_var = Type::SymInt(SymInt::Symbolic(Box::new(Type::Var(Var::ZERO))));
        assert!(!contains_var_in_type(&top_level_symbolic_var));

        let nested_var = Type::SymInt(SymInt::Symbolic(Box::new(Type::SymInt(SymInt::add(
            Type::Var(Var::ZERO),
            symint_literal(1),
        )))));
        assert!(contains_var_in_type(&nested_var));

        let nested_symbolic_var = Type::SymInt(SymInt::Symbolic(Box::new(Type::SymInt(
            SymInt::Symbolic(Box::new(Type::Var(Var::ZERO))),
        ))));
        assert!(contains_var_in_type(&nested_symbolic_var));
    }

    #[test]
    fn canonicalize_symbolic_overflow_int_literal_stays_symbolic() {
        let overflow_lit = Type::Literal(Box::new(Literal {
            value: Lit::Int(LitInt::from_ast(&Int::from(i64::MAX as u64 + 1))),
            style: LitStyle::Explicit,
        }));
        let symbolic_overflow = Type::SymInt(SymInt::Symbolic(Box::new(overflow_lit.clone())));
        assert_eq!(
            canonicalize(symbolic_overflow),
            Type::SymInt(SymInt::Symbolic(Box::new(overflow_lit)))
        );
    }

    #[test]
    fn type_is_gradual_matches_canonicalized_gradual_check() {
        use crate::class::ClassType;
        use crate::display::tests::fake_class;
        use crate::types::TArgs;

        let int_class = Type::ClassType(ClassType::new(
            fake_class("int", "builtins", 0),
            TArgs::default(),
        ));
        let str_class = Type::ClassType(ClassType::new(
            fake_class("str", "builtins", 0),
            TArgs::default(),
        ));

        // Each case must satisfy `type_is_gradual(x) == is_gradual_size(&canonicalize(x))`.
        let cases = vec![
            // A symbolic `int` leaf canonicalizes to the gradual size (regression case).
            Type::SymInt(SymInt::Symbolic(Box::new(int_class.clone()))),
            // Arithmetic over a symbolic `int` leaf is gradual too.
            Type::SymInt(SymInt::add(
                Type::SymInt(SymInt::Symbolic(Box::new(int_class))),
                symint_literal(1),
            )),
            // The bare gradual size.
            gradual_size(),
            // A concrete literal is not gradual.
            symint_literal(5),
            // A non-`int` symbolic leaf is not gradual.
            Type::SymInt(SymInt::Symbolic(Box::new(str_class))),
        ];
        for case in cases {
            assert_eq!(
                type_is_gradual(&case),
                is_gradual_size(&canonicalize(case.clone())),
                "mismatch for {case:?}"
            );
        }
    }

    #[test]
    #[should_panic(
        expected = "only integer literals can be converted into symbolic size expressions"
    )]
    fn canonicalize_symbolic_non_int_literal_panics() {
        let bool_literal = Type::Literal(Box::new(Literal {
            value: Lit::Bool(true),
            style: LitStyle::Explicit,
        }));
        let invalid_symint = Type::SymInt(SymInt::Symbolic(Box::new(bool_literal)));
        let _ = canonicalize(invalid_symint);
    }
}
