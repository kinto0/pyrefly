/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Benchmarks for the pyrefly type checker.
//!
//! Each benchmark drives the full parse -> bind -> type-check pipeline through
//! the public `Playground` API, feeding in representative Python source and
//! collecting the resulting diagnostics. This mirrors the work performed when
//! checking real Python code.

use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use pyrefly::playground::Playground;
use starlark_map::small_map::SmallMap;

/// Type-check a set of in-memory modules and collect the diagnostics.
///
/// A fresh `Playground` is created for every call so each measurement reflects
/// a cold check of the provided source rather than reusing cached state.
fn check(files: &[(&str, &str)]) -> usize {
    let mut playground = Playground::new(Some("3.12")).expect("failed to create playground");
    let mut sources = SmallMap::new();
    for (name, code) in files {
        sources.insert((*name).to_owned(), (*code).to_owned());
    }
    playground.update_sandbox_files(sources, true);
    playground.get_errors().len()
}

/// A small, idiomatic module exercising common typing features.
const SIMPLE: &str = r#"
from typing import Optional


class Point:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def translate(self, dx: int, dy: int) -> "Point":
        return Point(self.x + dx, self.y + dy)


def closest(points: list[Point], origin: Optional[Point] = None) -> Point:
    base = origin if origin is not None else Point(0, 0)
    best = points[0]
    best_dist = (best.x - base.x) ** 2 + (best.y - base.y) ** 2
    for p in points[1:]:
        dist = (p.x - base.x) ** 2 + (p.y - base.y) ** 2
        if dist < best_dist:
            best = p
            best_dist = dist
    return best
"#;

/// A workload that historically caused quadratic behaviour: a deeply nested
/// dictionary literal whose values all require type inference.
const NESTED_DICT: &str = r#"
from typing import TypeVar

_T = TypeVar("_T")


def table() -> _T: ...


class Configs:
    def __init__(self) -> None:
        self.value = {
            "1": {
                "a": table(),
                "b": table(),
                "c": table(),
                "d": table(),
                "e": table(),
                "f": table(),
                "g": table(),
                "h": table(),
                "i": table(),
                "j": table(),
            },
            "2": {
                "a": table(),
                "b": table(),
                "c": table(),
                "d": table(),
                "e": table(),
                "f": table(),
                "g": table(),
                "h": table(),
                "i": table(),
            },
            "3": {},
        }
"#;

/// Cross-protocol structural subtyping over a chain of protocols. Without the
/// subset cache this fans out into exponential work, making it a good guard
/// against regressions in protocol matching.
const PROTOCOL_CHAIN: &str = r#"
from typing import Protocol


class Array1(Protocol):
    def op_0(self, other: "Array1 | complex", /) -> "Array2": ...
    def op_1(self, other: "Array1 | complex", /) -> "Array2": ...
    def op_2(self, other: "Array1 | complex", /) -> "Array2": ...
    def op_3(self, other: "Array1 | complex", /) -> "Array2": ...
    def op_4(self, other: "Array1 | complex", /) -> "Array2": ...


class Array2(Protocol):
    def op_0(self, other: "Array2 | complex", /) -> "Array1": ...
    def op_1(self, other: "Array2 | complex", /) -> "Array1": ...
    def op_2(self, other: "Array2 | complex", /) -> "Array1": ...
    def op_3(self, other: "Array2 | complex", /) -> "Array1": ...
    def op_4(self, other: "Array2 | complex", /) -> "Array1": ...


class Impl:
    def op_0(self, other: object, /) -> "Impl": ...
    def op_1(self, other: object, /) -> "Impl": ...
    def op_2(self, other: object, /) -> "Impl": ...
    def op_3(self, other: object, /) -> "Impl": ...
    def op_4(self, other: object, /) -> "Impl": ...


def f1(x: Array1 | complex) -> Array1: ...
def f2(x: Array2 | complex) -> Array2: ...


def test() -> None:
    val = Impl()
    a1 = f1(val)
    a2 = f2(val)
    c1 = f1(a2)
    c2 = f2(a1)
"#;

/// Two modules so the checker resolves an import across files.
const CROSS_FILE_MAIN: &str = r#"
from utils import compute, Record


def run(values: list[int]) -> Record:
    total = compute(values)
    return Record(name="total", value=total)
"#;

const CROSS_FILE_UTILS: &str = r#"
from dataclasses import dataclass


@dataclass
class Record:
    name: str
    value: int


def compute(values: list[int]) -> int:
    acc = 0
    for v in values:
        acc += v
    return acc
"#;

fn bench_type_check(c: &mut Criterion) {
    let mut group = c.benchmark_group("type_check");

    group.bench_function("simple_module", |b| {
        b.iter(|| check(&[("main.py", SIMPLE)]))
    });

    group.bench_function("nested_dict", |b| {
        b.iter(|| check(&[("main.py", NESTED_DICT)]))
    });

    group.bench_function("protocol_chain", |b| {
        b.iter(|| check(&[("main.py", PROTOCOL_CHAIN)]))
    });

    group.bench_function("cross_file_imports", |b| {
        b.iter(|| check(&[("main.py", CROSS_FILE_MAIN), ("utils.py", CROSS_FILE_UTILS)]))
    });

    group.finish();
}

criterion_group!(benches, bench_type_check);
criterion_main!(benches);
