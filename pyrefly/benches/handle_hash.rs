/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Compares hashing `Handle` with and without its importer site-package path,
//! through the map types that `State` keys by `Handle`.

use std::collections::HashMap;
use std::hash::Hash;
use std::hash::Hasher;
use std::hint::black_box;
use std::path::PathBuf;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use fxhash::FxHashMap;
use pyrefly_build::handle::Handle;
use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::sys_info::SysInfo;
use pyrefly_util::locked_map::LockedMap;
use starlark_map::small_map::SmallMap;

/// `Handle` hashed on every field that equality compares, as `#[derive(Hash)]` would.
#[derive(Clone, PartialEq, Eq)]
struct AllFields(Handle);

impl Hash for AllFields {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
        self.0.importer_site_package_path().hash(state);
    }
}

const N: usize = 20_000;

fn handles() -> Vec<Handle> {
    let sys_info = SysInfo::default();
    (0..N)
        .map(|i| {
            let name = format!("pkg{}.sub{}.mod{}", i % 37, i % 211, i);
            let path = PathBuf::from(format!(
                "/data/users/me/project/pkg{}/sub{}/mod{}.py",
                i % 37,
                i % 211,
                i
            ));
            Handle::new(
                ModuleName::from_str(&name),
                ModulePath::filesystem(path),
                sys_info,
            )
        })
        .collect()
}

fn bench_keys<K: Clone + Eq + Hash + Send + Sync + 'static>(
    c: &mut Criterion,
    label: &str,
    keys: Vec<K>,
) {
    let mut group = c.benchmark_group("handle_hash");
    let std_map: HashMap<K, usize> = keys.iter().cloned().zip(0..).collect();
    let fx_map: FxHashMap<K, usize> = keys.iter().cloned().zip(0..).collect();
    let small_map: SmallMap<K, usize> = keys.iter().cloned().zip(0..).collect();
    let locked_map: LockedMap<K, usize> = LockedMap::new();
    for (i, k) in keys.iter().enumerate() {
        locked_map.insert(k.clone(), i);
    }
    group.bench_function(BenchmarkId::new("std_hashmap_get", label), |b| {
        b.iter(|| keys.iter().map(|k| std_map[black_box(k)]).sum::<usize>())
    });
    group.bench_function(BenchmarkId::new("fx_hashmap_get", label), |b| {
        b.iter(|| keys.iter().map(|k| fx_map[black_box(k)]).sum::<usize>())
    });
    group.bench_function(BenchmarkId::new("small_map_get", label), |b| {
        b.iter(|| {
            keys.iter()
                .map(|k| *small_map.get(black_box(k)).unwrap())
                .sum::<usize>()
        })
    });
    group.bench_function(BenchmarkId::new("locked_map_get", label), |b| {
        b.iter(|| {
            keys.iter()
                .map(|k| *locked_map.get(black_box(k)).unwrap())
                .sum::<usize>()
        })
    });
    group.finish();
}

fn handle_hash(c: &mut Criterion) {
    let hs = handles();
    bench_keys(c, "three_fields", hs.clone());
    bench_keys(c, "all_fields", hs.into_iter().map(AllFields).collect());
}

criterion_group!(benches, handle_hash);
criterion_main!(benches);
