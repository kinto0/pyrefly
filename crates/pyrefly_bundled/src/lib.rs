/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::io::Read;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context as _;
use starlark_map::small_map::SmallMap;
use starlark_map::small_set::SmallSet;
use tar::Archive;
use zstd::stream::read::Decoder;

static BUNDLED_TYPESHED_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/stdlib.tar.zst"));

/// The paths of the files in the stdlib archive, one per line; see this crate's `build.rs`.
static BUNDLED_TYPESHED_INDEX: &str = include_str!(concat!(env!("OUT_DIR"), "/stdlib.index"));

pub const BUNDLED_TYPESHED_DIGEST: &[u8; 32] =
    include_bytes!(concat!(env!("OUT_DIR"), "/stdlib.sha256"));

/// `stdlib/VERSIONS`, embedded uncompressed so it can be read without touching the archive.
static BUNDLED_TYPESHED_VERSIONS: &str = include_str!(concat!(env!("OUT_DIR"), "/stdlib_versions"));

/// Empty unless the `third-party-stubs` feature is on; see this crate's `build.rs`.
static BUNDLED_TYPESHED_THIRD_PARTY_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/typeshed_stubs.tar.zst"));

static BUNDLED_TYPESHED_THIRD_PARTY_INDEX: &str =
    include_str!(concat!(env!("OUT_DIR"), "/typeshed_stubs.index"));

pub const BUNDLED_TYPESHED_THIRD_PARTY_DIGEST: &[u8; 32] =
    include_bytes!(concat!(env!("OUT_DIR"), "/typeshed_stubs.sha256"));

/// Empty unless the `third-party-stubs` feature is on; see this crate's `build.rs`.
static BUNDLED_THIRD_PARTY_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/stubs.tar.zst"));

static BUNDLED_THIRD_PARTY_INDEX: &str = include_str!(concat!(env!("OUT_DIR"), "/stubs.index"));

pub const BUNDLED_THIRD_PARTY_DIGEST: &[u8; 32] =
    include_bytes!(concat!(env!("OUT_DIR"), "/stubs.sha256"));

#[derive(Clone, Copy)]
enum PathFilter {
    /// Filter for stdlib files (typeshed/stdlib/...)
    Stdlib,
    /// Filter for third-party stubs (typeshed/stubs/package-name/...)
    ThirdPartyTypeshedStubs,
    /// Filter for third-party stubs not from typeshed (stubs/package-name/...)
    ThirdPartyStubs,
}

impl PathFilter {
    fn expected_first_component(&self) -> Option<&str> {
        match self {
            PathFilter::Stdlib => Some("stdlib"),
            PathFilter::ThirdPartyTypeshedStubs => Some("stubs"),
            PathFilter::ThirdPartyStubs => None,
        }
    }

    fn archive_bytes(&self) -> &'static [u8] {
        match self {
            PathFilter::Stdlib => BUNDLED_TYPESHED_BYTES,
            PathFilter::ThirdPartyTypeshedStubs => BUNDLED_TYPESHED_THIRD_PARTY_BYTES,
            PathFilter::ThirdPartyStubs => BUNDLED_THIRD_PARTY_BYTES,
        }
    }

    fn archive_index(&self) -> &'static str {
        match self {
            PathFilter::Stdlib => BUNDLED_TYPESHED_INDEX,
            PathFilter::ThirdPartyTypeshedStubs => BUNDLED_TYPESHED_THIRD_PARTY_INDEX,
            PathFilter::ThirdPartyStubs => BUNDLED_THIRD_PARTY_INDEX,
        }
    }

    /// Map the path of a file in the archive to the path it is imported from, and for
    /// typeshed third-party stubs the package that provides it. Returns `None` for a file
    /// that is not a `.pyi` file of this archive.
    fn relative_path(&self, archive_path: &Path) -> Option<(PathBuf, Option<String>)> {
        let mut components = archive_path.components();
        let first_component = components.next();
        if let Some(expected) = self.expected_first_component()
            && first_component.is_none_or(|component| component.as_os_str() != expected)
        {
            return None;
        }

        // Typeshed stdlib stubs have a different directory structure than third-party stubs.
        // stdlib example: typeshed/stdlib/builtins.pyi
        // third-party example: typeshed/stubs/package-name/package-name/other.pyi
        // An example of this might be the path typeshed/stubs/JACK-Client/jack/other.pyi
        // In this case we want out path to be jack/other.pyi, so we skip over "stubs" and "JACK-Client".
        let (relative_path, package) = match self {
            PathFilter::Stdlib => (components.collect::<PathBuf>(), None),
            PathFilter::ThirdPartyTypeshedStubs => {
                // The package is recorded per path (e.g., "google/protobuf/__init__.pyi" →
                // "protobuf") to handle namespace packages like google/ that span multiple
                // typeshed packages (protobuf, google-cloud-ndb, etc.).
                let package = components
                    .next()
                    .and_then(|package| package.as_os_str().to_str())
                    .map(str::to_owned);
                (components.collect(), package)
            }
            PathFilter::ThirdPartyStubs => {
                // Include first_component (e.g., "conans-stubs") in the path
                let mut path = PathBuf::from(first_component?.as_os_str());
                path.extend(components);
                (path, None)
            }
        };
        relative_path
            .extension()
            .is_some_and(|ext| ext == "pyi")
            .then_some((relative_path, package))
    }
}

/// The `.pyi` files of a bundled archive, read from its uncompressed index.
pub struct BundledPaths {
    /// The path each file is imported from, in archive order. When several files of the
    /// archive map to the same path, only the first is listed, matching the contents.
    pub paths: Vec<PathBuf>,
    /// For typeshed third-party stubs, the package that first provides each path.
    pub path_to_package: SmallMap<PathBuf, String>,
}

fn archive_paths(filter: PathFilter) -> BundledPaths {
    let mut paths = SmallSet::new();
    let mut path_to_package = SmallMap::new();
    for (relative_path, package) in filter
        .archive_index()
        .lines()
        .filter_map(|line| filter.relative_path(Path::new(line)))
    {
        if let Some(package) = package {
            path_to_package
                .entry(relative_path.clone())
                .or_insert(package);
        }
        paths.insert(relative_path);
    }
    BundledPaths {
        paths: paths.into_iter().collect(),
        path_to_package,
    }
}

/// Decompress an archive and read the contents of its `.pyi` files, keyed by the path they
/// are imported from. This decompresses the whole archive, so prefer [`archive_paths`] when
/// the contents are not needed.
fn extract_files_from_archive(filter: PathFilter) -> anyhow::Result<SmallMap<PathBuf, String>> {
    let decoder = Decoder::new(filter.archive_bytes())?;
    let mut archive = Archive::new(decoder);
    let entries = archive
        .entries()
        .context("Cannot query all entries in archive")?;

    let mut items = SmallMap::new();
    for maybe_entry in entries {
        let mut entry = maybe_entry.context("Cannot read individual entry in typeshed archive")?;
        if entry.header().entry_type().is_dir() {
            continue;
        }
        let archive_path = entry
            .path()
            .context("Cannot extract path from archive entry")?;
        let Some((relative_path, _)) = filter.relative_path(&archive_path) else {
            continue;
        };
        if items.contains_key(&relative_path) {
            continue;
        }
        let mut contents = String::with_capacity(entry.size() as usize);
        entry
            .read_to_string(&mut contents)
            .context("Cannot read content of archive entry")?;
        items.insert(relative_path, contents);
    }
    Ok(items)
}

/// The paths of the bundled typeshed stdlib stubs, without decompressing them.
pub fn bundled_typeshed_paths() -> Vec<PathBuf> {
    archive_paths(PathFilter::Stdlib).paths
}

pub fn bundled_typeshed() -> anyhow::Result<SmallMap<PathBuf, String>> {
    extract_files_from_archive(PathFilter::Stdlib)
}

/// The contents of typeshed's `stdlib/VERSIONS`, which records the Python versions each
/// stdlib module is available on.
pub fn bundled_typeshed_versions() -> &'static str {
    BUNDLED_TYPESHED_VERSIONS
}

/// The paths of the bundled typeshed third-party stubs, without decompressing them.
pub fn bundled_third_party_stubs_paths() -> BundledPaths {
    archive_paths(PathFilter::ThirdPartyTypeshedStubs)
}

pub fn bundled_third_party_stubs() -> anyhow::Result<SmallMap<PathBuf, String>> {
    extract_files_from_archive(PathFilter::ThirdPartyTypeshedStubs)
}

/// The paths of the bundled non-typeshed third-party stubs, without decompressing them.
pub fn bundled_third_party_paths() -> Vec<PathBuf> {
    archive_paths(PathFilter::ThirdPartyStubs).paths
}

/// Extract third-party stubs from the bundled archive.
/// These are stubs that are not included in typeshed (e.g., pandas-stubs, boto3-stubs).
pub fn bundled_third_party() -> anyhow::Result<SmallMap<PathBuf, String>> {
    extract_files_from_archive(PathFilter::ThirdPartyStubs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bundled_typeshed_returns_stdlib_files() {
        let result = bundled_typeshed();
        assert!(result.is_ok(), "bundled_typeshed should succeed");

        let files = result.unwrap();
        assert!(!files.is_empty(), "Should contain stdlib .pyi files");

        // Verify all returned paths are .pyi files
        for (path, _) in files.iter() {
            assert_eq!(
                path.extension().and_then(|ext| ext.to_str()),
                Some("pyi"),
                "All files should have .pyi extension, found: {:?}",
                path
            );
        }
    }

    #[test]
    fn test_bundled_typeshed_returns_third_party_files() {
        let result = bundled_third_party_stubs();
        assert!(result.is_ok(), "bundled_typeshed should succeed");

        let files = result.unwrap();
        assert!(!files.is_empty(), "Should contain .pyi files");

        // Verify all returned paths are .pyi files
        for (path, _) in files.iter() {
            assert_eq!(
                path.extension().and_then(|ext| ext.to_str()),
                Some("pyi"),
                "All files should have .pyi extension, found: {:?}",
                path
            );
        }
    }

    #[test]
    fn test_bundled_typeshed_paths_are_relative() {
        let result = bundled_typeshed().unwrap();

        // Verify paths don't start with "stdlib" (it should be stripped)
        for (path, _) in result.iter() {
            let first_component = path.components().next();
            assert_ne!(
                first_component.and_then(|c| c.as_os_str().to_str()),
                Some("stdlib"),
                "Path should not start with 'stdlib', found: {:?}",
                path
            );
        }
    }

    #[test]
    fn test_bundled_typeshed_third_party_resolves_path() {
        let result = bundled_third_party_stubs().unwrap();

        // Verify paths don't start with "stubs" (it should be stripped)
        for (path, _) in result.iter() {
            let first_component = path.components().next();
            assert_ne!(
                first_component.and_then(|c| c.as_os_str().to_str()),
                Some("stubs"),
                "Path should not start with 'stubs', found: {:?}",
                path
            );
        }
    }

    #[test]
    fn test_bundled_typeshed_contains_common_modules() {
        let files = bundled_typeshed().unwrap();

        // Check that at least some .pyi files exist that match common module patterns
        let has_builtins = files.iter().any(|(path, _)| {
            path.to_str()
                .map(|s| s.contains("builtins"))
                .unwrap_or(false)
        });
        let has_sys = files
            .iter()
            .any(|(path, _)| path.to_str().map(|s| s.contains("sys")).unwrap_or(false));
        let has_os = files
            .iter()
            .any(|(path, _)| path.to_str().map(|s| s.contains("os")).unwrap_or(false));

        assert!(
            has_builtins || has_sys || has_os,
            "Should contain at least one common stdlib module (builtins, sys, or os)"
        );
    }

    #[test]
    fn test_bundled_typeshed_file_contents_not_empty() {
        let files = bundled_typeshed().unwrap();

        let non_empty_count = files
            .iter()
            .filter(|(_, content)| !content.is_empty())
            .count();
        assert!(
            non_empty_count > 0,
            "Should have at least some files with content"
        );
    }

    #[test]
    fn test_extract_pyi_files_from_archive_stdlib_filter() {
        let result = extract_files_from_archive(PathFilter::Stdlib);
        assert!(result.is_ok(), "Should successfully extract stdlib files");

        let files = result.unwrap();
        assert!(
            !files.is_empty(),
            "Should extract at least some stdlib files"
        );

        let unique_count = files.len();
        assert_eq!(files.len(), unique_count, "Should not have duplicate paths");
    }

    #[test]
    fn test_bundled_typeshed_versions_contains_removed_module_metadata() {
        let versions = bundled_typeshed_versions();
        assert!(
            versions.contains("distutils: 3.0-3.11"),
            "Bundled stdlib VERSIONS should include distutils availability"
        );
    }

    #[test]
    fn test_path_filter_expected_first_component() {
        assert_eq!(
            PathFilter::Stdlib.expected_first_component(),
            Some("stdlib"),
            "Stdlib filter should expect 'stdlib' component"
        );
        assert_eq!(
            PathFilter::ThirdPartyTypeshedStubs.expected_first_component(),
            Some("stubs"),
            "ThirdPartyStubs filter should expect 'stubs' component"
        );
    }

    #[test]
    fn test_bundled_typeshed_contains_valid_python_stubs() {
        let files = bundled_typeshed().unwrap();

        // Check that at least some files contain Python stub signatures
        let has_python_stub_content = files.iter().any(|(_, content)| {
            content.contains("def ") || content.contains("class ") || content.contains("import ")
        });

        assert!(
            has_python_stub_content,
            "At least some files should contain Python stub content"
        );
    }

    #[test]
    fn test_no_non_pyi_files_included() {
        let stdlib_files = bundled_typeshed().unwrap();
        let third_party_files = bundled_third_party_stubs().unwrap();

        for (path, _) in stdlib_files.iter().chain(third_party_files.iter()) {
            let ext = path.extension().and_then(|e| e.to_str());
            assert_eq!(
                ext,
                Some("pyi"),
                "Should only include .pyi files, found extension: {:?} in path: {:?}",
                ext,
                path
            );
        }
    }

    #[test]
    fn test_typeshed_archive_is_valid_zstd_compressed_tar() {
        let decoder = Decoder::new(BUNDLED_TYPESHED_BYTES);
        assert!(
            decoder.is_ok(),
            "Typeshed archive should be valid zstd compressed data"
        );
        let mut archive = Archive::new(decoder.unwrap());
        let entries = archive.entries();
        assert!(
            entries.is_ok(),
            "Typeshed archive should contain valid tar entries"
        );
    }

    #[test]
    fn test_stubs_archive_is_valid_zstd_compressed_tar() {
        let decoder = Decoder::new(BUNDLED_THIRD_PARTY_BYTES);
        assert!(
            decoder.is_ok(),
            "Stubs archive should be valid zstd compressed data"
        );
        let mut archive = Archive::new(decoder.unwrap());
        let entries = archive.entries();
        assert!(
            entries.is_ok(),
            "Stubs archive should contain valid tar entries"
        );
    }

    #[test]
    fn test_index_lists_the_extracted_files() {
        for filter in [
            PathFilter::Stdlib,
            PathFilter::ThirdPartyTypeshedStubs,
            PathFilter::ThirdPartyStubs,
        ] {
            let files = extract_files_from_archive(filter).unwrap();
            let extracted = files.keys().cloned().collect::<Vec<_>>();
            assert_eq!(
                archive_paths(filter).paths,
                extracted,
                "The index should list exactly the extracted files, in the same order"
            );
        }
    }

    #[test]
    fn test_index_records_third_party_packages() {
        let index = bundled_third_party_stubs_paths();
        assert!(!index.paths.is_empty(), "Should list third-party stubs");
        assert!(
            index
                .paths
                .iter()
                .all(|path| index.path_to_package.contains_key(path)),
            "Every third-party typeshed stub should have a package"
        );
    }

    #[test]
    fn test_archives_are_in_sorted_path_order() {
        for bytes in [
            BUNDLED_TYPESHED_BYTES,
            BUNDLED_TYPESHED_THIRD_PARTY_BYTES,
            BUNDLED_THIRD_PARTY_BYTES,
        ] {
            let mut archive = Archive::new(Decoder::new(bytes).unwrap());
            let paths = archive
                .entries()
                .unwrap()
                .map(|entry| entry.unwrap().path().unwrap().into_owned())
                .collect::<Vec<_>>();
            assert!(
                paths.is_sorted(),
                "Archive files should be in sorted path order, independent of the file system"
            );
        }
    }
}
