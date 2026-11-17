/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::env;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

use sha2::Digest;

fn get_typeshed_input_path() -> PathBuf {
    match env::var_os("TYPESHED_ROOT") {
        Some(root) => {
            // When building with Buck, typeshed filegroup dir is passed in using TYPESHED_ROOT
            Path::new(&root).join("typeshed")
        }
        None => {
            // When building with Cargo, we could locate typeshed directly using relative dir
            PathBuf::from("third_party/typeshed")
        }
    }
}

fn get_stubs_input_path() -> PathBuf {
    match env::var_os("STUBS_ROOT") {
        Some(root) => {
            // When building with Buck, stubs filegroup dir is passed in using STUBS_ROOT
            Path::new(&root).join("stubs")
        }
        None => {
            // When building with Cargo, we could locate stubs directly using relative dir
            PathBuf::from("third_party/stubs")
        }
    }
}

fn get_output_path() -> Result<PathBuf, std::env::VarError> {
    // When building with Buck, output artifact directory is specified using this env var
    match env::var_os("OUT") {
        Some(path) => Ok(Path::new(&path).to_path_buf()),
        None => {
            // When building with Cargo, this env var is the containing directory of the artifact
            let out_dir = env::var("OUT_DIR")?;
            Ok(Path::new(&out_dir).to_path_buf())
        }
    }
}

/// Appends every file under `dir` to `tar`, as `archive_dir/<relative path>`, in sorted path
/// order: paths compare component by component, as `Path` does, so `foo/bar.pyi` comes before
/// `foo.pyi`. The order is part of the output: it decides which of two equally good names
/// Pyrefly suggests for a misspelled import, so it must not depend on the order in which the
/// file system lists directories.
fn append_files_sorted<W: Write>(
    tar: &mut tar::Builder<W>,
    dir: &Path,
    archive_dir: &Path,
) -> Result<(), std::io::Error> {
    let mut entries = std::fs::read_dir(dir)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        let archive_path = archive_dir.join(entry.file_name());
        if std::fs::metadata(&path)?.is_dir() {
            append_files_sorted(tar, &path, &archive_path)?;
        } else {
            tar.append_path_with_name(&path, &archive_path)?;
        }
    }
    Ok(())
}

/// Creates a compressed tar archive from the given input path and writes it to the output path.
/// Also computes and writes a SHA256 digest of the archive, and an index listing the path of
/// every file in the archive, one per line and in archive order, so that a runtime lookup can
/// tell which files exist without decompressing the archive.
///
/// `input_path` of `None` writes a valid but empty archive.
fn create_archive(
    input_path: Option<&Path>,
    archive_root: &str,
    output_path: &Path,
    digest_name: &str,
    index_name: &str,
) -> Result<(), std::io::Error> {
    let digest_path = output_path.with_file_name(digest_name);
    let index_path = output_path.with_file_name(index_name);

    // Create the tar.zst archive
    let mut archive_bytes = Vec::new();
    let encoder = zstd::stream::write::Encoder::new(&mut archive_bytes, 0)?;
    let mut tar = tar::Builder::new(encoder);

    if let Some(input_path) = input_path {
        if !input_path.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Input path does not exist: {}", input_path.display()),
            ));
        }
        append_files_sorted(&mut tar, input_path, Path::new(archive_root))?;
    }

    let encoder = tar.into_inner()?;
    encoder.finish()?;

    let hash = sha2::Sha256::digest(&archive_bytes);

    let mut index = String::new();
    for entry in
        tar::Archive::new(zstd::stream::read::Decoder::new(archive_bytes.as_slice())?).entries()?
    {
        let entry = entry?;
        if entry.header().entry_type().is_dir() {
            continue;
        }
        let path = entry.path()?;
        let path = path.to_str().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Archive path is not UTF-8: {}", path.display()),
            )
        })?;
        index.push_str(path);
        index.push('\n');
    }

    std::fs::write(output_path, &archive_bytes)?;
    std::fs::write(&digest_path, hash)?;
    std::fs::write(&index_path, index)?;

    Ok(())
}

fn main() -> Result<(), std::io::Error> {
    // Only watch for metadata changes to avoid having Cargo repeatedly crawling for
    // changes in the entire typeshed dir.
    println!("cargo::rerun-if-changed=third_party/typeshed_metadata.json");
    println!("cargo::rerun-if-env-changed=CARGO_FEATURE_THIRD_PARTY_STUBS");
    println!("cargo::rerun-if-changed=third_party/typeshed/stdlib/VERSIONS");

    let output_dir = get_output_path().unwrap();

    // Archived empty rather than `cfg`-ed away, so every consumer keeps compiling and
    // simply resolves no third-party imports.
    let third_party = env::var_os("CARGO_FEATURE_THIRD_PARTY_STUBS").is_some();

    // Create separate archives so each runtime bundle only decodes its own files.
    let typeshed_input = get_typeshed_input_path();
    create_archive(
        Some(&typeshed_input.join("stdlib")),
        "stdlib",
        &output_dir.join("stdlib.tar.zst"),
        "stdlib.sha256",
        "stdlib.index",
    )?;

    // Copied out of the archive rather than read back from it: recovering one member of a
    // zstd tar means decompressing the whole stream, which costs as much as extracting
    // every stub, and every Pyrefly process needs this file to resolve stdlib imports.
    std::fs::copy(
        typeshed_input.join("stdlib").join("VERSIONS"),
        output_dir.join("stdlib_versions"),
    )?;
    create_archive(
        third_party.then(|| typeshed_input.join("stubs")).as_deref(),
        "stubs",
        &output_dir.join("typeshed_stubs.tar.zst"),
        "typeshed_stubs.sha256",
        "typeshed_stubs.index",
    )?;

    // Create third-party stubs archive (non-typeshed stubs)
    let stubs_output = output_dir.join("stubs.tar.zst");
    create_archive(
        third_party.then(get_stubs_input_path).as_deref(),
        "",
        &stubs_output,
        "stubs.sha256",
        "stubs.index",
    )?;

    Ok(())
}
