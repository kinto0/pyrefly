/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashSet;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use anyhow::Context;
use anyhow::anyhow;
use pyrefly_python::sys_info::PythonPlatform;
use pyrefly_python::sys_info::PythonVersion;
use serde::Deserialize;
use serde_json::Value;
use tracing::warn;
use url::Url;

use crate::environment::environment::PythonEnvironment;

/// The script run by [`query`] with `python -c`. It prints a JSON object matching [`QueryOutput`].
///
/// `python -c` inserts `''` (the working directory) at the front of `sys.path` after `site` has
/// run and before the script starts, so removing that entry before the first import keeps files
/// in the working directory, usually the user's project, from shadowing the modules the script
/// imports and being executed. `sys` is built in, so importing it cannot load such a file.
/// Python 3.11+ omits the entry under `-P` or `PYTHONSAFEPATH`, which the check allows for.
///
/// `distribution_urls` holds the PEP 610 `direct_url.json` of every distribution installed in a
/// `sys.path` directory. It is collected by listing those directories rather than through
/// `importlib.metadata`, which finds the same files but takes most of the script's run time to
/// import.
///
/// `pyrefly_version` is the version of the first `pyrefly` distribution on `sys.path`, the one
/// `importlib.metadata.version('pyrefly')` returns, read from that distribution's metadata file.
const QUERY_SCRIPT: &str = "\
import sys
if sys.path and sys.path[0] == '':
    del sys.path[0]
import json, os, sysconfig
platform = sys.platform
v = sys.version_info
version = '{}.{}.{}'.format(v.major, v.minor, v.micro)
stdlib_paths = [p for p in [sysconfig.get_path('stdlib')] if p is not None]
site_package_path = [p for p in sys.path if p != '' and '.zip' not in p and not p.endswith('/lib-dynload') and p not in stdlib_paths]
distribution_urls = []
pyrefly_version = None
for path in sys.path:
    try:
        names = os.listdir(path)
    except OSError:
        continue
    for name in names:
        if name.lower().endswith(('.dist-info', '.egg-info')):
            if pyrefly_version is None and name.lower().startswith('pyrefly-'):
                metadata = 'METADATA' if name.lower().endswith('.dist-info') else 'PKG-INFO'
                try:
                    with open(os.path.join(path, name, metadata), encoding='utf-8') as f:
                        pyrefly_version = next((line[len('Version:'):].strip() for line in f if line.startswith('Version:')), None)
                except (OSError, ValueError):
                    pass
            try:
                with open(os.path.join(path, name, 'direct_url.json'), encoding='utf-8') as f:
                    distribution_url = json.load(f)
            except (OSError, ValueError):
                continue
            if isinstance(distribution_url, dict):
                distribution_urls.append(distribution_url)
print(json.dumps({'python_platform': platform, 'python_version': version, 'site_package_path': site_package_path, 'stdlib_paths': stdlib_paths, 'distribution_urls': distribution_urls, 'pyrefly_version': pyrefly_version}))
";

/// The JSON contract emitted by the interpreter-query script run in [`query`].
/// `distribution_urls` is left untyped because its per-entry shape is best-effort
/// PEP 610 metadata that [`editable_paths_from_query`] tolerates being partial.
#[derive(Deserialize)]
struct QueryOutput {
    python_platform: Option<PythonPlatform>,
    python_version: Option<PythonVersion>,
    site_package_path: Option<Vec<PathBuf>>,
    #[serde(default)]
    stdlib_paths: Vec<PathBuf>,
    #[serde(default)]
    distribution_urls: Value,
    #[serde(default)]
    pyrefly_version: Option<String>,
}

/// Given a path to a Python interpreter executable, query that interpreter for its
/// version, platform, and site package path. Return an error in the case of failure during
/// execution, parsing, or deserializing.
pub fn query(interpreter: &Path) -> anyhow::Result<PythonEnvironment> {
    if let Ok(pythonpath) = std::env::var("PYTHONPATH") {
        warn!(
            "PYTHONPATH environment variable is set to `{}`. Checks in other environments may not include these paths.",
            pythonpath
        );
    }

    let mut command = Command::new(interpreter);
    command.arg("-c");
    command.arg(QUERY_SCRIPT);

    let python_info = command.output()?;

    let stdout = String::from_utf8(python_info.stdout).with_context(|| {
        format!(
            "while parsing Python interpreter (`{}`) stdout for environment configuration",
            interpreter.display()
        )
    })?;
    if !python_info.status.success() {
        let stderr = String::from_utf8(python_info.stderr)
            .unwrap_or("<Failed to parse STDOUT from UTF-8 string>".to_owned());
        return Err(anyhow::anyhow!(
            "Unable to query interpreter {} for environment info:\nSTDOUT: {}\nSTDERR: {}",
            interpreter.display(),
            stdout,
            stderr
        ));
    }

    let query_output: QueryOutput = serde_json::from_str(&stdout)?;

    let python_platform = query_output.python_platform.ok_or_else(|| {
        anyhow!("Expected `python_platform` from Python interpreter query to be non-empty")
    })?;
    let python_version = query_output.python_version.ok_or_else(|| {
        anyhow!("Expected `python_version` from Python interpreter query to be non-empty")
    })?;
    let interpreter_site_package_path = query_output.site_package_path.ok_or_else(|| {
        anyhow!("Expected `site_package_path` from Python interpreter query to be non-empty")
    })?;
    let interpreter_editable_path = editable_paths_from_query(
        interpreter,
        &query_output.distribution_urls,
        &interpreter_site_package_path,
    )?;

    Ok(PythonEnvironment {
        python_platform: Some(python_platform),
        python_version: Some(python_version),
        // The queried value lives in `interpreter_site_package_path`; this is left
        // `Some(empty)` (rather than `None`) so `set_empty_to_default` treats it as
        // already resolved instead of re-applying Pyrefly's default.
        site_package_path: Some(Vec::new()),
        interpreter_site_package_path,
        interpreter_editable_path,
        interpreter_stdlib_path: query_output.stdlib_paths,
        interpreter_pyrefly_version: query_output.pyrefly_version,
    })
}

fn editable_paths_from_query(
    interpreter: &Path,
    distribution_urls: &Value,
    interpreter_paths: &[PathBuf],
) -> anyhow::Result<Vec<PathBuf>> {
    let distribution_urls = distribution_urls.as_array().ok_or_else(|| {
        anyhow!(
            "Expected `distribution_urls` from Python interpreter (`{}`) query to be an array",
            interpreter.display()
        )
    })?;
    let editable_roots: HashSet<PathBuf> = distribution_urls
        .iter()
        .filter(|distribution_url| {
            distribution_url
                .pointer("/dir_info/editable")
                .and_then(Value::as_bool)
                == Some(true)
        })
        .filter_map(|distribution_url| distribution_url.get("url").and_then(Value::as_str))
        .filter_map(|url| Url::parse(url).ok()?.to_file_path().ok())
        // The direct URL points at the project root, but a `src`-layout
        // package (e.g. a UV workspace member) puts importable code
        // under `<root>/src` instead, so both must be accepted.
        .flat_map(|root| [root.clone(), root.join("src")])
        .filter_map(|path| path.canonicalize().ok())
        .collect();
    Ok(interpreter_paths
        .iter()
        .filter(|path| {
            path.canonicalize()
                .is_ok_and(|canonical_path| editable_roots.contains(&canonical_path))
        })
        .cloned()
        .collect())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use serde_json::json;
    use tempfile::tempdir;
    use which::which;

    use super::*;

    /// A working Python interpreter on `PATH`. Tests that run the query return early when
    /// there is none, since the test environment is not guaranteed to provide one.
    fn python() -> Option<PathBuf> {
        ["python3", "python"]
            .into_iter()
            .filter_map(|name| which(name).ok())
            .find(|python| {
                Command::new(python)
                    .args(["-c", "import sys"])
                    .output()
                    .is_ok_and(|output| output.status.success())
            })
    }

    /// Run [`QUERY_SCRIPT`] with `python` in `cwd`, returning whether it succeeded and its stdout.
    fn run_query_script(python: &Path, cwd: &Path) -> (bool, String) {
        let output = Command::new(python)
            .arg("-c")
            .arg(QUERY_SCRIPT)
            .current_dir(cwd)
            .output()
            .unwrap();
        (
            output.status.success(),
            String::from_utf8(output.stdout).unwrap(),
        )
    }

    #[test]
    fn interpreter_query_output_format() {
        let Some(python) = python() else {
            eprintln!("Skipping: no Python interpreter on PATH");
            return;
        };
        let cwd = tempdir().unwrap();

        let (success, stdout) = run_query_script(&python, cwd.path());

        assert!(success);
        let output: Value = serde_json::from_str(&stdout).unwrap();
        let mut keys: Vec<&str> = output
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            vec![
                "distribution_urls",
                "pyrefly_version",
                "python_platform",
                "python_version",
                "site_package_path",
                "stdlib_paths",
            ]
        );
        assert!(
            output["distribution_urls"]
                .as_array()
                .unwrap()
                .iter()
                .all(Value::is_object)
        );
        let output: QueryOutput = serde_json::from_value(output).unwrap();
        assert!(output.python_platform.is_some());
        assert_eq!(output.python_version.unwrap().major, 3);
        assert!(!output.stdlib_paths.is_empty());
        let site_package_path = output.site_package_path.unwrap();
        assert!(
            output
                .stdlib_paths
                .iter()
                .all(|path| !site_package_path.contains(path)),
            "The standard library is not a site package path"
        );
    }

    #[test]
    fn interpreter_query_reports_venv_site_packages_and_editable_installs() {
        let Some(python) = python() else {
            eprintln!("Skipping: no Python interpreter on PATH");
            return;
        };
        let temp = tempdir().unwrap();
        let venv = temp.path().join("venv");
        let project = temp.path().join("project");
        fs::create_dir(&project).unwrap();
        assert!(
            Command::new(&python)
                .args(["-m", "venv", "--without-pip"])
                .arg(&venv)
                .status()
                .unwrap()
                .success()
        );
        let interpreter = if cfg!(windows) {
            venv.join("Scripts").join("python.exe")
        } else {
            venv.join("bin").join("python")
        };
        let canonical_venv = venv.canonicalize().unwrap();
        let site_packages = query(&interpreter)
            .unwrap()
            .interpreter_site_package_path
            .into_iter()
            .find(|path| {
                path.ends_with("site-packages")
                    && path
                        .canonicalize()
                        .is_ok_and(|path| path.starts_with(&canonical_venv))
            })
            .unwrap();
        // An editable install of `foo` from `project`, a regular install of `bar`, and
        // distributions whose `direct_url.json` is missing, malformed, or not an object.
        fs::write(
            site_packages.join("foo.pth"),
            format!("{}\n", project.display()),
        )
        .unwrap();
        for (distribution, direct_url) in [
            (
                "foo-1.0.dist-info",
                json!({
                    "url": Url::from_file_path(&project).unwrap().to_string(),
                    "dir_info": {"editable": true},
                }),
            ),
            (
                "bar-1.0.dist-info",
                json!({"url": "https://example.com/bar-1.0.tar.gz", "archive_info": {}}),
            ),
            ("baz-1.0.dist-info", json!("not an object")),
        ] {
            fs::create_dir(site_packages.join(distribution)).unwrap();
            fs::write(
                site_packages.join(distribution).join("direct_url.json"),
                direct_url.to_string(),
            )
            .unwrap();
        }
        fs::create_dir(site_packages.join("qux-1.0.dist-info")).unwrap();
        fs::create_dir(site_packages.join("quux-1.0.dist-info")).unwrap();
        fs::write(
            site_packages
                .join("quux-1.0.dist-info")
                .join("direct_url.json"),
            "{",
        )
        .unwrap();

        let env = query(&interpreter).unwrap();

        assert_eq!(env.python_version.unwrap().major, 3);
        assert!(env.interpreter_site_package_path.contains(&site_packages));
        assert!(env.interpreter_site_package_path.contains(&project));
        assert_eq!(env.interpreter_editable_path, vec![project]);
    }

    #[test]
    fn interpreter_query_ignores_modules_in_working_directory() {
        let Some(python) = python() else {
            eprintln!("Skipping: no Python interpreter on PATH");
            return;
        };
        let cwd = tempdir().unwrap();
        // The query script imports `json`, so a project file with that name must not be run.
        fs::write(
            cwd.path().join("json.py"),
            "open('json-was-imported', 'w').close()\n",
        )
        .unwrap();

        let (success, stdout) = run_query_script(&python, cwd.path());

        assert!(success);
        assert!(serde_json::from_str::<QueryOutput>(&stdout).is_ok());
        assert!(!cwd.path().join("json-was-imported").exists());
    }

    #[test]
    fn interpreter_query_classifies_editable_file_urls() {
        let temp = tempdir().unwrap();
        let project = temp.path().join("project");
        let source = project.join("src");
        let project_alias = source.join("..");
        let site_packages = temp.path().join("site-packages");
        let vendor = project.join("vendor");
        for path in [&source, &site_packages, &vendor] {
            std::fs::create_dir_all(path).unwrap();
        }
        let interpreter_paths = vec![
            project_alias.clone(),
            source.clone(),
            site_packages.clone(),
            vendor,
        ];
        let distribution_urls = json!([
            {
                "url": Url::from_file_path(&project).unwrap().to_string(),
                "dir_info": {"editable": true},
            },
            {
                "url": Url::from_file_path(&site_packages).unwrap().to_string(),
                "dir_info": {"editable": false},
            },
            {
                "url": "https://example.com/vendor",
                "dir_info": {"editable": true},
            },
            {
                "url": "not a URL",
                "dir_info": {"editable": true},
            },
            {"dir_info": {"editable": true}},
            {"url": Url::from_file_path(&project).unwrap().to_string()},
        ]);

        let paths =
            editable_paths_from_query(Path::new("python"), &distribution_urls, &interpreter_paths)
                .unwrap();

        assert_eq!(paths, vec![project_alias, source]);
    }

    #[test]
    fn interpreter_query_rejects_missing_distribution_urls() {
        let message = editable_paths_from_query(Path::new("python"), &Value::Null, &[])
            .unwrap_err()
            .to_string();

        assert!(message.contains("distribution_urls"));
        assert!(message.contains("Python interpreter (`python`)"));
    }
}
