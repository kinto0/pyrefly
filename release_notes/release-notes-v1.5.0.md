*Release date: November 18, 2026*

Pyrefly v1.5.0 bundles **1466 commits** from **81 contributors**.

---

## 🔍 Release Highlights

- **100% typing spec conformance.** Pyrefly now passes all 145 test files in the [typing conformance suite](https://github.com/python/typing/tree/main/conformance). Some of these fixes change how names in annotations resolve; see [Behavior Changes](#️-behavior-changes) below.
- **Faster checks.** v1.5.0 checks 40% faster than v1.4.1 and uses 9% less memory across the 71 benchmarks in Astral's [ty_benchmark](https://github.com/astral-sh/ruff/tree/main/scripts/ty_benchmark) and the [python-type-checking benchmark](https://python-type-checking.com/typecheck_benchmark/) suites. (Tested on an M5 Max MacBook Pro with 18 threads.)
- **A new `hint` severity.** Diagnostics that were previously IDE-only are now regular error kinds at a new `hint` severity: `unused-import`, `unused-parameter`, `unused-variable`, `unreachable`, and `inactive-code` (formerly `unreachable-code`). By default, they appear in your editor and are hidden from `pyrefly check` output, but you can configure them in `pyrefly.toml`, silence them with suppression comments, or promote them to errors like any other error kind.

---

## ⚠️ Behavior Changes

These may surface new output or errors in code that checked cleanly under v1.4.1.

- **`pyrefly check` prints warnings by default.** This includes uses of deprecated APIs, `redundant-cast`, and `unnecessary-type-conversion`. Warnings don't affect the exit code, and `--suppress-errors`, baseline updates, and JUnit reports ignore them. To hide them, set `min-severity = "error"` in your config or pass `--min-severity=error`.
- **Type annotations look up names in the enclosing class body first.**
  - Postponed annotations, which include string annotations, annotations under `from __future__ import annotations`, and annotations in stub files, look names up in the class body first. So if a class defines a method named `int`, `x: int` in that class refers to the method - even if the method is defined *after* `x`.
  - Pyrefly now reports uninitialized names in every eagerly evaluated annotation, and illegal circular references in annotations.
- **`pyrefly check <dir>` honors config excludes.** A directory argument now uses the config found from that directory and honors its `project-excludes`, so excluded files are no longer checked and `pyrefly check .` checks the same files as `pyrefly check`. Glob arguments are expanded as a shell would, so quoted and unquoted globs give the same result.
- **`unreachable` defaults to `hint` instead of `warn`.** Unreachable code defaults to an IDE-only hint. Set `unreachable` to `"warn"` or `"error"` to keep it visible on the command line.
- **Unused-code hints follow the normal error pipeline.** Because `unused-import`, `unused-parameter`, and `unused-variable` are now error kinds, `disable-type-errors-in-ide`, project includes, and generated-code filtering hide them, and they appear in notebook cells.

---

## ✨ New & Improved

### Type Checking

- Narrowing is more precise on the result of `and` and `or` operations, `type(x) is C` comparisons, and tuple elements guarded by `any(...)` or `all(...)`.
- Exhaustiveness checking for `match` statements is more accurate, especially for tuple subjects and class patterns.
- Pyrefly catches more reads of names that may be uninitialized, such as a name assigned in a loop that may `break`.
- `ParamSpec` and `TypeVarTuple` accept variance keywords and respect declared variance.
- Methods without a positional `self` or `cls` parameter are reported as `bad-function-definition`.
- A new `unexpected-never` warning reports functions that can never return but aren't annotated `NoReturn` or `Never`.
- A new `fallback-search-path-relative-import` warning reports imports that resolve only through the fallback search path heuristic and may fail at runtime, which usually means a search path or site package path is missing from your config.
- The new opt-in error kinds `explicit-any-use` and `partially-unknown-type` show where explicit `Any` and partially unknown types flow through your code.
- The new opt-in `protected-access` error kind reports access to single-underscore members from outside their class, and is what Pyright's `reportPrivateUsage` now maps to.
- The new opt-in `invalid-format-spec` error kind checks f-string format specifications against the value's type and the builtin formatting rules.
- Files whose names aren't Python identifiers, such as Django's `0001_initial.py` migrations, now get module names.
- The new opt-in `infer-param-types-from-docstring` option uses types from Sphinx (reStructuredText) `:param <type> <name>:` and `:type <name>:` fields for unannotated parameters, which helps with older libraries that document types instead of annotating them.

### Library Support

- Pydantic `model_validate` and `model_construct` calls are type checked.
- Pydantic constructor checks follow lax-mode coercion more closely, reducing false positives for enums, named tuples, `NewType`s, and fields with custom schemas.
- Django model fields and managers have more accurate types, including fields accessed on the class rather than an instance, custom `id` primary keys, and querysets from `Manager.as_manager()`.
- SQLAlchemy `select().filter_by()` checks field names and value types.
- Completion and go-to-definition work for FastAPI endpoints called through `TestClient`.

### Language Server

- A new "Type source" section in hover shows where a type was narrowed or first inferred.
- Hover and signature help show parameter documentation from `Annotated[..., Doc()]` and numpydoc docstrings.
- A class hierarchy CodeLens shows overrides and implementations.
- Completions list recently used items first and deprecated items last.
- Auto-import completions and quick fixes use common aliases, such as `import numpy as np`.
- A new quick fix inserts the inferred type annotation for an unannotated declaration.
- A new quick fix inserts stubs for a class's unimplemented abstract members.
- Go-to-definition, find references, and rename work inside quoted annotations.
- Inlay hints show parameter names at call sites, except when the argument already has the same name.
- An experimental `workspace` value for `diagnosticMode` publishes diagnostics across projects.
- LSP requests run on a dedicated thread pool, so slow requests don't block the main thread.

### Configuration & CLI

- The `--only` CLI flag, which limits reported diagnostics to listed error kinds, can now be set in a config file as well via the new `only` option.
- `--extend-project-excludes` adds to the configured excludes, and `--force-exclude` applies `project-excludes` to paths given on the command line.
- PEP 723 scripts can be checked in their own uv environments with `script-command`, or skipped with `--exclude-scripts`.
- Pixi default environments are discovered automatically.
- `--watch` re-reads configs when a config file changes.
- `dump-config --full` shows the fully resolved configuration.
- `reveal_type` output is shown under the `basic` preset, which applies when there is no config file.
- A `required-version` mismatch exits with code 4 instead of 1, so it can be told apart from type errors. When several files are given, Pyrefly now stops before checking them instead of reporting errors from the wrong version.
- When `pyrefly init` finds a BasedPyright baseline, it offers to create a Pyrefly baseline, so a migrated project starts with a clean check.

### Performance

These are some of the changes that make v1.5.0 faster.

- Release executables are built with profile-guided optimization (PGO), and Linux x86-64 executables are further optimized with BOLT. On Linux x86-64, `pyrefly check` is about 20% faster on projects outside the training set, and a language server editing session is about 19% faster.
- Threads checking in parallel no longer contend on shared reference counts: a type refers to its class through a small ID instead of a reference-counted pointer, and non-generic class types no longer allocate type arguments. The class ID change alone cuts total CPU time by 24% and wall time by 10% on the 53 packages of the python-type-checking benchmark on an M5 Max laptop.
- Constructing an instance of an abstract class no longer infers the return type of every method in its class hierarchy, which could cascade through whole libraries. Checks of networkx, dask, and ComfyUI are 40-49% faster, scipy 23%, and tensorflow 19%.
- Checks against large unions of TypedDicts are faster: a union assigned to a wider union that contains it is accepted without comparing members one by one, a dict literal resolves only the keys it names, and TypedDicts are compared field by field. Together, these changes take checking pydantic, whose `CoreSchema` is a union of about 60 TypedDicts, from 3.84 s to 0.88 s.

### Tensor Shapes

- NumPy, PyTorch, and JAX shape stubs cover many more APIs, including NumPy random generators and linear algebra, and JAX's `jax.jit` and `Array.at` indexing.
- Einops shape inference works with JAX and NumPy.
- Shape stubs are discovered through `{package}-shape-stubs` packages.
- The new `use-shape-comments-and-stubs` option turns shape features off.

---

## 🐛 Bug fixes

We closed **123** bug issues this release 👏

- **#3556:** Checking several directories with their own environments no longer reports a class as not assignable to itself (for example, "`Frame` is not assignable to `Frame`"). A library installed in editable mode is now checked in its importer's environment.
- **#804:** Return types of mutually recursive functions are inferred precisely instead of falling back to `Unknown`.
- **#3062:** A `cached_property` that overrides an unannotated class variable no longer causes a false-positive override error, a common pattern in Django code.
- **#3820:** Narrowing a union with `is SomeEnum.MEMBER` now gives the correct type.
- **#2873:** Fixed a false positive with overloaded async iterators.
- And more! #3670, #4131, #5147, #3553, #3931, #4080, #5193, #1332

Thank-you to all our contributors who found these bugs and reported them! Did you know this is one of the most helpful contributions you can make to an open-source project? If you find any bugs in Pyrefly we want to know about them! Please open a bug report issue [here](https://github.com/pyrefly-org/pyrefly/issues).

---

## 📦 Upgrade

```bash
pip install --upgrade pyrefly==1.5.0
```

### How to safely upgrade your codebase

Upgrading the version of Pyrefly you're using or a third-party library you depend on can reveal new type errors in your code. Fixing them all at once is often unrealistic. We've written scripts to help you temporarily silence them. After upgrading, follow these steps:

1. `pyrefly check --suppress-errors`
2. Run your code formatter of choice
3. `pyrefly check --remove-unused-ignores`
4. Repeat until you achieve a clean formatting run and a clean type check.

This will add `# pyrefly: ignore` comments to your code, enabling you to silence errors and return to fix them later. This can make the process of upgrading a large codebase much more manageable.

Read more about error suppressions in the [Pyrefly documentation](https://pyrefly.org/en/docs/error-suppressions/).

---

## 🖊️ Contributors this release

@stroxler, @rchen152, @kinto0, @samwgoldman, @asukaminato0721, @connernilsen, @shahhamdihassan, generatedunixname2066905484085733, David Tolnay, Catherine Gasnier, @mangeshraut712, @ndmitchell, @jorenham, @jakevdp, @Aayushyaash, @1-DARK, @MarcoGorelli, @nitishagar, @AlessandroRuggiero, @QEDady, @bdelwood, @Divyansh151005, @ImBinarry, @ysheikh2, @ting-hong-shieh, @omkar-334, generatedunixname2382037498884706, @WilliamK112, @yeonwooson, Aleksandr Lvov, @soaring0616, @La1itchauhan, @i098, generatedunixname89002005307016, @TimonGaertner, @dhitalkamal, @jny0444, @bniladridas, @sathariels, @KalininDL, @DetachHead, @mvanhorn, @slawlor, Gutti Torvadal, @javabster, @8Keep, @spachgade-meta, @pavelzw, @ak4-sh, ALEXANDRE GEORGE LUSTOSA, generatedunixname873325592204101, @tonyyuyiding, @Cha-Imaa, @willy-b, @Sufiyan-MSA, @thomaspolasek, @Nevsor, @nbadami123, Patrick Lawrence, @nikmos3, @Hirujan-R, @n1shanthb, dtolnay, @austin3dickey, @rexledesma, @ravivats, @Sano007, @swaminimore, @felixscherz, Felipe, @Arham-Begani, @yangdanny97, @Quaden2307, @Nidhi-Gahlawat, @akhil9tiet, @Aling33, @MannXo, @crypt0lith, @GautamKumarOffical, @maharshi-coding, @amas127

---

*Please note: These release notes summarize major updates and features. For brevity, not all individual commits are listed.*
