# Pyrefly: A fast type checker and language server for Python with powerful IDE features

[![pyrefly](https://img.shields.io/endpoint?url=https://pyrefly.org/badge.json)](https://github.com/pyrefly-org/pyrefly)
[![PyPI](https://img.shields.io/pypi/v/pyrefly?color=blue&label=pypi)](https://pypi.python.org/pypi/pyrefly)
[![VS Code](https://img.shields.io/badge/VS%20Code-Marketplace-blue)](https://marketplace.visualstudio.com/items?itemName=meta.pyrefly)
[![Open VSX](https://img.shields.io/open-vsx/dt/meta/pyrefly?color=blue&label=Open%20VSX)](https://open-vsx.org/extension/meta/pyrefly)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.gg/Cf7mFQtW7W)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Pyrefly is a type checker and language server for Python, written in Rust. It
provides fast type checking along with IDE features such as code navigation,
semantic highlighting, and code completion.

## Who Uses Pyrefly

- **[Google](https://opensource.googleblog.com/2026/10/adopting-pyrefly-as-the-python-type-checker-at-google.html)**: the Python type checker for Google's monorepo, replacing Pytype ([blog post](https://opensource.googleblog.com/2026/10/adopting-pyrefly-as-the-python-type-checker-at-google.html)).
- **[Microsoft Pylance](https://github.com/microsoft/pylance-release)**: can use Pyrefly for type checking through the [Type Server Protocol (TSP)](https://github.com/microsoft/pylance-release/blob/main/docs/tsp/type-server-protocol.md) ([announcement](https://devblogs.microsoft.com/python/announcing-tsp/)).
- **[Instagram](https://www.instagram.com/)**: the default type checker for 20 million lines of Python at Meta.
- Open source projects type checked with Pyrefly:
  - [PyTorch](https://github.com/pytorch/pytorch)
  - [JAX](https://github.com/jax-ml/jax)
  - [NumPy](https://github.com/numpy/numpy)
  - [SciPy](https://github.com/scipy/scipy)
  - [scikit-learn](https://github.com/scikit-learn/scikit-learn)
  - [Polars](https://github.com/pola-rs/polars)
  - [pandas-stubs](https://github.com/pandas-dev/pandas-stubs)
  - [Narwhals](https://github.com/narwhals-dev/narwhals)
  - [napari](https://github.com/napari/napari) ([blog post](https://napari.org/island-dispatch/blog/from-any-to-certainty.html))
  - [MONAI](https://github.com/Project-MONAI/MONAI)
  - [Nilearn](https://github.com/nilearn/nilearn)
- **[Dify](https://github.com/langgenius/dify)**: Pyrefly reports type coverage in CI ([case study](https://pyrefly.org/blog/dify-pyrefly-coverage-ci/)).
- **[Positron](https://positron.posit.co/blog/posts/2026-03-31-python-type-checkers/)**: Pyrefly is the default Python language server.

If your project uses Pyrefly, [create an issue](https://github.com/pyrefly-org/pyrefly/issues) to add it to this list.

## Key Features

- **Fast.** Pyrefly checks over 1.85 million lines of code per second, type checking projects like PyTorch 15x faster than Mypy and Pyright. In the IDE, rechecks typically complete in under 10 milliseconds after saving a file.
- **Full-featured language server.** Go-to-definition, autocomplete, hover, inlay hints, semantic highlighting, and more, with consistent results across the CLI and your editor.
- **Editor support.** [VS Code](https://marketplace.visualstudio.com/items?itemName=meta.pyrefly), [Neovim](https://pyrefly.org/en/docs/IDE/#neovim), [Zed](https://zed.dev/extensions/pyrefly), and [more](https://pyrefly.org/en/docs/IDE/).
- **Understands real-world Python.** Built-in support for [Pydantic](https://pyrefly.org/en/docs/pydantic/), [Django](https://pyrefly.org/en/docs/django/), and [pytest](https://pyrefly.org/en/docs/pytest/).
- **Tensor shape checking.** Experimental [tensor shape checking](https://pyrefly.org/en/docs/tensor-shapes/) finds shape mismatches in PyTorch code.
- **Adoption-ready.** Migrate from Mypy or Pyright with `pyrefly init`, silence existing errors with `pyrefly suppress`, and add annotations with `pyrefly infer`. Start with one file and grow at your own pace.

Pyrefly is [stable](https://github.com/pyrefly-org/pyrefly/releases/tag/1.0.0).
See the [Pyrefly website](https://pyrefly.org) for full documentation.

## Getting Started

- Try it in your browser: [Sandbox](https://pyrefly.org/sandbox/)
- Get the command-line tool: `pip install pyrefly`
- Get the IDE extension: [IDE installation page](https://pyrefly.org/en/docs/IDE/)

## Version Policy

Pyrefly releases new minor versions (`1.x.0`) monthly and patch versions in between
as-needed for critical fixes. Pyrefly does *not* follow strict semantic versioning:
minor versions contain more significant changes than patch versions, but any
version may introduce new type errors and other breaking changes. The
[`pyrefly suppress`](https://pyrefly.org/en/docs/error-suppressions/) command can be used
to easily silence errors when upgrading to a new version.

## Getting Involved

If you have questions or would like to report a bug, please
[create an issue](https://github.com/pyrefly-org/pyrefly/issues).

See our
[contributing guide](https://github.com/pyrefly-org/pyrefly/blob/main/CONTRIBUTING.md)
and
[architecture overview](https://github.com/pyrefly-org/pyrefly/blob/main/ARCHITECTURE.md)
for information on how to contribute to Pyrefly.

Join our [Discord](https://discord.com/invite/Cf7mFQtW7W) to chat about Pyrefly
and types. This is also where we hold biweekly office hours.
