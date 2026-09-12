# Packaging and launch

EdgeMesh has standard Python wheel/source distribution metadata and an `edgemesh` command. The package reads its version from `version.py`; release versions still belong to the version-manager workflow.

The one-command launch for this local checkout is:

```powershell
uvx --python 3.12 --exclude-newer "1 week" --from "D:\Documents\Code\GitHub\EdgeMesh[depth,assistants]" edgemesh
```

This requires uv. It creates an isolated Python 3.12 environment, installs the desktop, depth and optional Antigravity SDK dependencies, and opens EdgeMesh. The first run can download large native libraries; subsequent runs reuse uv's cache. Model weights remain separate and use the application's model-management flow. The explicit local source selects this checkout; no public PyPI release is assumed. [Copyable launch and installation commands](Packaging.html).

For regular use, `uv tool install` creates a persistent environment and exposes `edgemesh` on PATH; `uvx` is convenient for trying a specific local source or wheel. Both isolate app dependencies from other projects. uv reports when its executable directory needs adding to PATH. See [uv's official tool guide](https://docs.astral.sh/uv/guides/tools/).

## Runtime profiles

| Profile | Included |
| --- | --- |
| Base | Image processing, mask editing, contour geometry tools, PySide6, Open3D and the embedded VTK viewer |
| `depth` extra | Pinned torch, torchvision and Transformers; model weights remain separate |
| `assistants` extra | Pinned Google Antigravity SDK; provider authentication remains a separate setup step |
| `spacemouse` extra | Optional pygame-ce and hidapi hardware support |

The depth extra supplies the current Hugging Face inference stack. Legacy MiDaS/DPT registration also requires a reviewed, immutable source checkout and matching checkpoint; registration does not install that checkout's additional Python dependencies. Those checkout-specific dependencies were not verified in the clean packaging environment.

The base profile does not require torch or Transformers. Claude and Codex command-line runtimes are external tools and are not bundled into this Python wheel. Installing the Antigravity SDK does not authenticate it. EdgeMesh source is 0BSD; bundled MeshTools helpers are CC0-1.0. Model-specific and other dependency terms remain separate.

## Command-line behavior

The installed command supports `edgemesh --help`, `edgemesh --version`, an optional project file/directory argument, and `edgemesh diagnose-depth --help`. Help and version do not import PySide6 or torch. Depth diagnostics use their own argument parser and preserve JSON output without launcher messages on standard output. Missing depth dependencies are logged and reported on standard error with an instruction to install the `depth` extra.

## Supported Python and platform boundary

The packaged release requires CPython 3.12. This matches the source's minimum syntax version and Open3D 0.19.0's available wheels. Open3D publishes CPython 3.12 wheels for Windows x64, Linux x64 with glibc 2.31+, and macOS universal2; it does not publish 3.13/3.14 wheels for this pinned release. [Open3D 0.19.0 distribution files](https://pypi.org/project/open3d/0.19.0/#files).

| Platform | Validation |
| --- | --- |
| Windows x64, CPython 3.12 | Target platform; measured checks are recorded below |
| Linux x64 / macOS | Upstream wheel availability alone does not establish a working EdgeMesh desktop installation; not tested here |
| Python 3.13/3.14, Windows ARM64/32-bit | Outside the packaged support matrix |

The package pins direct dependencies to the checkout's tested versions. The Antigravity SDK pin was separately installed and its configuration probe passed in a clean Python 3.12 environment. uv launch/install commands exclude packages less than one week old, including transitive/build dependencies. Existing source-run environments are unchanged. GPU acceleration depends on the selected torch build and matching driver; CPU checks do not establish GPU behavior.

## Contents and state

The wheel includes application Python modules, MeshTools Python helper modules and their license, the app icon, and one example image. It excludes Git metadata, saved user projects, logs, caches, test trees and model weights. Read-only resources resolve independently of the shell's current directory through `edgemesh_bootstrap.resources.resource_path`. User settings, projects, downloaded models and assistant credentials belong in user state directories.

The source manifest preserves the Git-tracked `README.md`, `MeshTools/` and `Images/` casing. Local uv source-cache keys include application modules and declared assets without traversing virtual environments or model caches.

## Validation

Validated on Windows x64 with CPython 3.12.10 on 2026-09-11.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| [Wheel](../dist/edgemesh-0.6.7-py3-none-any.whl) | 1,689,272 | `52abf638f0038c1929bd4d939a747882fff5c7b718e79ad626a63f36462e7c86` |
| [Source archive](../dist/edgemesh-0.6.7.tar.gz) | 1,669,598 | `d11927383f46296306b1b49da0b3bfe23c24c259c971e03173ba9ad8c73334a7` |

- All nine packaging tests, scoped Ruff checks and packaging compilation passed. The broader 325-test run preceded two added packaging regressions; it was not rerun after those build-only fixes.
- The wheel was built from the corrected source archive. Archive inspection verified 45 application modules, 13 MeshTools modules, four bootstrap modules, both assets, both licenses and exact `ReadMe.md` casing. Neither archive contains tests, Git metadata, caches, user state or model weights.
- Clean base and full environments contain 87 and 131 compatible packages respectively; both passed `uv pip check`.
- Both installed desktop probes ran from the Windows Temp directory with `PYTHONPATH` removed and separate user-state directories. They verified initialized/product-ready UI, the assistant panel, empty UI error logs and module/resource paths inside their installed `site-packages`. The base probe also verified that torch, Transformers and the Antigravity distribution are absent.
- Installed `--help`, `--version` and `diagnose-depth --help` passed. The installed SDK configuration probe returned available=true, version=0.1.16, without authentication or inference.
- The final `uvx` wheel check resolved 131 packages offline in 545 ms, installed them from the populated cache in 2m 21s, printed EdgeMesh help and exited 0.

The final build pass corrected two packaging issues: PEP 517 did not add the source root to `sys.path`, so `setup.py` now loads its helper by file path; setuptools automatically included root tests in the source archive, so the manifest explicitly prunes them. Each correction has a regression test.

To launch the built artifact in a normal user PowerShell session:

```powershell
uvx --python 3.12 --exclude-newer "1 week" --from "edgemesh[depth,assistants] @ file:///D:/Documents/Code/GitHub/EdgeMesh/dist/edgemesh-0.6.7-py3-none-any.whl" edgemesh
```

The sandbox could not access uv's default Roaming Python/tool state directories. Its successful test used the existing Python 3.12 interpreter explicitly, task-owned `UV_TOOL_DIR`/`UV_CACHE_DIR`, `--offline`, `--no-config` and the public PyPI index. The generic `--python 3.12` launch commands above were therefore not literally validated in this sandbox. The exact tested command and environment setup have copy buttons in [the launch guide](Packaging.html).

```powershell
uvx --verbose --no-config --cache-dir "D:\Documents\Code\GitHub\EdgeMesh\.cache\packaging-uv" --offline --python "D:\Documents\Code\GitHub\EdgeMesh\.venv\Scripts\python.exe" --exclude-newer "1 week" --default-index "https://pypi.org/simple" --from "edgemesh[depth,assistants] @ file:///D:/Documents/Code/GitHub/EdgeMesh/dist/edgemesh-0.6.7-py3-none-any.whl" edgemesh --help
```

[Compact validation report and evidence paths](../Notes/Packaging_Validation-2026-09-11.json) record the exact environment, test counts, artifact hashes and probe results. Packaging validation covers offscreen desktop initialization; it does not establish GPU rendering, frozen executable behavior, cross-platform operation, model inference or live provider authentication.


No package has been uploaded to PyPI or a release host. A public `uvx edgemesh` command is not claimed until the intended package name, release license metadata, artifacts and publication are verified. cx_Freeze and Nuitka executable builds are outside this wheel/source-distribution validation.
