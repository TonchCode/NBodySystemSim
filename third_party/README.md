# third_party

Vendored, pinned dependencies of the viewer only (`nbody_core` and `nbody_bench` have none).
Decision: docs/decisions.md D-004.

| Dir | What | Version | Source | License |
|---|---|---|---|---|
| `glfw/` | Window, GL context, input | 3.4 | https://github.com/glfw/glfw/releases/download/3.4/glfw-3.4.zip (sha256 `b5ec004b2712fd08e8861dc271428f048775200a2df719ccf575143ba749a3e9`), minus `docs/`, `examples/`, `tests/` | zlib (`glfw/LICENSE.md`) |
| `glad/` | GL function loader, header-only | glad 2.0.0-beta, GL 3.3 compatibility + KHR_debug | copied from `glfw-3.4/deps/glad/gl.h` | MIT / Apache-2.0 / public-domain spec (see header) |
| `imgui/` | Immediate-mode UI for the viewer's control panel (core + GLFW and OpenGL3 backends) | 1.91.5 | https://github.com/ocornut/imgui/archive/refs/tags/v1.91.5.zip (sha256 `15c60ad10e331378cda381002418f42ea2e98b9c05758eb493b23181bb38ca45`), core files + `backends/imgui_impl_glfw.*`, `imgui_impl_opengl3.*` only | MIT (`imgui/LICENSE.txt`) |

Don't edit these. To upgrade, replace the directory wholesale and update this table.
