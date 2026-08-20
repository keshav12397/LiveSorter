# Vendored third-party sources

Dropped in as plain source, unmodified, because this project builds with
`cl.exe`/`nvcc` invoked directly from `.vscode/tasks.json` -- there is no
package manager, no CMake, and no network fetch in the build.

Only the files `SpikeViewer.exe` actually compiles or includes are kept;
demos, docs, examples, and the backends for the toolkits this project does
not use were dropped. Nothing that remains was edited -- if a file here ever
needs a change, record it in this file, because "unmodified upstream" is the
only reason it is safe to re-pull a newer tag over the top.

| directory | upstream | version | license |
|---|---|---|---|
| `imgui/`  | https://github.com/ocornut/imgui  | v1.90.9 (tag) | MIT, `imgui/LICENSE.txt` |
| `implot/` | https://github.com/epezent/implot | v0.16 (tag)   | MIT, `implot/LICENSE` |

`imgui/backends/` carries the **Win32 + Direct3D 11** backend only. That pair
was chosen over GLFW+OpenGL3 because both halves ship with the Windows SDK
this repo already builds against (`d3d11.lib`/`d3dcompiler.lib` are in the
same `10.0.19041.0` `um\x64` directory the existing tasks already put on
`LIB`), so `SpikeViewer.exe` adds no external dependency to the machine at
all. GLFW would have meant vendoring and building a third library.

ImPlot depends on ImGui and on nothing else; it needs `imgui_internal.h`,
which is why that header is kept even though no ClosedLoop code includes it.
