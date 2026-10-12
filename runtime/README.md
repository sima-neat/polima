# PoLiMa runtime

The C++ runtime executes compiled bundles on the SiMa.ai Modalix DevKit.
`polima_server` serves policy inference; `polima_cli` manages installed bundles,
the server and robot operations.

Use the [DevKit installer](../docs/install.md) for a complete installation.
Building requires the DevKit's SimaLMM development package, a C++20 compiler,
CMake, jsoncpp and nlohmann/json headers.

```bash
cmake -S runtime -B build/runtime -DCMAKE_BUILD_TYPE=Release
cmake --build build/runtime -j 8
```

The build produces `polima_server`, `polima_cli` and kernel test executables.
A native build requires the Modalix SDK; a host Python test does not validate
accelerator execution.

After installation, `polima server start`, `polima server status` and
`polima server stop` manage inference. See [deployment](../docs/deploy.md)
for bundle installation and [robot operation](../docs/robot.md) for SO-101 use.
