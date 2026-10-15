# PoLiMa Workbench

Import checkpoints, compile for the Modalix DevKit and compare measured results.
Start with the [installation guide](../../../docs/install.md).

```bash
polima workbench start
polima workbench status
polima workbench stop
```

Open the URL printed by `start`. For a foreground process, use
`polima workbench serve`; Ctrl-C stops it. Workspace and connection records
are managed automatically; normal setup needs no environment variables or
manual JSON configuration. For a server accessed from a laptop, follow
[remote access](../../../docs/install.md#remote-workbench).

1. **Import.** Upload a checkpoint folder and its representative dataset from
   your browser. Files are stored on the Workbench host. The upload limit is
   64 GiB per folder. Keep checkpoint configuration and weights together.
   In a source checkout, checkpoints under `models/` are listed automatically.
   Use `--models DIR` or `POLIMA_MODELS` to select other folders, including
   when PoLiMa is installed from a wheel.
   A checkpoint or dataset on Hugging Face can be imported by repository id
   instead: the Workbench host downloads it directly, using the token saved
   there with `hf auth login` for private or gated repositories. Starting the
   same repository again resumes an interrupted download. When a checkpoint's
   `train_config.json` names a Hub dataset, the Compile page offers it.
   Downloads happen only at import; compilation stays offline.
2. **Compile and validate.** Select a checkpoint, review prerequisite checks
   and choose its dataset. Compilation requires an x86_64 Linux host with the
   ModelSDK and the selected policy's export environment.
3. **Deploy and run.** Connect a DevKit, install PoLiMa using the installation
   guide, then deploy a bundle and open Studio.

The in-app **Setup guide** explains data preparation and environment setup.
macOS can run Workbench for imports and deployment; compilation runs on Linux.
When using a remote Workbench, typed paths and SSH keys belong to that host,
while uploaded files come from your browser's computer.

**Compare** displays recorded measurements. Missing measurements remain
unavailable until validation or benchmarking runs. Output agreement with
PyTorch measures numerical fidelity; robot task success requires robot trials.
Compare builds using matching inputs, tasks, hardware and run settings.
