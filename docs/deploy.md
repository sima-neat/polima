# Deploying bundles

A *bundle* is a folder that holds everything the DevKit needs to run one model:
the compiled model files, `plan.json` (how to run them), `bundle.json` (what the
bundle contains, with a checksum for every model file), test inputs, and the
robot client. Its name ends in a short hash of the model files, so two bundles
with the same name hold the same model.

`polima deploy` runs on the host and copies a bundle to the DevKit over `ssh`
and `rsync`. It also builds PoLiMa's runtime on the DevKit, so the DevKit always
runs the runtime that matches your PoLiMa version.

## Before the first deploy

1. Install PoLiMa on the DevKit once with `./scripts/install_polima_modalix.sh`
   (see the README).
2. Make sure `ssh sima@<devkit-address>` works from the host without a password
   prompt (use an SSH key).
3. Tell PoLiMa the address:

   ```bash
   export POLIMA_DEVKIT=sima@<devkit-address>
   ```

   or pass `--devkit sima@<devkit-address>` to every command, or set `board.host`
   in `.polima.yaml` (copy `.polima.yaml.example`). The name of a DevKit added
   in the Workbench works too, with its saved ssh key: `--devkit <name>`.

## Deploy a bundle

```bash
polima deploy --bundle <bundle-id>                 # a bundle under outputs/bundles/
polima deploy --bundle /path/to/<bundle-folder>    # or any bundle folder
```

Provide a bundle with `--bundle` and configure the DevKit address before
deployment. Use `--list` to inspect bundles already installed on the DevKit.

What happens, in order (the command prints each step):

| Step | What it does |
|---|---|
| preflight | Checks the DevKit: it is a Modalix, and it has `cmake`, the SiMa runtime development files and `jsoncpp`. Fails before copying anything. |
| layout | Creates the folders under `/media/nvme/polima`. |
| sync-bundle | Copies the bundle to `/media/nvme/polima/models/<bundle-id>/`. Files that are already there are not copied again. |
| python-runtime | Copies PoLiMa's Python package to `/media/nvme/polima/python/`, which the robot client uses. |
| build | Builds `polima_server` and `polima_cli` on the DevKit and installs them in `/media/nvme/polima/bin/`. Skipped when the runtime source has not changed since the last build. Then links them onto the `PATH` where it can. |
| verify | Checks every model file on the DevKit against its checksum in `bundle.json`. |
| guard | Checks that the bundle folder on the DevKit holds no archive files. |
| activate | Points `/media/nvme/polima/current` at the new bundle. `--no-activate` skips this. |
| service | Starts the model server, only with `--start`. |

The copy on the DevKit is made identical to the host's copy. If you changed
files inside the bundle on the DevKit (for example the arm calibration in
`robot_client/calibration/`), keep a copy: a later deploy of a changed bundle
replaces them.

Useful options:

| Option | Use |
|---|---|
| `--start` | Start the model server after deploying. |
| `--no-activate` | Copy the bundle but keep the current one active. |
| `--force` | Copy everything again and rebuild the runtime. |
| `--no-build` | Do not build the runtime (only if it is already up to date). |
| `--json` | Print the result as JSON. |

## Manage the DevKit from the host

```bash
polima deploy --list                 # bundles on the DevKit
polima deploy --status               # is the model server running?
polima deploy --activate <bundle-id> # switch the active bundle and restart the server
polima deploy --logs                 # last 40 lines of the server log (--logs 200 for more)
polima deploy --stop                 # stop the model server
polima deploy --reset-mla            # stop the server and reset the accelerator
```

Check a running model from the host:

```bash
polima run --bundle <bundle-id> --fixture              # test input, checked against the expected answer
polima run --bundle <bundle-id> --fixture --repeat 20  # also report inference latency
```

## Manage the DevKit on the DevKit

```bash
polima activate                  # pick the active bundle from a list
polima activate <bundle-id> --start
polima server status
polima server start
polima server stop
```

Activating a bundle stops a running server first, so the server never reports
one bundle while running another.

## Accelerator reset and sudo

A model load failure (`MLA_LOAD_FAILED`) can require accelerator recovery.
`polima deploy` attempts recovery and retries once. Recovery restarts shared
DevKit runtime services and can interrupt other accelerator workloads. End
robot runs and other accelerator work before recovery.

To recover manually, run on the DevKit:

```bash
polima server stop
sudo /usr/bin/fix_devkit_runtime.sh
```

or from the host: `polima deploy --reset-mla`.

The reset needs `sudo` on the DevKit. PoLiMa does not store a password. Either
allow passwordless `sudo` for `/usr/bin/fix_devkit_runtime.sh` and
`/usr/bin/init_mla_memory.sh`, or set the password in the environment of the
host shell that runs `polima deploy`:

```bash
export POLIMA_MLA_SUDO_PASSWORD=...
```

## Installing from a browser instead

PoLiMa Studio can install a bundle uploaded from a browser
([Studio guide](../src/polima/studio/README.md)). It checks the model files the same way
but does not rebuild the runtime, so the DevKit must already have the runtime
from this PoLiMa version (from the installer or an earlier `polima deploy`).
