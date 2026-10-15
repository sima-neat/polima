# Install

PoLiMa Workbench imports and compiles robot policies on a host computer.
PoLiMa Studio runs on the Modalix DevKit for deployment and robot control.
The Workbench host connects to the DevKit over SSH.

| Task | Where it runs | Needs |
|---|---|---|
| Compile a checkpoint | x86_64 Linux computer | Python 3.12, the SiMa ModelSDK; for GR00T also NVIDIA's GR00T code ([your-own-model.md](your-own-model.md)) |
| Deploy and check bundles, Workbench | Linux, macOS, or Windows with WSL | Python 3.10 to 3.12, `ssh`, `rsync` |
| Run the model and the arm | Modalix DevKit | SiMa.ai SDK 2.1.2 or 2.1.3, internet access for the first install |

You do not need the ModelSDK to deploy and run a bundle that is already
compiled. See [examples/so101](../examples/so101/README.md) for the deployment workflow.

## 1. Your computer

Linux and macOS: use a terminal. Windows: install
[WSL](https://learn.microsoft.com/windows/wsl/install) with Ubuntu, and run
everything below in the Ubuntu terminal.

Use the source checkout supplied with your release. Repository access is
required to clone it. Confirm that the checkout contains the intended release
before installation; an unpublished release is not available from the default
GitHub branch.

```bash
git clone https://github.com/sima-neat/polima.git
cd polima
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[deploy]"
polima --version
polima doctor --deploy-only
```

Keep the virtual environment active when using `polima`. In a new terminal,
run `source /path/to/polima/.venv/bin/activate` using your checkout location.
The deploy installation supports imports and deployment; compilation needs
the additional dependencies below.

**To compile**, on an x86_64 Linux computer, build the full environment
instead. It needs Python 3.12 with its `venv` module,
`cmake` and a C++ compiler, and internet access. It installs a CPU-only
PyTorch and LeRobot from a LeRobot checkout at the supported commit. If you
already created `.venv` above with a different Python, remove it first.

```bash
git clone https://github.com/huggingface/lerobot.git ../lerobot
git -C ../lerobot checkout 36b8face988669509272b00f4abe6592d0b17aa0
make venv LEROBOT_SOURCE=../lerobot
source .venv/bin/activate
export MODEL_COMPILER_BIN=/path/to/model-compiler/bin   # the ModelSDK's bin folder
polima doctor                                           # finds the compiler
```

The ModelSDK (model compiler 2.1.3) is licensed separately; get it from the
SiMa.ai Developer Portal.

## 2. The DevKit

Use the same release on both computers. Log in with the account configured
on your DevKit (`ssh <username>@<devkit-address>`) and run:

```bash
git clone https://github.com/sima-neat/polima.git
cd polima
./scripts/install_polima_modalix.sh
```

The DevKit must have its SDK build tools installed, writable installation
directories and the privileges required to register a system service. The
installer checks these prerequisites and reports missing requirements.

The installer registers the Studio service with `sudo` and cannot answer a
password prompt. If `sudo` asks for your password, run `sudo -v` right before
the installer. If the installation outlasts sudo's timeout, run the installer
again; it continues from what is already installed.

The installer:

- creates the LeRobot 0.4.4 environment at `/media/nvme/lerobot` (or reuses it),
- installs PoLiMa and PoLiMa Studio into it,
- builds the runtime (`polima_server`, `polima_cli`) under `/media/nvme/polima`,
- puts the `polima` command on the `PATH`.

If the LeRobot step adds your account to the `dialout` group (needed to open
the arm's serial port), log out and back in before using the arm.

Supported DevKit software:

| DevKit SDK | `sima_lmm` runtime |
|---|---|
| 2.1.2 | 0.3.0 |
| 2.1.3 | 0.4.0 |
| 2.1.3 with LLiMa 0.5.0 | 0.5.0 |

## 3. Connect your computer to the DevKit

PoLiMa needs an ssh login to the DevKit that does not ask for a password. Set
it up once, in either of two ways.

**With the Workbench.** Run `polima workbench start`, open its URL, go to **DevKits**, and
enter the address (for example `sima@192.168.1.20`), a name, and the DevKit's
password. The password is used once to authorize a dedicated PoLiMa SSH key on the
DevKit and is never saved. Unless you select an existing key, PoLiMa creates or reuses
`~/.ssh/id_ed25519_polima` on the Workbench host. This key has no passphrase so
Workbench can connect unattended. Your existing personal SSH keys are unchanged. The Workbench verifies the connection before saving it. Connection setup does
not install Studio. An SSH key path entered here belongs to the Workbench host,
not the browser computer.

**From the terminal.**

```bash
ssh-keygen -t ed25519            # only if you have no key yet
ssh-copy-id sima@<devkit-address>
```

Then tell PoLiMa which DevKit to use:

```bash
export POLIMA_DEVKIT=sima@<devkit-address>   # or the name you gave it in the Workbench
```

or pass `--devkit <address or name>` to `polima deploy` and `polima run`, or
copy `.polima.yaml.example` to `.polima.yaml` and set `board.host`.

**DevKit cabled straight to your computer.** It works the same way: use the
address the DevKit has on that cable. On Windows, WSL reaches the DevKit
through Windows' network connection; if WSL cannot reach an address that
Windows can, turn on WSL's mirrored networking mode.

The first connection to a DevKit records its identity. If the DevKit is
reinstalled later, ssh refuses the changed identity; remove the old record
once with `ssh-keygen -R <devkit-address>`.

## Check

For a deployment-only host:

```bash
polima doctor --deploy-only
```

On a compiler host or the DevKit, run `polima doctor` for the full checks.

It checks Python, the compiler (if installed), and the DevKit when one is set.

## Workbench and Studio commands

After installation, Workbench runs as a per-user background process:

```bash
polima workbench start
polima workbench status
polima workbench stop
```

By default, Workbench listens on `127.0.0.1`, accessible from the same
computer. Open the address printed by `start`. `polima workbench serve` runs in the
foreground; Ctrl-C stops it. Workspace directories and saved DevKit records
are managed automatically. No environment variables or JSON files are required
for normal setup.

### Remote Workbench

Start Workbench on the Linux compiler host, then forward its port from your
laptop. Replace `<user>@<host>` with your server login:

```bash
# On the compiler host
polima workbench start --host 127.0.0.1 --port 8686

# In a separate terminal on your laptop
ssh -N -L 8686:127.0.0.1:8686 <user>@<host>
```

Keep the SSH session open and visit `http://localhost:8686` on your laptop.
If Workbench prints a different port, use that port on the right side of the
forwarding rule. `127.0.0.1` always refers to the computer where the connection
is made; it is not a DevKit address.

To allow direct access on a trusted network, explicitly start with
`--host 0.0.0.0` and open the compiler host's address in your browser. Workbench
does not provide login authentication; use an SSH tunnel for remote access.

Paths and SSH keys entered in Workbench refer to the Workbench host. Folder
uploads come from your browser computer, with a 64 GiB limit per upload.
macOS can host Workbench for imports and deployment, or use a Linux-hosted
Workbench through its browser; compilation requires Linux.

The **Setup guide** in Workbench explains checkpoint contents, representative
LeRobot data and GR00T environment configuration. Missing prerequisites are
shown before compilation. Dataset inspection is a structure check, not a
complete model compatibility test.

On the DevKit, the installer installs `polima-studio.service` when systemd is
available and leaves it stopped. Registering the service requires the privileges
reported by the installer. Installing the Python package alone is insufficient.
After the installer completes:

```bash
polima studio start
polima studio status
polima studio stop
```

`Unit polima-studio.service not found` means service installation is incomplete.
Inspect the installer output. `polima studio serve` runs in the foreground when
the Studio dependencies are installed. Neither command installs dependencies.
