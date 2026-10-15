# PoLiMa Studio

Operate robot policies on the Modalix DevKit. Studio manages installed bundles,
cameras, calibration, inference benchmarks and robot runs.
Install it with the [DevKit installer](../../../docs/install.md).

```bash
polima studio start
polima studio status
polima studio stop
```

Open the URL printed by `status`. The installer registers the service and
leaves it stopped. `polima studio serve --port 8080` runs in the foreground.
`polima studio enable` enables startup at boot; `polima studio logs --follow`
shows service logs. Installing the Python package alone does not register the
system service.

![Studio navigation: robot setup, bundle transfer, runtime and benchmarks](../../../docs/images/studio.gif)

This preview uses example data; it does not show a hardware run.

## Use Studio

1. Upload a compiled bundle folder or tar archive in **Bundles & files**.
2. Activate it and run a fixture benchmark to check inference and output fidelity.
3. Configure the cameras and calibrate your follower arm.
4. Review robot readiness, then start a run from the Run page.
5. Review run history and download logs for analysis.

Benchmarking does not move the arm. It reports latency distributions,
throughput, available stage timings and fixture-output agreement. Those
measurements do not establish task success on a physical robot.

Studio accepts resumable bundle uploads in 16 MiB chunks. Installation checks
ELF checksums against the manifest. Robot calibration is managed separately
from model weights and has backups for restoration.

## Access and control

Studio has no login or TLS. Use a trusted DevKit network. Requests that change
state require same-origin CSRF validation. One browser holds the controller
lease; closing that browser does not stop a running robot. See
[robot operation](../../../docs/robot.md) for stop behavior.

Run records are stored under the installation's `var/lib`, and logs under
`var/log/studio`. The API is rooted at `/api/v1`; the browser uses `session`,
`snapshot`, `lease`, `robot/*`, `benchmarks/*`, `uploads` and `events`.
