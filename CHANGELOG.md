# Changelog

## 0.3.0 (unreleased)

- Compile ACT and GR00T N1.6/N1.7 checkpoints for Modalix.
- Import checkpoint and dataset folders through Workbench.
- Check compiler prerequisites before starting a build.
- Compare recorded output fidelity, inference latency and graph profiles.
- Manage DevKit connections and review installation status.
- Start, stop and inspect Workbench with the `polima workbench` command.
- Install bundles and benchmark model inference in Studio.

## 0.2.0

First public version.

- **ACT**: compile a LeRobot ACT checkpoint (`polima compile`), deploy it
  (`polima deploy`) and drive an SO-101 arm from the DevKit (`polima robot run`).
- **GR00T N1.6 and N1.7**: run prebuilt mixed-precision bundles on the DevKit
  and drive an SO-101 arm. Measured on the arm: 317 ms (N1.6) and 233 ms (N1.7)
  of model time per 32-move plan.
- **Faster DevKit runtime**: model data stays on the accelerator between model
  parts, and camera pictures are prepared in parallel. Outputs are unchanged.
- **SO-101 motion settings**: plan blending, half-steps, eased return to rest
  and early next-plan requests, all opt-in ([docs/robot.md](docs/robot.md)).
- **PoLiMa Studio** (DevKit) and **Workbench** (host): web pages to install
  bundles, calibrate, run and review the robot, and track bundles and results.
- **Your DevKits in the Workbench**: add any DevKit by address and ssh key; it
  is checked over ssh and kept, and its name works as `--devkit <name>`.
- Supports SiMa.ai SDK 2.1.2 and 2.1.3 on the DevKit.
- Licensed under Apache-2.0.
