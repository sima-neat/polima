export function initialConfig(data, blank) {
  const saved = data.last_run_config || {};
  const savedExists = data.bundles.some(bundle => bundle.id === saved.bundle);
  const bundle = data.bundles.find(item => item.active)?.id || (savedExists && saved.bundle) || data.bundles[0]?.id || "";
  const sameBundle = savedExists && saved.bundle === bundle;
  const selected = data.bundles.find(item => item.id === bundle);
  return {
    ...blank,
    ...(sameBundle ? saved : {}),
    bundle,
    autocomplete: selected?.policy === "groot" ? null : (sameBundle ? saved.autocomplete ?? null : null),
    task: (sameBundle && saved.task) || selected?.default_task || "",
    robot_port: saved.robot_port || data.hardware.arms[0] || "",
    overhead_camera: saved.overhead_camera || data.hardware.cameras.find(c => /046d|c920/i.test(c)) || data.hardware.cameras[0] || "",
    wrist_camera: saved.wrist_camera || data.hardware.cameras.find(c => /sonix|cam1/i.test(c)) || data.hardware.cameras[1] || "",
  };
}
