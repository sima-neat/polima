// Benchmark history is newest first. Never borrow another bundle's measurements.
export function benchmarkSelection(records, bundle) {
  const own = records.filter(item => item.bundle === bundle);
  return { latest: own[0], previous: own.slice(1).find(item => item.status === "completed" && item.result) };
}
export function meanChange(latest, previous) {
  const now = latest?.result?.mean, before = previous?.result?.mean;
  if (!latest || !previous || latest.status !== "completed" || previous.status !== "completed" || latest.bundle !== previous.bundle ||
      !Number.isFinite(now) || now < 0 || !Number.isFinite(before) || before <= 0) return null;
  return (now - before) / before * 100;
}

// Inter-event rates describe the recorded window, including pauses within it.
export function eventTiming(events, kind) {
  const times = events.filter(e => e.event === kind && Number.isFinite(e.t)).map(e => e.t).sort((a,b) => a-b);
  if (times.length < 2 || times.at(-1) <= times[0]) return { rate: null, gap: null };
  let gap = 0;
  for (let i=1; i<times.length; i++) gap = Math.max(gap, (times[i]-times[i-1])*1000);
  return { rate: (times.length-1)/(times.at(-1)-times[0]), gap };
}
