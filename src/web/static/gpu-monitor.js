export function mountGpuMonitor() {
  const $ = (selector) => document.querySelector(selector);
  const dialog = $("#gpuMonitorDialog");
  let timer;
  let controller;
  let revision = 0;

  const escape = (value) => String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
  const numeric = (value) => typeof value === "number" && Number.isFinite(value) && value >= 0;
  const number = (value, digits = 0) => numeric(value) ? value.toFixed(digits) : "—";
  const gib = (value) => numeric(value) ? `${(value / 1024).toFixed(2)}` : "—";

  function bar(value, label) {
    if (!numeric(value)) return '<div class="gpu-meter unavailable">—</div>';
    const percent = Math.max(0, Math.min(100, value));
    const tone = percent >= 85 ? "high" : percent >= 60 ? "medium" : "low";
    return `<div class="gpu-meter"><div class="gpu-meter-track" role="progressbar" aria-label="${escape(label)}" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${percent.toFixed(1)}"><span class="${tone}" style="width:${percent}%"></span></div><strong>${percent.toFixed(1)}%</strong></div>`;
  }
  function render(data) {
    const rows = data.gpus || [];
    $("#gpuMonitorMessage").textContent = data.error || (!rows.length ? "未检测到 NVIDIA GPU" : "");
    $("#gpuMonitorTable").hidden = !rows.length;
    $("#gpuMonitorRows").innerHTML = rows.map((gpu) => `<tr data-gpu-id="${escape(gpu.id)}">
      <td><strong>GPU ${escape(gpu.id)}</strong><div class="gpu-device-name">${escape(gpu.name)}</div></td>
      <td><div class="gpu-memory-value">${gib(gpu.memory_used_mb)} / ${gib(gpu.memory_total_mb)} GiB</div>${bar(gpu.memory_percent, `GPU ${gpu.id} 显存`)}</td>
      <td>${bar(gpu.utilization_percent, `GPU ${gpu.id} 利用率`)}</td>
      <td>${number(gpu.temperature_c)}${numeric(gpu.temperature_c) ? " °C" : ""}</td>
      <td>${number(gpu.power_w)} / ${number(gpu.power_limit_w)} W</td>
    </tr>`).join("");
    $("#gpuMonitorTitle").title = [rows[0]?.driver_version ? `驱动 ${rows[0].driver_version}` : "", data.updated_at ? new Date(data.updated_at).toLocaleTimeString() : ""].filter(Boolean).join(" · ");
  }
  function stop() {
    revision += 1;
    clearTimeout(timer);
    controller?.abort();
    controller = null;
  }
  async function refresh() {
    if (!dialog.open || document.hidden || controller) return;
    const currentRevision = revision;
    const request = new AbortController();
    controller = request;
    const timeout = setTimeout(() => request.abort(), 8000);
    try {
      const response = await fetch("/api/system/gpus", { cache: "no-store", signal: request.signal });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "GPU 状态读取失败");
      if (currentRevision === revision && dialog.open) render(data);
    } catch (error) {
      if (currentRevision === revision && dialog.open) render({ gpus: [], error: request.signal.aborted ? "GPU 状态读取超时" : error.message });
    } finally {
      clearTimeout(timeout);
      if (controller === request) controller = null;
      if (currentRevision === revision && dialog.open && !document.hidden) timer = setTimeout(refresh, 2000);
    }
  }
  $("#gpuUsageBtn").addEventListener("click", () => {
    if (dialog.open) return;
    stop();
    $("#gpuMonitorMessage").textContent = "加载中…";
    $("#gpuMonitorRows").replaceChildren();
    $("#gpuMonitorTable").hidden = true;
    dialog.showModal();
    refresh();
  });
  $("#closeGpuMonitorBtn").addEventListener("click", () => dialog.close());
  dialog.addEventListener("close", stop);
  dialog.addEventListener("click", (event) => {
    if (event.target !== dialog) return;
    const box = dialog.getBoundingClientRect();
    if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) dialog.close();
  });
  document.addEventListener("visibilitychange", () => {
    stop();
    if (dialog.open && !document.hidden) refresh();
  });
  window.addEventListener("pagehide", stop);
}
