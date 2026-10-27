const escapeHtml = (value) => String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");

export function mountBenchmarkPicker(root, bootstrap, initial, changed) {
  const $ = (selector) => root.querySelector(selector);
  const order = new Map([["t2i", 0], ["ti2i", 1], ["multi_reference", 2], ["demo", 99]]);
  const available = bootstrap.benchmarks.filter((item) => !item.error).sort((a, b) => (order.get(a.name) ?? 50) - (order.get(b.name) ?? 50) || a.name.localeCompare(b.name));
  const savedRows = Array.isArray(initial.benchmarks) ? initial.benchmarks : null;
  const entries = available.map((item, index) => {
    const saved = savedRows?.find((row) => row?.name === item.name);
    const legacy = initial.benchmark === item.name;
    const ids = saved ? saved.sample_ids : legacy ? initial.selectedSamples : null;
    return { ...item, language: (saved?.language || (legacy ? initial.language : "zh")) === "en" ? "en" : "zh", checked: savedRows ? Boolean(saved && saved.enabled !== false) : initial.benchmark ? legacy : index === 0,
      ids: Array.isArray(ids) ? new Set(ids) : null, samples: null, cache: new Map(), loading: false, error: "", request: 0 };
  });
  let disabled = false;
  let disposed = false;
  let dialogEntry = null;
  let draft = new Set();
  let page = 1;
  const pageSize = 50;

  function notify() { if (!disposed) changed(); }
  function selectedCount(entry) { return entry.ids === null ? entry.sample_count : entry.ids.size; }

  function render() {
    $("#benchmarkList").innerHTML = entries.length ? entries.map((entry, index) => `<div class="benchmark-item ${entry.checked ? "selected" : ""}">
      <label class="benchmark-check"><input type="checkbox" data-benchmark-check="${index}" ${entry.checked ? "checked" : ""} ${disabled ? "disabled" : ""} aria-label="选择 Benchmark ${escapeHtml(entry.name)}" /><strong>${escapeHtml(entry.name)}</strong></label>
      <div class="benchmark-actions"><span>${entry.loading ? "…" : `${selectedCount(entry)}/${entry.samples?.length ?? entry.sample_count}`}</span><button type="button" class="text-button" data-benchmark-filter="${index}" ${disabled ? "disabled" : ""}>筛选样本</button></div>
      <select class="benchmark-language" data-benchmark-language="${index}" aria-label="${escapeHtml(entry.name)} Prompt 语言" ${disabled ? "disabled" : ""}><option value="zh" ${entry.language === "zh" ? "selected" : ""}>中文</option><option value="en" ${entry.language === "en" ? "selected" : ""}>English</option></select>
      ${entry.error ? `<p class="benchmark-error" role="alert">${escapeHtml(entry.error)}</p>` : ""}
    </div>`).join("") : '<div class="empty-state">暂无可用 Benchmark</div>';
  }

  async function load(entry) {
    const request = ++entry.request;
    const language = entry.language;
    entry.loading = true; entry.error = ""; entry.samples = null; render(); notify();
    try {
      let samples = entry.cache.get(language);
      if (!samples) {
        const response = await fetch(`/api/benchmarks/${encodeURIComponent(entry.name)}/samples?language=${language}`);
        const data = await response.json();
        if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "无法加载 Benchmark");
        samples = data.samples;
        entry.cache.set(language, samples);
      }
      if (disposed || request !== entry.request) return;
      entry.samples = samples;
      entry.sample_count = samples.length;
      const valid = new Set(samples.map((sample) => sample.id));
      entry.ids = entry.ids === null ? valid : new Set([...entry.ids].filter((id) => valid.has(id)));
    } catch (error) {
      if (!disposed && request === entry.request) entry.error = error.message;
    } finally {
      if (!disposed && request === entry.request) { entry.loading = false; render(); notify(); }
    }
  }

  function filtered() {
    const query = $("#sampleSearch").value.trim().toLowerCase();
    const task = $("#sampleTaskFilter").value;
    return (dialogEntry?.samples || []).filter((sample) => (!task || sample.task_type === task) && (!query || `${sample.id} ${sample.prompt}`.toLowerCase().includes(query)));
  }

  function renderSamples() {
    const visible = filtered();
    const pages = Math.max(1, Math.ceil(visible.length / pageSize));
    page = Math.min(page, pages);
    const rows = visible.slice((page - 1) * pageSize, page * pageSize);
    $("#sampleList").innerHTML = rows.length ? rows.map((sample) => `<label class="sample-item">
      <input type="checkbox" data-sample-id="${escapeHtml(sample.id)}" ${draft.has(sample.id) ? "checked" : ""} />
      <span class="sample-copy"><strong>${escapeHtml(sample.id)}</strong><p>${escapeHtml(sample.prompt || "无 Prompt")}</p>${(sample.input_image_urls || []).length ? `<span class="sample-thumbnails">${sample.input_image_urls.slice(0, 3).map((url) => `<img src="${escapeHtml(url)}" loading="lazy" alt="参考图" />`).join("")}</span>` : ""}</span>
      <span class="task-tag">${escapeHtml(sample.task_type)}</span></label>`).join("") : '<div class="empty-state">没有匹配的样本</div>';
    $("#sampleCount").textContent = `已选 ${draft.size} / ${dialogEntry?.samples?.length || 0}`;
    $("#samplePageLabel").textContent = `${page} / ${pages}`;
    $("#prevSamplesBtn").disabled = page <= 1;
    $("#nextSamplesBtn").disabled = page >= pages;
    $("#sampleFilterCount").textContent = `${visible.length} 个样本`;
  }

  async function openFilter(entry) {
    if (disabled) return;
    dialogEntry = entry;
    $("#benchmarkDialogTitle").textContent = entry.name;
    $("#sampleSearch").value = ""; $("#sampleTaskFilter").value = ""; page = 1;
    $("#sampleList").innerHTML = '<div class="empty-state">加载中…</div>';
    $("#sampleCount").textContent = ""; $("#sampleFilterCount").textContent = "";
    $("#applyBenchmarkSamplesBtn").disabled = true;
    $("#benchmarkFilterError").textContent = "";
    $("#benchmarkSamplesDialog").showModal();
    await load(entry);
    if (disposed || disabled || dialogEntry !== entry || !$("#benchmarkSamplesDialog").open) return;
    if (entry.error) { $("#benchmarkFilterError").textContent = entry.error; return; }
    draft = new Set(entry.ids || []); renderSamples();
    $("#applyBenchmarkSamplesBtn").disabled = false;
  }

  $("#benchmarkList").addEventListener("change", (event) => {
    if (disabled) return;
    if (event.target.hasAttribute("data-benchmark-check")) {
      const entry = entries[Number(event.target.dataset.benchmarkCheck)];
      entry.checked = event.target.checked;
      render(); notify();
      if (entry.checked && (!entry.samples || entry.error)) load(entry);
    } else if (event.target.hasAttribute("data-benchmark-language")) {
      const entry = entries[Number(event.target.dataset.benchmarkLanguage)];
      entry.language = event.target.value;
      load(entry);
    }
  });
  $("#benchmarkList").addEventListener("click", (event) => {
    const button = event.target.closest("[data-benchmark-filter]");
    if (button) openFilter(entries[Number(button.dataset.benchmarkFilter)]);
  });
  $("#sampleSearch").addEventListener("input", () => { page = 1; renderSamples(); });
  $("#sampleTaskFilter").addEventListener("change", () => { page = 1; renderSamples(); });
  $("#sampleList").addEventListener("change", (event) => {
    const id = event.target.dataset.sampleId;
    if (!id) return;
    event.target.checked ? draft.add(id) : draft.delete(id);
    $("#sampleCount").textContent = `已选 ${draft.size} / ${dialogEntry?.samples?.length || 0}`;
  });
  $("#allBenchmarkSamplesBtn").addEventListener("click", () => { filtered().forEach((sample) => draft.add(sample.id)); renderSamples(); });
  $("#clearBenchmarkSamplesBtn").addEventListener("click", () => { filtered().forEach((sample) => draft.delete(sample.id)); renderSamples(); });
  $("#prevSamplesBtn").addEventListener("click", () => { if (page > 1) { page -= 1; renderSamples(); $("#sampleList").scrollTop = 0; } });
  $("#nextSamplesBtn").addEventListener("click", () => { page += 1; renderSamples(); $("#sampleList").scrollTop = 0; });
  $("#applyBenchmarkSamplesBtn").addEventListener("click", () => {
    if (disabled || !dialogEntry || dialogEntry.loading || dialogEntry.error) return;
    dialogEntry.ids = new Set(draft);
    $("#benchmarkSamplesDialog").close(); render(); notify();
  });
  $("#benchmarkSamplesDialog").addEventListener("close", () => { dialogEntry = null; });
  render();
  // Start after the caller has received the controller, so change callbacks are safe.
  const ready = Promise.resolve().then(() => Promise.all(entries.filter((entry) => entry.checked).map(load)));
  return {
    ready,
    selectedSamples() { return entries.filter((entry) => entry.checked).flatMap((entry) => (entry.samples || []).filter((sample) => entry.ids?.has(sample.id))); },
    count() { return entries.filter((entry) => entry.checked).reduce((sum, entry) => sum + selectedCount(entry), 0); },
    blockers() { return entries.filter((entry) => entry.checked).flatMap((entry) => entry.loading ? [`正在加载 ${entry.name}…`] : entry.error ? [`${entry.name}: ${entry.error}`] : []); },
    selections() { return entries.filter((entry) => entry.checked).map((entry) => ({ name: entry.name, language: entry.language, sample_ids: entry.ids === null ? null : [...entry.ids] })); },
    snapshot() { return entries.map((entry) => ({ name: entry.name, enabled: entry.checked, language: entry.language, sample_ids: entry.ids === null ? null : [...entry.ids] })); },
    setDisabled(value) { if (disabled !== value) { disabled = value; if (disabled && $("#benchmarkSamplesDialog").open) $("#benchmarkSamplesDialog").close(); render(); } },
    dispose() { disposed = true; entries.forEach((entry) => { entry.request += 1; }); },
  };
}
