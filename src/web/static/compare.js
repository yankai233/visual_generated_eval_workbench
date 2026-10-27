import { mountAuxiliaryWindow } from "./auxiliary-window.js?v=4";

const $ = (selector) => document.querySelector(selector);
const escapeHtml = (value) => String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
const state = { catalog: { sources: [], pipeline_labels: {} }, columns: [], rows: [], page: 1, total: 0, selected: null, samples: [], sampleDraft: new Set(), groupRecords: [], groupDraft: new Set(), requestId: 0 };
let columnSequence = 0;

async function api(url, payload) {
  const response = await fetch(url, payload === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "请求失败");
  return data;
}

function showError(error) {
  $("#compareError").textContent = error?.message || String(error);
  $("#compareError").classList.remove("hidden");
}

function currentSource() {
  return state.catalog.sources.find((source) => source.id === $("#sourceSelect").value);
}

function label(pipeline) {
  return state.catalog.pipeline_labels[pipeline] || pipeline;
}

function dateLabel(value) {
  return String(value).replace(/(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2}).*/, "$1-$2-$3 $4:$5:$6");
}

function durationLabel(value) {
  const seconds = Number(value);
  if (!Number.isFinite(seconds) || seconds < 0) return "未记录";
  if (seconds >= 60) {
    const minutes = Math.floor(seconds / 60);
    return `${minutes}分 ${String((seconds % 60).toFixed(1)).padStart(4, "0")}秒`;
  }
  return `${seconds.toFixed(seconds >= 10 ? 1 : 2)} 秒`;
}

function methods() {
  return currentSource()?.methods || [];
}

function renderColumns() {
  $("#methodColumns").innerHTML = state.columns.map((column, index) => {
    const method = methods().find((item) => item.pipeline === column.pipeline);
    const options = methods().map((item) => `<option value="${escapeHtml(item.pipeline)}" ${column.pipeline === item.pipeline ? "selected" : ""}>${escapeHtml(label(item.pipeline))}</option>`).join("");
    const configs = (method?.configs || []).map((config) => `<option value="${escapeHtml(config)}" ${column.config_name === config ? "selected" : ""}>${escapeHtml(config)}</option>`).join("");
    const runs = (method?.runs || []).filter((run) => !column.config_name || run.config_name === column.config_name);
    if (column.run_id && !runs.some((run) => run.id === column.run_id)) column.run_id = null;
    return `<div class="method-column" data-column="${index}"><header>方法 ${index + 1}<button data-remove-column aria-label="移除方法" ${state.columns.length === 1 ? "disabled" : ""}>×</button></header>
      <select data-method-field="pipeline" aria-label="Pipeline">${options}</select>
      <select data-method-field="config_name" aria-label="配置"><option value="">全部配置</option>${configs}</select>
      <select data-method-field="run_id" aria-label="生成批次"><option value="">最新结果（按样本）</option>${runs.map((run) => `<option value="${escapeHtml(run.id)}" ${column.run_id === run.id ? "selected" : ""}>${escapeHtml(dateLabel(run.batch_started_at) + " · " + run.config_name + " · " + run.id)}</option>`).join("")}</select></div>`;
  }).join("");
  $("#addMethodBtn").disabled = !methods().length || state.columns.length >= 8;
}

function imageButtons(urls) {
  return `<div class="reference-images">${urls.map((url, index) => `<button data-full-image="${escapeHtml(url)}" title="参考图 ${index + 1}"><img src="${escapeHtml(url)}" loading="lazy" alt="参考图 ${index + 1}" /></button>`).join("")}</div>`;
}

function renderCell(rowIndex, columnIndex, version = 0, imageIndex = 0) {
  const records = state.rows[rowIndex].cells[columnIndex];
  if (!records.length) return '<div class="cell-missing">—</div>';
  const record = records[version];
  const url = record.image_urls[imageIndex];
  return `<button class="cell-image-button" data-full-image="${escapeHtml(url)}"><img src="${escapeHtml(url)}" alt="${escapeHtml(label(record.pipeline) + " · " + record.label)}" loading="lazy" /></button>
    <div class="cell-controls">
      ${records.length > 1 ? `<select data-cell-version aria-label="历史结果">${records.map((item, index) => `<option value="${index}" ${index === version ? "selected" : ""}>${escapeHtml(dateLabel(item.batch_started_at) + " · " + item.config_name + " · " + item.run_id)}</option>`).join("")}</select>` : ""}
      ${record.image_urls.length > 1 ? `<select data-cell-image data-version="${version}" aria-label="生成图像">${record.image_urls.map((_, index) => `<option value="${index}" ${index === imageIndex ? "selected" : ""}>图 ${index + 1} / ${record.image_urls.length}</option>`).join("")}</select>` : ""}
    </div>
    <details class="cell-details"><summary>详情</summary><p>${escapeHtml(dateLabel(record.batch_started_at))}<br />${escapeHtml(record.config_name)} · seed ${escapeHtml(record.seed ?? "随机")} · ${escapeHtml(record.width)}×${escapeHtml(record.height)} · ${escapeHtml(record.num_inference_steps)} steps<br />生成耗时 ${escapeHtml(durationLabel(record.generation_duration_seconds))}</p><p>${escapeHtml(record.prompt)}</p>${record.negative_prompt ? `<p>Negative: ${escapeHtml(record.negative_prompt)}</p>` : ""}${imageButtons(record.reference_urls)}</details>`;
}

function renderMatrix() {
  if (!state.rows.length) {
    $("#matrixViewport").innerHTML = '<div class="compare-empty">暂无匹配的生成结果</div>';
    return;
  }
  $("#matrixViewport").innerHTML = `<table class="comparison-table" style="--method-count: ${state.columns.length}"><colgroup><col class="sample-table-col" />${state.columns.map(() => '<col class="method-table-col" />').join("")}</colgroup><thead><tr><th>样本</th>${state.columns.map((column) => `<th>${escapeHtml(label(column.pipeline))}</th>`).join("")}</tr></thead><tbody>${state.rows.map((row, rowIndex) => `<tr>
    <td class="sample-column"><strong>${escapeHtml(row.label)}</strong><div class="row-tags"><span>${escapeHtml(row.task_type)}</span>${row.match_kind === "unresolved" ? '<span class="warning-tag">待分组</span>' : ""}${row.match_kind === "manual" ? '<span>手动分组</span>' : ""}${row.different_inputs ? '<span class="warning-tag" title="这些结果的输入内容不同，可展开各单元格详情核对">输入不同</span>' : ""}</div><p class="sample-prompt">${escapeHtml(row.prompt)}</p>${imageButtons(row.reference_urls)}</td>
    ${state.columns.map((_, columnIndex) => `<td data-row="${rowIndex}" data-cell-column="${columnIndex}">${renderCell(rowIndex, columnIndex)}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}

async function loadView() {
  const requestId = ++state.requestId;
  $("#compareError").classList.add("hidden");
  try {
    const source = currentSource();
    let data = { rows: [], total: 0, page: 1 };
    if (source && state.columns.length && !(state.selected instanceof Set && state.selected.size === 0)) {
      data = await api("/api/comparison/view", {
        source: source.id, columns: state.columns.map(({ pipeline, config_name, run_id }) => ({ pipeline, config_name, run_id })),
        sample_ids: state.selected === null ? [] : [...state.selected], query: $("#queryInput").value,
        only_common: $("#onlyCommonInput").checked, page: state.page, page_size: Number($("#pageSizeSelect").value),
      });
    }
    if (requestId !== state.requestId) return;
    state.rows = data.rows; state.total = data.total; state.page = data.page;
    renderMatrix();
    const pages = Math.max(1, Math.ceil(state.total / Number($("#pageSizeSelect").value)));
    $("#rowCount").textContent = `${state.total} 个样本`;
    $("#pageLabel").textContent = `${state.page} / ${pages}`;
    $("#prevPageBtn").disabled = state.page <= 1;
    $("#nextPageBtn").disabled = state.page >= pages;
  } catch (error) {
    if (requestId === state.requestId) showError(error);
  }
}

function selectSource() {
  state.selected = null; state.page = 1;
  state.columns = methods().slice(0, 2).map((method) => ({ id: ++columnSequence, pipeline: method.pipeline, config_name: null, run_id: null }));
  $("#groupsBtn").disabled = $("#sourceSelect").value !== "users";
  $("#samplesBtn").disabled = !currentSource();
  renderColumns();
  return loadView();
}

async function refresh(keep = true) {
  const previous = $("#sourceSelect").value;
  state.catalog = await api("/api/comparison/catalog");
  $("#sourceSelect").innerHTML = state.catalog.sources.length
    ? state.catalog.sources.map((source) => `<option value="${escapeHtml(source.id)}">${escapeHtml(source.label)} · ${source.sample_count}</option>`).join("")
    : '<option value="">暂无生成结果</option>';
  if (keep && state.catalog.sources.some((source) => source.id === previous)) {
    $("#sourceSelect").value = previous;
    const valid = new Set(methods().map((method) => method.pipeline));
    state.columns = state.columns.filter((column) => valid.has(column.pipeline));
    if (state.columns.length) { renderColumns(); await loadView(); return; }
  }
  await selectSource();
}

function renderSampleChoices() {
  const query = $("#sampleFilter").value.toLowerCase();
  $("#sampleChoices").innerHTML = state.samples.filter((sample) => `${sample.label} ${sample.prompt}`.toLowerCase().includes(query)).map((sample) => `<label class="compare-choice"><input type="checkbox" data-sample-key="${escapeHtml(sample.case_id)}" ${state.sampleDraft.has(sample.case_id) ? "checked" : ""} /><span><strong>${escapeHtml(sample.label)}</strong><p>${escapeHtml(sample.prompt)}</p></span></label>`).join("");
  $("#sampleSelectionCount").textContent = `${state.sampleDraft.size} / ${state.samples.length}`;
}

async function openSamples() {
  try {
    const data = await api(`/api/comparison/samples?source=${encodeURIComponent(currentSource().id)}`);
    state.samples = data.samples;
    state.sampleDraft = new Set(state.selected === null ? state.samples.map((sample) => sample.case_id) : state.selected);
    $("#sampleFilter").value = ""; renderSampleChoices(); $("#samplesDialog").showModal();
  } catch (error) { showError(error); }
}

function renderGroupChoices() {
  const query = $("#groupFilter").value.toLowerCase();
  $("#groupChoices").innerHTML = state.groupRecords.filter((record) => `${record.pipeline} ${record.prompt} ${record.label}`.toLowerCase().includes(query)).map((record) => `<label class="compare-choice"><input type="checkbox" data-result-key="${escapeHtml(record.result_id)}" ${state.groupDraft.has(record.result_id) ? "checked" : ""} /><img src="${escapeHtml(record.image_urls[0])}" loading="lazy" alt="结果缩略图" /><span><strong>${escapeHtml(label(record.pipeline))} · ${escapeHtml(record.label)}</strong><p>${escapeHtml(record.prompt)}</p><p>${escapeHtml(dateLabel(record.batch_started_at))} · ${escapeHtml(record.config_name)}</p></span></label>`).join("");
}

async function saveGroup(reset) {
  $("#groupError").textContent = "";
  try {
    await api("/api/comparison/group", { result_ids: [...state.groupDraft], name: reset ? null : $("#groupNameInput").value });
    $("#groupsDialog").close(); state.selected = null; state.page = 1; await refresh();
  } catch (error) { $("#groupError").textContent = error.message; }
}

$("#sourceSelect").addEventListener("change", selectSource);
$("#addMethodBtn").addEventListener("click", () => {
  if (state.columns.length >= 8 || !methods().length) return;
  const unused = methods().find((method) => !state.columns.some((column) => column.pipeline === method.pipeline)) || methods()[0];
  state.columns.push({ id: ++columnSequence, pipeline: unused.pipeline, config_name: null, run_id: null });
  renderColumns(); loadView();
});
$("#methodColumns").addEventListener("change", (event) => {
  const name = event.target.dataset.methodField;
  if (!name) return;
  const column = state.columns[Number(event.target.closest("[data-column]").dataset.column)];
  column[name] = event.target.value || null;
  if (name === "pipeline") { column.config_name = null; column.run_id = null; }
  if (name === "config_name") column.run_id = null;
  state.page = 1; renderColumns(); loadView();
});
$("#methodColumns").addEventListener("click", (event) => {
  if (!event.target.closest("[data-remove-column]") || state.columns.length <= 1) return;
  state.columns.splice(Number(event.target.closest("[data-column]").dataset.column), 1);
  renderColumns(); loadView();
});
$("#matrixViewport").addEventListener("change", (event) => {
  const cell = event.target.closest("[data-row]");
  if (!cell) return;
  const version = event.target.hasAttribute("data-cell-version") ? Number(event.target.value) : Number(event.target.dataset.version);
  const index = event.target.hasAttribute("data-cell-image") ? Number(event.target.value) : 0;
  cell.innerHTML = renderCell(Number(cell.dataset.row), Number(cell.dataset.cellColumn), version, index);
});
document.addEventListener("click", (event) => {
  const target = event.target.closest("[data-full-image]");
  if (!target) return;
  $("#largeImage").src = target.dataset.fullImage; $("#imageDialog").showModal();
});
$("#closeImageBtn").addEventListener("click", () => $("#imageDialog").close());
$("#refreshBtn").addEventListener("click", () => refresh().catch(showError));
$("#samplesBtn").addEventListener("click", openSamples);
$("#sampleFilter").addEventListener("input", renderSampleChoices);
$("#sampleChoices").addEventListener("change", (event) => {
  const key = event.target.dataset.sampleKey;
  if (!key) return;
  event.target.checked ? state.sampleDraft.add(key) : state.sampleDraft.delete(key);
  $("#sampleSelectionCount").textContent = `${state.sampleDraft.size} / ${state.samples.length}`;
});
$("#allSamplesBtn").addEventListener("click", () => { state.sampleDraft = new Set(state.samples.map((sample) => sample.case_id)); renderSampleChoices(); });
$("#noSamplesBtn").addEventListener("click", () => { state.sampleDraft.clear(); renderSampleChoices(); });
$("#applySamplesBtn").addEventListener("click", () => { state.selected = new Set(state.sampleDraft); state.page = 1; $("#samplesDialog").close(); loadView(); });
$("#groupsBtn").addEventListener("click", async () => {
  try {
    state.groupRecords = (await api("/api/comparison/records?source=users")).records;
    state.groupDraft.clear(); $("#groupNameInput").value = ""; $("#groupFilter").value = ""; $("#groupError").textContent = "";
    renderGroupChoices(); $("#groupsDialog").showModal();
  } catch (error) { showError(error); }
});
$("#groupFilter").addEventListener("input", renderGroupChoices);
$("#groupChoices").addEventListener("change", (event) => {
  const key = event.target.dataset.resultKey;
  if (key) event.target.checked ? state.groupDraft.add(key) : state.groupDraft.delete(key);
});
$("#applyGroupsBtn").addEventListener("click", () => saveGroup(false));
$("#resetGroupsBtn").addEventListener("click", () => saveGroup(true));
let searchTimer;
$("#queryInput").addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => { state.page = 1; loadView(); }, 200); });
$("#onlyCommonInput").addEventListener("change", () => { state.page = 1; loadView(); });
$("#pageSizeSelect").addEventListener("change", () => { state.page = 1; loadView(); });
$("#prevPageBtn").addEventListener("click", () => { if (state.page > 1) { state.page -= 1; loadView(); } });
$("#nextPageBtn").addEventListener("click", () => { state.page += 1; loadView(); });
mountAuxiliaryWindow();
try { await refresh(false); } catch (error) { showError(error); $("#matrixViewport").innerHTML = '<div class="compare-empty">加载失败，请刷新</div>'; }
