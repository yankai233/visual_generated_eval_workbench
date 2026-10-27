import { mountAuxiliaryWindow } from "./auxiliary-window.js?v=4";

const $ = (selector) => document.querySelector(selector);
const escapeHtml = (value) => String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
const state = { catalog: [], selected: new Set(), page: 1, total: 0, rows: [], runs: [], request: 0 };

async function api(url) {
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || "请求失败");
  return payload;
}

function frameMarkup(row) {
  const box = row.input_box;
  if (!Array.isArray(box) || box.length !== 4 || !row.width || !row.height) return "";
  const left = 100 * box[0] / row.width;
  const top = 100 * box[1] / row.height;
  const width = 100 * (box[2] - box[0]) / row.width;
  const height = 100 * (box[3] - box[1]) / row.height;
  return `<span class="known-frame" style="left:${left}%;top:${top}%;width:${width}%;height:${height}%"></span>`;
}

function generatedCell(row, urls) {
  if (!urls.length) return '<div class="cell-missing">—</div>';
  return `<div class="generated-images${urls.length === 1 ? " single" : ""}">${urls.map((url, index) => `<button class="outpaint-image" type="button" data-image="${escapeHtml(url)}" data-row="${escapeHtml(row.sample_id)}" data-index="${index}" style="aspect-ratio:${row.width}/${row.height}"><img src="${escapeHtml(url)}" loading="lazy" alt="样本 ${escapeHtml(row.sample_id)} 扩图结果 ${index + 1}" />${frameMarkup(row)}</button>`).join("")}</div>`;
}

function renderMatrix() {
  if (!state.rows.length || !state.runs.length) {
    $("#matrixViewport").innerHTML = '<div class="empty-state">暂无匹配结果</div>';
    return;
  }
  $("#matrixViewport").innerHTML = `<table class="outpaint-table" style="--method-count:${state.runs.length}"><colgroup><col />${state.runs.map(() => "<col />").join("")}</colgroup><thead><tr><th>输入图 + GT + 文本</th>${state.runs.map((run) => `<th>${escapeHtml(run.label)}</th>`).join("")}</tr></thead><tbody>${state.rows.map((row) => `<tr><td><div class="sample-head"><strong>样本 ${escapeHtml(row.sample_id)}</strong><span>${row.width} × ${row.height}</span></div><div class="source-pair">${row.input_url ? `<figure><figcaption>输入图</figcaption><img class="sample-input" src="${escapeHtml(row.input_url)}" loading="lazy" alt="样本 ${escapeHtml(row.sample_id)} 输入图" /></figure>` : ""}${row.target_url ? `<figure><figcaption>GT</figcaption><span class="target-image" style="aspect-ratio:${row.width}/${row.height}"><img src="${escapeHtml(row.target_url)}" loading="lazy" alt="样本 ${escapeHtml(row.sample_id)} GT" />${frameMarkup(row)}</span></figure>` : ""}</div><p class="sample-prompt">${escapeHtml(row.prompt)}</p></td>${row.cells.map((urls) => `<td>${generatedCell(row, urls)}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}

async function loadView() {
  const request = ++state.request;
  $("#pageError").classList.add("hidden");
  const params = new URLSearchParams({ query: $("#queryInput").value, page: state.page, page_size: $("#pageSizeSelect").value });
  for (const id of state.selected) params.append("runs", id);
  try {
    const payload = await api(`/api/outpainting-comparison/view?${params}`);
    if (request !== state.request) return;
    state.rows = payload.rows; state.runs = payload.runs; state.total = payload.total; state.page = payload.page;
    renderMatrix();
    const pages = Math.max(1, Math.ceil(state.total / Number($("#pageSizeSelect").value)));
    $("#rowCount").textContent = `${state.total} 个样本`;
    $("#pageLabel").textContent = `${state.page} / ${pages}`;
    $("#prevBtn").disabled = state.page <= 1;
    $("#nextBtn").disabled = state.page >= pages;
  } catch (error) {
    $("#pageError").textContent = error.message; $("#pageError").classList.remove("hidden");
  }
}

function renderChoices() {
  $("#runChoices").innerHTML = state.catalog.length ? state.catalog.map((run) => `<label class="run-choice"><input type="checkbox" value="${escapeHtml(run.id)}" ${state.selected.has(run.id) ? "checked" : ""}/><span>${escapeHtml(run.label)}</span><b>${run.sample_count}</b></label>`).join("") : '<span>未发现扩图评测结果</span>';
}

async function refresh(force = false) {
  const payload = await api(`/api/outpainting-comparison/catalog${force ? "?refresh=true" : ""}`);
  const previous = new Set(state.selected);
  state.catalog = payload.runs;
  state.selected = new Set(state.catalog.filter((run) => previous.size ? previous.has(run.id) : true).map((run) => run.id));
  renderChoices(); state.page = 1; await loadView();
}

$("#runChoices").addEventListener("change", (event) => {
  const id = event.target.value;
  event.target.checked ? state.selected.add(id) : state.selected.delete(id);
  state.page = 1; loadView();
});
let queryTimer;
$("#queryInput").addEventListener("input", () => { clearTimeout(queryTimer); queryTimer = setTimeout(() => { state.page = 1; loadView(); }, 180); });
$("#pageSizeSelect").addEventListener("change", () => { state.page = 1; loadView(); });
$("#refreshBtn").addEventListener("click", () => refresh(true));
$("#prevBtn").addEventListener("click", () => { state.page -= 1; loadView(); });
$("#nextBtn").addEventListener("click", () => { state.page += 1; loadView(); });
$("#matrixViewport").addEventListener("click", (event) => {
  const button = event.target.closest("[data-image]");
  if (!button) return;
  $("#largeImage").src = button.dataset.image;
  const row = state.rows.find((item) => item.sample_id === button.dataset.row);
  $("#largeImageStage").querySelector(".known-frame")?.remove();
  if (row) $("#largeImageStage").insertAdjacentHTML("beforeend", frameMarkup(row));
  $("#imageDialog").showModal();
});
$("#closeImageBtn").addEventListener("click", () => $("#imageDialog").close());
$("#imageDialog").addEventListener("click", (event) => { if (event.target === $("#imageDialog")) $("#imageDialog").close(); });

mountAuxiliaryWindow();
refresh().catch((error) => { $("#pageError").textContent = error.message; $("#pageError").classList.remove("hidden"); });
