import { mountWorkspace } from "./workspace.js?v=20261024-2";
import { mountPipelineEditor } from "./pipeline-editor.js?v=2";
import { mountGpuMonitor } from "./gpu-monitor.js?v=1";
import { mountAuxiliaryWindow } from "./auxiliary-window.js?v=4";

const workspaces = new Map();
const sessionKey = "visual-gen-eval-workspaces-v1";
let bootstrap;
let nextNumber = 1;
let controlling = false;
const $ = (selector) => document.querySelector(selector);
const terminal = new Set(["completed", "completed_with_errors", "cancelled", "failed", "interrupted"]);
mountGpuMonitor();
mountAuxiliaryWindow();

function escapeHtml(value) {
  return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
}

function durationLabel(value) {
  const seconds = Number(value);
  if (seconds >= 60) return `${Math.floor(seconds / 60)}分 ${String((seconds % 60).toFixed(1)).padStart(4, "0")}秒`;
  return `${seconds.toFixed(seconds >= 10 ? 1 : 2)} 秒`;
}

function renderTimingSummary(catalog) {
  const rows = catalog.timing_summary || [];
  const sources = (catalog.sources || []).filter((source) => source.id.startsWith("benchmark:"));
  const pipelineLabel = (name) => catalog.pipeline_labels?.[name] || bootstrap?.pipelines?.find((item) => item.name === name)?.display_name || name;
  const benchmarks = [...new Set([...sources.map((source) => source.id.slice("benchmark:".length)), ...rows.map((row) => row.benchmark)])]
    .sort((left, right) => left.localeCompare(right, "zh-CN", { numeric: true }));
  const pipelines = [...new Set([...sources.flatMap((source) => source.methods.map((method) => method.pipeline)), ...rows.map((row) => row.pipeline)])]
    .sort((left, right) => pipelineLabel(left).localeCompare(pipelineLabel(right), "zh-CN"));
  const values = new Map(rows.map((row) => [`${row.pipeline}\u0000${row.benchmark}`, row.average_generation_seconds]));
  $("#timingTableHead").innerHTML = `<tr><th>模型</th>${benchmarks.map((benchmark) => `<th>${escapeHtml(benchmark)}</th>`).join("")}</tr>`;
  $("#timingTableBody").innerHTML = pipelines.length
    ? pipelines.map((pipeline) => `<tr><td>${escapeHtml(pipelineLabel(pipeline))}</td>${benchmarks.map((benchmark) => {
      const key = `${pipeline}\u0000${benchmark}`;
      const value = values.get(key);
      return typeof value === "number" && Number.isFinite(value) && value >= 0
        ? `<td>${escapeHtml(durationLabel(value))}</td>` : '<td class="timing-missing">-</td>';
    }).join("")}</tr>`).join("")
    : `<tr><td colspan="${benchmarks.length + 1}" class="timing-empty">暂无耗时记录</td></tr>`;
}

async function openTimingSummary() {
  const button = $("#timingBtn");
  button.disabled = true;
  $("#timingDialog").showModal();
  $("#timingTableHead").innerHTML = "<tr><th>模型</th></tr>";
  $("#timingTableBody").innerHTML = '<tr><td class="timing-empty">加载中</td></tr>';
  try {
    const response = await fetch("/api/comparison/catalog", { cache: "no-store" });
    if (!response.ok) throw new Error(`加载失败 (${response.status})`);
    renderTimingSummary(await response.json());
  } catch (error) {
    $("#timingTableBody").innerHTML = `<tr><td class="timing-empty">${escapeHtml(error.message)}</td></tr>`;
  } finally {
    button.disabled = false;
  }
}

$("#timingBtn").addEventListener("click", openTimingSummary);

function saveSessions() {
  try {
    sessionStorage.setItem(sessionKey, JSON.stringify(
      [...workspaces.values()].map((item) => item.controller.snapshot()).filter(Boolean)
    ));
  } catch { /* Browser storage is optional. */ }
  updateSharedOutputs();
}

function updateSharedOutputs() {
  const models = [...workspaces.entries()].map(([id, item]) => ({ id, ...item.controller.outputState() }));
  const jobs = models.filter((item) => item.status);
  $("#sharedProgress").classList.toggle("hidden", !jobs.length);
  $("#sharedResults").classList.toggle("hidden", !models.some((item) => item.hasResults));
  const select = $("#controlModel");
  const chosen = select.value;
  const options = [["all", "全部模型"], ...jobs.map((item) => [item.id, item.label])];
  if (JSON.stringify(options) !== select.dataset.options) {
    select.replaceChildren(...options.map(([value, label]) => new Option(label, value)));
    select.dataset.options = JSON.stringify(options);
    select.value = options.some(([value]) => value === chosen) ? chosen : "all";
  }
  select.hidden = jobs.length < 2;
  const targets = jobs.filter((item) => select.value === "all" || item.id === select.value);
  $("#pauseAllBtn").disabled = controlling || !targets.some((item) => item.status === "running");
  $("#resumeAllBtn").disabled = controlling || !targets.some((item) => item.status === "paused");
  $("#cancelAllBtn").disabled = controlling || !targets.some((item) => !terminal.has(item.status) && item.status !== "cancelling");
}

const editor = mountPipelineEditor(async (pipeline, workspaceId) => {
  const index = bootstrap.pipelines.findIndex((item) => item.name === pipeline.name);
  if (index < 0) bootstrap.pipelines.push(pipeline);
  else bootstrap.pipelines[index] = pipeline;
  for (const [id, workspace] of workspaces) {
    workspace.controller.updatePipelines(bootstrap.pipelines, id === workspaceId ? pipeline.name : null);
  }
  saveSessions();
});

function addWorkspace(initial = {}) {
  const id = initial.id || globalThis.crypto?.randomUUID?.() || `workspace-${Date.now()}-${nextNumber}`;
  const number = initial.number || nextNumber;
  nextNumber = Math.max(nextNumber, number + 1);
  const host = document.createElement("pipeline-workspace");
  const outputHost = document.createElement("pipeline-output");
  const resultsHost = document.createElement("model-results");
  for (const element of [host, outputHost, resultsHost]) element.dataset.workspaceId = id;
  host.setAttribute("aria-label", `Model ${number}`);
  outputHost.setAttribute("aria-label", `Model ${number} 进度`);
  resultsHost.setAttribute("aria-label", `Model ${number} 结果`);
  outputHost.hidden = true;
  resultsHost.hidden = true;
  $("#workspaces").append(host);
  $("#workspaceOutputs").append(outputHost);
  $("#modelResults").append(resultsHost);
  const controller = mountWorkspace(host, bootstrap, { ...initial, id, number }, {
    changed: saveSessions,
    outputChanged: updateSharedOutputs,
    newPipeline: () => editor.open(id),
    remove() {
      workspaces.get(id)?.controller.dispose();
      workspaces.delete(id);
      host.remove();
      outputHost.remove();
      resultsHost.remove();
      saveSessions();
    },
  }, outputHost, resultsHost);
  workspaces.set(id, { host, outputHost, resultsHost, controller });
  controller.ready.then(saveSessions);
  saveSessions();
  return host;
}

for (const action of ["pause", "resume", "cancel"]) {
  $(`#${action}AllBtn`).addEventListener("click", async () => {
    const selected = $("#controlModel").value;
    controlling = true;
    updateSharedOutputs();
    try {
      await Promise.all([...workspaces.entries()]
        .filter(([id]) => selected === "all" || selected === id)
        .map(([, item]) => item.controller.control(action)));
    } finally {
      controlling = false;
      updateSharedOutputs();
    }
  });
}
$("#controlModel").addEventListener("change", updateSharedOutputs);

async function initialize() {
  const addButton = $("#addModelBtn");
  addButton.disabled = true;
  try {
    const response = await fetch("/api/bootstrap");
    if (!response.ok) throw new Error(`加载工作台失败 (${response.status})`);
    bootstrap = await response.json();
    const status = $("#systemStatus");
    status.classList.add("online");
    status.title = "服务可用";
    status.setAttribute("aria-label", "服务可用");
    $("#serviceStatusText").textContent = "服务可用";
    let restored;
    try { restored = JSON.parse(sessionStorage.getItem(sessionKey)); } catch { /* Fresh workspace. */ }
    if (Array.isArray(restored)) restored.filter((row) => row && typeof row === "object").forEach(addWorkspace);
    else {
      const latestJob = bootstrap.jobs?.[0];
      const pipeline = latestJob?.results?.[0]?.unit_id?.split(":")[0];
      addWorkspace({ pipeline, jobId: latestJob?.id });
    }
    addButton.addEventListener("click", async () => {
      addButton.disabled = true;
      try {
        const previous = [...workspaces.values()].at(-1)?.controller;
        if (previous) await previous.ready;
        addWorkspace(previous?.cloneInputs() || {});
      } finally { addButton.disabled = false; }
    });
    addButton.disabled = false;
    if (bootstrap.pipeline_errors?.length) throw new Error("部分 Pipeline 加载失败：" + bootstrap.pipeline_errors.join("；"));
  } catch (error) {
    if (!bootstrap) {
      $("#serviceStatusText").textContent = "服务不可用";
      $("#systemStatus").classList.add("offline");
      $("#systemStatus").title = "服务不可用";
      $("#systemStatus").setAttribute("aria-label", "服务不可用");
    }
    $("#appError").textContent = error.message;
    $("#appError").classList.remove("hidden");
  }
}

await initialize();
