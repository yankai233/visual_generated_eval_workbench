const $ = (selector) => document.querySelector(selector);
if (new URLSearchParams(location.search).get("embedded") === "1") document.documentElement.classList.add("embedded-tools");
const draftKey = "visual-gen-eval-markdown-draft-v1";
const translationDraftKey = "visual-gen-eval-translation-draft-v1";
const activeToolKey = "visual-gen-eval-active-tool-v1";
let translationMode = "web";
let translating = false;
let translationVersion = 0;
let translationTimer;
let translationController;
let openedName = "";
let revision = null;
let savedContent = "";
let previewVersion = 0;
let previewTimer;
let saving = false;
let documentVersion = 0;

async function api(path, payload, signal) {
  const response = await fetch(path, payload === undefined ? { signal } : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload), signal });
  const result = await response.json();
  if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "请求失败");
  return result;
}

function errorMessage(selector, message = "") {
  $(selector).textContent = message;
  $(selector).classList.toggle("hidden", !message);
}

function chooseTool(name) {
  name = name === "markdown" ? "markdown" : "translate";
  for (const button of document.querySelectorAll("[data-tool]")) {
    const selected = button.dataset.tool === name;
    button.setAttribute("aria-selected", String(selected)); button.tabIndex = selected ? 0 : -1;
  }
  $("#translationPanel").classList.toggle("hidden", name !== "translate");
  $("#markdownPanel").classList.toggle("hidden", name !== "markdown");
  writeDraft(activeToolKey, { tool: name });
  history.replaceState(null, "", `#${name}`);
  if (window.parent !== window) window.parent.postMessage({ type: "visual-eval-auxiliary-tool", tool: name }, location.origin);
}

function readDraft(key) {
  try { return JSON.parse(localStorage.getItem(key)) || {}; } catch { return {}; }
}

function writeDraft(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* Explicit file saves remain available. */ }
}

function syncTranslation(invalidate = true) {
  if (invalidate) {
    translationVersion += 1;
  }
  const text = $("#translationInput").value;
  const source = $("#sourceLanguage").value;
  const target = $("#targetLanguage").value;
  $("#swapLanguagesBtn").disabled = source === "auto";
  $("#translationCount").textContent = `${text.length} / 5000`;
  writeDraft(translationDraftKey, { text, source, target });
}

async function translate() {
  if (translating || !$("#translationInput").value.trim()) return;
  const version = ++translationVersion;
  const text = $("#translationInput").value;
  const source = $("#sourceLanguage").value;
  const target = $("#targetLanguage").value;
  translationController = new AbortController();
  translating = true; syncTranslation(false);
  $("#translationOutput").setAttribute("aria-busy", "true");
  errorMessage("#translationError");
  try {
    const result = await api("/api/tools/translate", { text, source, target }, translationController.signal);
    if (version !== translationVersion) return;
    $("#translationOutput").value = result.text;
    $("#copyTranslationBtn").disabled = !result.text;
  } catch (error) { if (error.name !== "AbortError" && version === translationVersion) errorMessage("#translationError", error.message); }
  finally {
    if (version === translationVersion) {
      translating = false;
      $("#translationOutput").removeAttribute("aria-busy");
      syncTranslation(false);
    }
  }
}

function scheduleTranslation(immediate = false) {
  clearTimeout(translationTimer);
  translationVersion += 1;
  translationController?.abort();
  translating = false;
  const available = Boolean($("#translationInput").value.trim());
  if (!available) {
    $("#translationOutput").value = "";
    $("#translationOutput").removeAttribute("aria-busy");
    $("#copyTranslationBtn").disabled = true;
    errorMessage("#translationError");
    syncTranslation(false);
    return;
  }
  syncTranslation(false);
  translationTimer = setTimeout(translate, immediate ? 0 : 260);
}

async function copyTranslation() {
  const text = $("#translationOutput");
  try {
    if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(text.value);
    else { text.focus(); text.select(); if (!document.execCommand("copy")) throw new Error("copy failed"); }
    $("#copyTranslationBtn").textContent = "已复制";
    setTimeout(() => { $("#copyTranslationBtn").textContent = "复制译文"; }, 1500);
  } catch { text.focus(); text.select(); errorMessage("#translationError", "请使用 Ctrl/Cmd+C 复制已选中的译文"); }
}

function dirty() {
  return $("#markdownEditor").value !== savedContent || Boolean($("#markdownEditor").value && $("#markdownFilename").value !== openedName);
}

function persistMarkdown() {
  writeDraft(draftKey, { content: $("#markdownEditor").value, name: $("#markdownFilename").value, openedName, revision, savedContent });
  $("#markdownStatus").textContent = dirty() ? "未保存" : openedName ? "已保存" : "未保存";
}

async function renderPreview() {
  clearTimeout(previewTimer);
  const version = ++previewVersion;
  try {
    const result = await api("/api/tools/markdown/preview", { content: $("#markdownEditor").value });
    if (version !== previewVersion) return;
    $("#markdownPreview").srcdoc = `<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:;"><style>
      body{font:14px/1.75 system-ui,sans-serif;color:#17231e;padding:18px;margin:0;overflow-wrap:anywhere}h1{font-size:28px}h2{font-size:22px}h3{font-size:18px}h1,h2{border-bottom:1px solid #e1e8e3;padding-bottom:8px}pre{background:#f4f7f5;padding:12px;border-radius:8px;overflow:auto}code{font-family:ui-monospace,Consolas,monospace;font-size:.92em}p code{background:#f0f4f1;padding:2px 4px}blockquote{border-left:3px solid #8dbba2;color:#66766e;padding-left:14px;margin-left:0}table{border-collapse:collapse;display:block;overflow:auto}th,td{border:1px solid #d6e3dc;padding:6px 10px}img{max-width:100%}a{color:#18734b}hr{border:0;border-top:1px solid #d6e3dc}
      </style></head><body>${result.html}</body></html>`;
  } catch (error) { if (version === previewVersion) errorMessage("#markdownError", error.message); }
}

async function refreshFiles() {
  const result = await api("/api/tools/markdown/files");
  const select = $("#markdownFileSelect");
  select.replaceChildren(new Option("打开已保存文件…", ""));
  result.files.forEach((file) => select.add(new Option(file.name, file.name)));
  select.value = openedName;
}

function replaceDocument(name, content, serverName = "", serverRevision = null) {
  documentVersion += 1;
  $("#markdownFilename").value = name;
  $("#markdownEditor").value = content;
  openedName = serverName; revision = serverRevision; savedContent = serverName ? content : "";
  $("#markdownFileSelect").value = serverName;
  errorMessage("#markdownError"); persistMarkdown(); renderPreview();
}

function mayDiscard() {
  return !dirty() || confirm("当前内容尚未保存到文件，确定切换吗？");
}

async function openDocument() {
  const name = $("#markdownFileSelect").value;
  if (!name) return;
  if (!mayDiscard()) { $("#markdownFileSelect").value = openedName; return; }
  const version = ++documentVersion;
  try {
    const file = await api(`/api/tools/markdown/file?name=${encodeURIComponent(name)}`);
    if (version !== documentVersion) return;
    replaceDocument(file.name, file.content, file.name, file.revision);
  } catch (error) { if (version === documentVersion) { errorMessage("#markdownError", error.message); $("#markdownFileSelect").value = openedName; } }
}

async function saveDocument() {
  if (saving) return;
  let name = $("#markdownFilename").value.trim();
  if (!name) { errorMessage("#markdownError", "请输入文件名"); $("#markdownFilename").focus(); return; }
  if (!/\.[^./\\]+$/.test(name)) name += ".md";
  if (!/\.(md|markdown)$/i.test(name)) { errorMessage("#markdownError", "文件名必须以 .md 或 .markdown 结尾"); $("#markdownFilename").focus(); return; }
  $("#markdownFilename").value = name;
  const content = $("#markdownEditor").value;
  const version = documentVersion;
  saving = true; $("#saveMarkdownBtn").disabled = true; errorMessage("#markdownError");
  try {
    const file = await api("/api/tools/markdown/file", { name, content, revision: name === openedName ? revision : null });
    if (version !== documentVersion) { await refreshFiles(); return; }
    openedName = file.name; revision = file.revision; savedContent = content;
    // Preserve any edits made while the save request was in flight.
    persistMarkdown(); await refreshFiles();
  } catch (error) { errorMessage("#markdownError", error.message); }
  finally { saving = false; $("#saveMarkdownBtn").disabled = false; }
}

function downloadDocument() {
  const url = URL.createObjectURL(new Blob([$("#markdownEditor").value], { type: "text/markdown;charset=utf-8" }));
  const anchor = document.createElement("a"); anchor.href = url;
  anchor.download = $("#markdownFilename").value.trim() || "untitled.md";
  anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}

$(".tool-tabs").addEventListener("click", (event) => { if (event.target.dataset.tool) chooseTool(event.target.dataset.tool); });
$(".tool-tabs").addEventListener("keydown", (event) => {
  if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
  event.preventDefault();
  const tool = event.key === "Home" ? "translate" : event.key === "End" ? "markdown" : $("#translateTab").getAttribute("aria-selected") === "true" ? "markdown" : "translate";
  chooseTool(tool); $(`[data-tool='${tool}']`).focus();
});
window.addEventListener("hashchange", () => chooseTool(location.hash.slice(1)));
$("#translationInput").addEventListener("input", () => scheduleTranslation());
for (const selector of ["#sourceLanguage", "#targetLanguage"]) $(selector).addEventListener("change", () => scheduleTranslation(true));
$("#copyTranslationBtn").addEventListener("click", copyTranslation);
$("#clearTranslationBtn").addEventListener("click", () => { $("#translationInput").value = ""; scheduleTranslation(true); });
$("#swapLanguagesBtn").addEventListener("click", () => {
  if ($("#sourceLanguage").value === "auto") return;
  const source = $("#sourceLanguage").value;
  $("#sourceLanguage").value = $("#targetLanguage").value; $("#targetLanguage").value = source;
  if ($("#translationOutput").value) $("#translationInput").value = $("#translationOutput").value;
  scheduleTranslation(true);
});
$("#markdownEditor").addEventListener("input", () => {
  previewVersion += 1; clearTimeout(previewTimer); persistMarkdown(); previewTimer = setTimeout(renderPreview, 220);
});
$("#markdownFilename").addEventListener("input", persistMarkdown);
$("#markdownFileSelect").addEventListener("change", openDocument);
$("#refreshFilesBtn").addEventListener("click", () => refreshFiles().catch((error) => errorMessage("#markdownError", error.message)));
$("#newMarkdownBtn").addEventListener("click", () => { if (mayDiscard()) replaceDocument("untitled.md", ""); });
$("#saveMarkdownBtn").addEventListener("click", saveDocument);
$("#downloadMarkdownBtn").addEventListener("click", downloadDocument);
$("#importMarkdownBtn").addEventListener("click", () => $("#markdownImport").click());
$("#markdownImport").addEventListener("change", async () => {
  const file = $("#markdownImport").files[0]; $("#markdownImport").value = "";
  if (!file || !mayDiscard()) return;
  if (file.size > 1024 * 1024 || !/\.(md|markdown)$/i.test(file.name)) { errorMessage("#markdownError", "请选择不超过 1 MiB 的 Markdown 文件"); return; }
  try { replaceDocument(file.name, new TextDecoder("utf-8", { fatal: true }).decode(await file.arrayBuffer())); }
  catch { errorMessage("#markdownError", "请导入 UTF-8 编码的文件"); }
});

const requestedTool = location.hash.slice(1);
const storedTool = readDraft(activeToolKey).tool;
chooseTool(["translate", "markdown"].includes(requestedTool) ? requestedTool : storedTool);
const draft = readDraft(draftKey);
if (typeof draft.content === "string" && typeof draft.name === "string") {
  $("#markdownEditor").value = draft.content; $("#markdownFilename").value = draft.name;
  openedName = typeof draft.openedName === "string" ? draft.openedName : "";
  revision = typeof draft.revision === "string" ? draft.revision : null;
  savedContent = typeof draft.savedContent === "string" ? draft.savedContent : "";
}
persistMarkdown(); renderPreview();
refreshFiles().catch((error) => errorMessage("#markdownError", error.message));
try {
  const result = await api("/api/tools/status");
  translationMode = result.translation.mode;
  const languages = Object.entries(result.translation.languages);
  $("#sourceLanguage").replaceChildren(new Option("自动检测", "auto"));
  $("#targetLanguage").replaceChildren();
  for (const [code, name] of languages) { $("#sourceLanguage").add(new Option(name, code)); $("#targetLanguage").add(new Option(name, code)); }
  const translation = readDraft(translationDraftKey);
  $("#sourceLanguage").value = translation.source === "auto" || languages.some(([code]) => code === translation.source) ? translation.source : "auto";
  $("#targetLanguage").value = languages.some(([code]) => code === translation.target) ? translation.target : "en";
  if (typeof translation.text === "string") $("#translationInput").value = translation.text.slice(0, 5000);
  const apiMode = translationMode === "api";
  $("#translationMode").textContent = apiMode ? "Google Cloud · 实时" : "Google · 实时";
  for (const selector of ["#translationResultField", "#copyTranslationBtn"]) $(selector).classList.remove("hidden");
  $("#translateBtn").classList.add("hidden");
  $("#translationGrid").classList.add("api-mode");
  $("#markdownStatus").title = `服务器保存目录：${result.markdown_directory}`;
  syncTranslation(false);
  if ($("#translationInput").value.trim()) scheduleTranslation();
} catch (error) { $("#translationMode").textContent = "Google · 实时"; syncTranslation(false); errorMessage("#translationError", error.message); }
