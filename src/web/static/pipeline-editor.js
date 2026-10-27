export function mountPipelineEditor(onSaved) {
  const $ = (selector) => document.querySelector(selector);
  const dialog = $("#pipelineEditorDialog");
  let targetId;
  let revision = null;
  let savedName = null;
  let draftName = "my_pipeline";
  let busy = false;
  const fields = ["savePipelineSourceBtn", "loadPipelineSourceBtn", "pipelineSourceName", "pipelineSourceCode", "pipelineSourceYaml", "managedPipeline"];
  function setBusy(value) {
    busy = value;
    fields.forEach((id) => { $("#" + id).disabled = value; });
  }

  async function api(url, options = {}) {
    const response = await fetch(url, options);
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : `请求失败 (${response.status})`);
    return data;
  }
  function headers() {
    return { "Content-Type": "application/json", "Authorization": `Bearer ${$("#pipelineEditorToken").value}` };
  }
  function populate(data) {
    $("#pipelineSourceName").value = data.name;
    $("#pipelineSourceCode").value = data.code;
    $("#pipelineSourceYaml").value = data.yaml;
    revision = data.revision;
    savedName = revision ? data.name : null;
    draftName = data.name;
  }
  async function template() {
    const data = await api("/api/pipeline-editor/template");
    populate(data);
    $("#managedPipeline").replaceChildren(new Option("新建", ""),
      ...(data.managed || []).map((name) => new Option(name, name)));
  }
  $("#pipelineSourceName").addEventListener("input", () => {
    const nextName = $("#pipelineSourceName").value.trim();
    const editor = $("#pipelineSourceCode");
    // Keep template identity synchronized; do not replace a custom display name.
    for (const attribute of ["pipeline_name", "display_name"]) {
      editor.value = editor.value.replace(`${attribute} = "${draftName}"`, `${attribute} = "${nextName}"`);
    }
    draftName = nextName;
  });
  $("#loadPipelineSourceBtn").addEventListener("click", async () => {
    if (busy) return;
    setBusy(true);
    $("#pipelineEditorError").textContent = "";
    try {
      const name = $("#managedPipeline").value;
      if (!name) await template();
      else populate(await api(`/api/pipeline-editor/source/${encodeURIComponent(name)}`, { headers: headers() }));
    } catch (error) { $("#pipelineEditorError").textContent = error.message; }
    finally { setBusy(false); }
  });
  $("#savePipelineSourceBtn").addEventListener("click", async () => {
    if (busy) return;
    setBusy(true);
    $("#pipelineEditorError").textContent = "";
    try {
      const name = $("#pipelineSourceName").value.trim();
      const data = await api("/api/pipeline-editor/source", {
        method: "POST", headers: headers(),
        body: JSON.stringify({ name, code: $("#pipelineSourceCode").value, yaml: $("#pipelineSourceYaml").value,
          revision: name === savedName ? revision : null }),
      });
      populate(data);
      await onSaved(data.pipeline, targetId);
      $("#pipelineEditorToken").value = "";
      dialog.close();
    } catch (error) { $("#pipelineEditorError").textContent = error.message; }
    finally { setBusy(false); }
  });
  dialog.addEventListener("cancel", (event) => { if (busy) event.preventDefault(); });
  dialog.addEventListener("submit", (event) => { if (busy) event.preventDefault(); });
  dialog.addEventListener("close", () => { $("#pipelineEditorToken").value = ""; });
  return {
    async open(workspaceId) {
      if (busy) return;
      targetId = workspaceId;
      $("#pipelineEditorError").textContent = "";
      dialog.showModal();
      setBusy(true);
      try { await template(); }
      catch (error) { $("#pipelineEditorError").textContent = error.message; }
      finally { setBusy(false); }
    },
  };
}
