export function mountResultViewer(root, onSaved) {
  const $ = (selector) => root.querySelector(selector);
  const dialog = $("#imageViewerDialog");
  let selected = null;
  let saving = false;
  let downloading = false;
  let failed = false;
  let disposed = false;
  const pendingDownloads = new Set();

  function render() {
    if (!selected || disposed) return;
    $("#imageViewerTitle").textContent = selected.label;
    const image = $("#viewerImage");
    if (image.getAttribute("src") !== selected.url) {
      failed = false;
      image.src = selected.url;
    }
    $("#saveResultBtn").disabled = saving || selected.saved || failed;
    $("#saveResultBtn").textContent = selected.saved ? "已保存" : saving ? "保存中…" : "保存到项目";
    $("#downloadResultBtn").disabled = downloading || failed;
    $("#downloadResultBtn").textContent = downloading ? "下载中…" : "下载";
  }
  $("#viewerImage").addEventListener("error", () => {
    failed = true;
    $("#imageViewerError").textContent = "图像不可用或临时预览已过期，请重新生成。";
    render();
  });
  $("#saveResultBtn").addEventListener("click", async () => {
    const record = selected;
    if (!record || record.saved || saving || failed) return;
    saving = true;
    $("#imageViewerError").textContent = "";
    render();
    try {
      const response = await fetch(`/api/previews/${encodeURIComponent(record.previewId)}/${record.index}/save`, { method: "POST" });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "保存失败");
      record.url = data.url;
      record.saved = true;
      onSaved(record, data);
    } catch (error) {
      if (selected === record && !disposed) $("#imageViewerError").textContent = error.message;
    } finally {
      saving = false;
      render();
    }
  });
  $("#downloadResultBtn").addEventListener("click", async () => {
    const record = selected;
    if (!record || downloading || failed) return;
    downloading = true;
    $("#imageViewerError").textContent = "";
    render();
    try {
      const url = record.saved ? record.url : `/api/previews/${encodeURIComponent(record.previewId)}/${record.index}/download`;
      const response = await fetch(url);
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.detail || "下载失败");
      }
      const blob = await response.blob();
      if (disposed) return;
      const objectUrl = URL.createObjectURL(blob);
      pendingDownloads.add(objectUrl);
      const anchor = document.createElement("a");
      anchor.href = objectUrl;
      anchor.download = record.filename;
      root.append(anchor);
      anchor.click();
      anchor.remove();
      setTimeout(() => { URL.revokeObjectURL(objectUrl); pendingDownloads.delete(objectUrl); }, 1000);
    } catch (error) {
      if (selected === record && !disposed) $("#imageViewerError").textContent = error.message;
    } finally {
      downloading = false;
      render();
    }
  });
  $("#closeImageViewerBtn").addEventListener("click", () => dialog.close());
  dialog.addEventListener("close", () => {
    selected = null;
    $("#viewerImage").removeAttribute("src");
  });
  dialog.addEventListener("click", (event) => {
    if (event.target !== dialog) return;
    const box = dialog.getBoundingClientRect();
    if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) dialog.close();
  });
  return {
    open(record) {
      selected = record;
      failed = false;
      $("#imageViewerError").textContent = "";
      render();
      dialog.showModal();
    },
    refresh: render,
    close() { dialog.close(); },
    dispose() {
      disposed = true;
      dialog.close();
      for (const url of pendingDownloads) URL.revokeObjectURL(url);
      pendingDownloads.clear();
    },
  };
}
