const STORAGE_KEY = "visual-gen-eval-auxiliary-window-v2";

function readState() {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE_KEY));
    const width = Number(value?.width);
    const height = Number(value?.height);
    return {
      width: Number.isFinite(width) ? width : null,
      height: Number.isFinite(height) ? height : null,
      x: Number(value?.x),
      y: Number(value?.y),
      open: value?.open === true,
      collapsed: value?.collapsed === true,
      tool: value?.tool === "markdown" ? "markdown" : "translate",
    };
  } catch { /* Use the CSS defaults. */ }
  return { width: null, height: null, x: NaN, y: NaN, open: false, collapsed: false, tool: "translate" };
}

function clampSize(width, height) {
  return {
    width: Math.max(360, Math.min(width, window.innerWidth - 24)),
    height: Math.max(320, Math.min(height, window.innerHeight - 96)),
  };
}

export function mountAuxiliaryWindow() {
  const triggers = [...document.querySelectorAll("[data-auxiliary-trigger]")];
  if (!triggers.length || document.querySelector("#auxiliaryWindow")) return;

  const panel = document.createElement("section");
  panel.id = "auxiliaryWindow";
  panel.className = "auxiliary-window hidden";
  panel.setAttribute("role", "dialog");
  panel.setAttribute("aria-label", "辅助工具");
  panel.setAttribute("aria-modal", "false");
  panel.innerHTML = `
    <header class="auxiliary-window-heading" tabindex="-1">
      <strong>辅助工具</strong>
      <div class="auxiliary-window-actions">
        <button type="button" id="collapseAuxiliaryWindow" aria-label="折叠辅助工具" title="折叠">−</button>
        <button type="button" id="closeAuxiliaryWindow" aria-label="关闭辅助工具" title="关闭">×</button>
      </div>
    </header>
    <div class="auxiliary-window-content">
      <iframe id="auxiliaryWindowFrame" title="翻译与 Markdown 工具"></iframe>
    </div>`;
  document.body.append(panel);

  const content = panel.querySelector(".auxiliary-window-content");
  const frame = panel.querySelector("#auxiliaryWindowFrame");
  const heading = panel.querySelector(".auxiliary-window-heading");
  const collapseButton = panel.querySelector("#collapseAuxiliaryWindow");
  let loaded = false;
  let state = readState();
  let expandedLayout = Number.isFinite(state.width) && Number.isFinite(state.height)
    ? { width: state.width, height: state.height, x: state.x, y: state.y }
    : null;
  let dragging = null;

  const writeState = (changes = {}) => {
    state = { ...state, ...changes };
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); } catch { /* Storage may be disabled. */ }
  };

  const loadFrame = () => {
    if (loaded) return;
    frame.src = `/tools?embedded=1&v=20261023-2#${state.tool}`;
    loaded = true;
  };

  const isCompact = () => matchMedia("(max-width: 600px)").matches;

  const clampLayout = (layout) => {
    const size = clampSize(layout.width, layout.height);
    const defaultX = window.innerWidth - size.width - 22;
    const x = Number.isFinite(layout.x) ? layout.x : defaultX;
    const y = Number.isFinite(layout.y) ? layout.y : 76;
    return {
      ...size,
      x: Math.max(12, Math.min(x, window.innerWidth - size.width - 12)),
      y: Math.max(12, Math.min(y, window.innerHeight - size.height - 12)),
    };
  };

  const setExpandedLayout = () => {
    if (isCompact()) {
      for (const property of ["left", "top", "right", "width", "height"]) panel.style.removeProperty(property);
      return;
    }
    if (!expandedLayout) {
      const rect = panel.getBoundingClientRect();
      expandedLayout = { width: rect.width, height: rect.height, x: rect.x, y: rect.y };
    }
    expandedLayout = clampLayout(expandedLayout);
    panel.style.width = `${expandedLayout.width}px`;
    panel.style.height = `${expandedLayout.height}px`;
    panel.style.left = `${expandedLayout.x}px`;
    panel.style.top = `${expandedLayout.y}px`;
    panel.style.right = "auto";
  };

  const rememberLayout = () => {
    if (panel.classList.contains("collapsed") || panel.classList.contains("hidden")) return;
    const rect = panel.getBoundingClientRect();
    expandedLayout = { width: Math.round(rect.width), height: Math.round(rect.height), x: Math.round(rect.x), y: Math.round(rect.y) };
    writeState(expandedLayout);
  };

  const expand = (persist = true) => {
    panel.classList.remove("collapsed");
    content.hidden = false;
    collapseButton.textContent = "−";
    collapseButton.title = "折叠";
    collapseButton.setAttribute("aria-label", "折叠辅助工具");
    setExpandedLayout();
    if (persist) writeState({ open: true, collapsed: false });
  };

  const open = () => {
    panel.classList.remove("hidden");
    setExpandedLayout();
    loadFrame();
    expand();
    heading.focus();
  };

  const collapse = (persist = true) => {
    rememberLayout();
    panel.classList.add("collapsed");
    content.hidden = true;
    panel.style.height = "";
    collapseButton.textContent = "□";
    collapseButton.title = "展开";
    collapseButton.setAttribute("aria-label", "展开辅助工具");
    if (persist) writeState({ open: true, collapsed: true });
  };

  triggers.forEach((trigger) => trigger.addEventListener("click", open));
  collapseButton.addEventListener("click", () => {
    if (panel.classList.contains("collapsed")) {
      expand();
      return;
    }
    collapse();
  });
  panel.querySelector("#closeAuxiliaryWindow").addEventListener("click", () => {
    rememberLayout();
    panel.classList.add("hidden");
    writeState({ open: false });
    triggers[0]?.focus();
  });

  heading.addEventListener("pointerdown", (event) => {
    if (isCompact() || event.button !== 0 || event.target.closest("button")) return;
    const rect = panel.getBoundingClientRect();
    dragging = { pointerId: event.pointerId, offsetX: event.clientX - rect.left, offsetY: event.clientY - rect.top };
    heading.setPointerCapture(event.pointerId);
    panel.classList.add("dragging");
    event.preventDefault();
  });
  heading.addEventListener("pointermove", (event) => {
    if (!dragging || dragging.pointerId !== event.pointerId) return;
    const rect = panel.getBoundingClientRect();
    expandedLayout = clampLayout({ width: rect.width, height: rect.height, x: event.clientX - dragging.offsetX, y: event.clientY - dragging.offsetY });
    panel.style.left = `${expandedLayout.x}px`;
    panel.style.top = `${expandedLayout.y}px`;
    panel.style.right = "auto";
  });
  const finishDrag = (event) => {
    if (!dragging || dragging.pointerId !== event.pointerId) return;
    dragging = null;
    panel.classList.remove("dragging");
    if (heading.hasPointerCapture(event.pointerId)) heading.releasePointerCapture(event.pointerId);
    rememberLayout();
  };
  heading.addEventListener("pointerup", finishDrag);
  heading.addEventListener("pointercancel", finishDrag);

  window.addEventListener("message", (event) => {
    if (event.origin !== location.origin || event.source !== frame.contentWindow || event.data?.type !== "visual-eval-auxiliary-tool") return;
    state.tool = event.data.tool === "markdown" ? "markdown" : "translate";
    writeState({ tool: state.tool });
    if (!panel.classList.contains("hidden") && !panel.classList.contains("collapsed")) setExpandedLayout();
  });

  new ResizeObserver(() => rememberLayout()).observe(panel);
  window.addEventListener("resize", () => {
    if (!panel.classList.contains("hidden") && !panel.classList.contains("collapsed")) setExpandedLayout();
  });
  if (state.open) {
    panel.classList.remove("hidden");
    setExpandedLayout();
    loadFrame();
    if (state.collapsed) collapse(false);
    else expand(false);
  } else if (expandedLayout) setExpandedLayout();
}
