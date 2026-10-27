import { mountBenchmarkPicker } from "./benchmark-picker.js?v=1";
import { mountResultViewer } from "./result-viewer.js?v=1";

export function mountWorkspace(host, bootstrap, initial = {}, callbacks = {}, outputHost, resultsHost) {
  const root = host.attachShadow({ mode: "open" });
  root.append(document.querySelector("#workspaceTemplate").content.cloneNode(true));
  const outputRoot = outputHost.attachShadow({ mode: "open" });
  outputRoot.append(document.querySelector("#workspaceOutputTemplate").content.cloneNode(true));
  const resultsRoot = resultsHost.attachShadow({ mode: "open" });
  resultsRoot.append(document.querySelector("#modelResultsTemplate").content.cloneNode(true));
  const state = {
    bootstrap,
    pipelineCards: [],
    nextCardId: 1,
    mode: "custom",
    references: [],
    activeReferenceIndex: 0,
    submitting: false,
    importingReferences: false,
    activeJobId: null,
    eventSource: null,
    yamlCardId: null,
    displayedImages: new Map(),
    resultGroups: new Map(),
    manuallySavedImages: new Map(),
    lastJobId: initial.jobId || null,
    lastJobPipeline: initial.jobPipeline || "",
    lastJobConfig: initial.jobConfig || "",
    lastJob: null,
    disposed: false,
  };

  const roots = [root, outputRoot, resultsRoot];
  const $ = (selector) => roots.map((item) => item.querySelector(selector)).find(Boolean);
  const $$ = (selector) => roots.flatMap((item) => [...item.querySelectorAll(selector)]);
  const terminalStatuses = new Set(["completed", "completed_with_errors", "cancelled", "failed", "interrupted"]);
  const benchmarkPicker = mountBenchmarkPicker(root, bootstrap, initial, () => updateSummary());
  const resultViewer = mountResultViewer(resultsRoot, (record, saved) => {
    if (state.disposed || !state.displayedImages.has(record.key)) return;
    state.manuallySavedImages.set(record.key, saved.url);
    record.element.querySelector("img").src = saved.url;
    record.element.querySelector("button").title = saved.output_dir;
  });

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  async function api(url, options = {}) {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || `请求失败 (${response.status})`);
    return data;
  }

  let toastTimer;
  function toast(message, isError = false) {
    const element = $("#toast");
    element.textContent = message;
    element.className = `toast show${isError ? " error" : ""}`;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (element.className = "toast"), 3600);
  }

  function pipelineInfo(name) {
    return state.bootstrap.pipelines.find((item) => item.name === name);
  }

  function applyPipelineDefaults(pipelineName) {
    const steps = Number(pipelineInfo(pipelineName)?.default_num_inference_steps);
    if (Number.isInteger(steps) && steps > 0) $("#stepsInput").value = String(steps);
  }

  function configStatus(card) {
    return card.temporaryStatus || pipelineInfo(card.pipeline)?.config_statuses?.[card.config] || null;
  }

  const pipelineGroups = ["FLUX", "Qwen", "Outpainting", "Other", "Test"];

  function pipelineGroup(info) {
    if (pipelineGroups.includes(info.group)) return info.group;
    if (info.name.startsWith("flux")) return "FLUX";
    if (info.name.startsWith("qwen")) return "Qwen";
    return info.name === "demo" ? "Test" : "Other";
  }

  function pipelineOptions(card) {
    return pipelineGroups.map((group) => {
      const members = state.bootstrap.pipelines.filter((info) => pipelineGroup(info) === group);
      const options = members.map((info) => {
        const status = info.config_statuses?.[info.configs?.[0]];
        const suffix = status && !status.available ? " · 当前不可运行" : "";
        return `<option value="${escapeHtml(info.name)}" ${info.name === card.pipeline ? "selected" : ""}>${escapeHtml(info.display_name + suffix)}</option>`;
      }).join("");
      return `<optgroup label="${group}">${options || '<option disabled>暂无 Pipeline</option>'}</optgroup>`;
    }).join("");
  }

  function addPipelineCard() {
    const available = pipelineInfo(initial.pipeline) || state.bootstrap.pipelines[0];
    state.pipelineCards.push({
      id: state.nextCardId++,
      pipeline: available?.name || state.bootstrap.pipelines[0]?.name || "",
      config: available?.configs?.includes(initial.config) ? initial.config : available?.configs?.[0] || "",
      gpus: new Set(initial.gpus || (state.bootstrap.gpus[0] ? [state.bootstrap.gpus[0].id] : ["cpu"])),
    });
    if (initial.fields?.stepsInput === undefined) {
      applyPipelineDefaults(state.pipelineCards.at(-1).pipeline);
    }
    renderPipelineCards();
  }

  function renderPipelineCards() {
    const container = $("#pipelineCards");
    if (!state.pipelineCards.length) {
      container.innerHTML = '<div class="empty-state">点击“添加 Pipeline”开始配置</div>';
      renderReferences();
      updateSummary();
      return;
    }
    container.innerHTML = state.pipelineCards.map((card, index) => {
      const info = pipelineInfo(card.pipeline);
      const options = pipelineOptions(card);
      const configs = info?.configs || [];
      if (!configs.includes(card.config)) card.config = configs[0] || "";
      const configOptions = configs.length
        ? configs.map((name) => `<option value="${escapeHtml(name)}" ${name === card.config ? "selected" : ""}>${escapeHtml(name)}</option>`).join("")
        : '<option value="">暂无配置</option>';
      const status = configStatus(card);
      const readiness = status
        ? status.available ? "" : `<div class="readiness blocked"><strong>当前不可运行</strong><span>${escapeHtml(status.issues.join("；"))}</span></div>`
        : '<div class="readiness blocked"><strong>缺少配置</strong></div>';
      const referenceText = info?.max_reference_images === null
        ? "参考图不限"
        : info?.max_reference_images > 0
          ? `最多 ${info.max_reference_images} 张参考图`
          : "不支持参考图";
      const modelDetails = [info?.description, info?.parameter_size, (info?.supported_tasks || []).join(" · "), info?.reference_limit_note || referenceText].filter(Boolean).join("\n");
      const devices = state.bootstrap.gpus.length
        ? state.bootstrap.gpus.map((gpu) => `<label class="gpu-chip" title="${escapeHtml(gpu.name)} · 空闲 ${escapeHtml(gpu.memory_free_mb)} MB"><input type="checkbox" data-action="gpu" data-gpu="${gpu.id}" ${card.gpus.has(gpu.id) ? "checked" : ""}/> GPU ${gpu.id}</label>`).join("")
        : `<label class="gpu-chip"><input type="checkbox" data-action="gpu" data-gpu="cpu" ${card.gpus.has("cpu") ? "checked" : ""}/> CPU</label>`;
      return `
        <div class="pipeline-card" data-card-id="${card.id}">
          <div class="pipeline-card-head">
            <span class="pipeline-number">PIPELINE ${String(initial.number || 1).padStart(2, "0")}</span>
            <button class="remove-pipeline" data-action="remove" title="移除此工作区" aria-label="移除此工作区">×</button>
          </div>
          <select data-action="pipeline" aria-label="选择 Pipeline" title="${escapeHtml(modelDetails)}">${options}</select>
          ${readiness}
          <div class="config-row">
            <select data-action="config">${configOptions}</select>
            <button class="secondary-button" data-action="yaml">编辑 YAML</button>
          </div>
          ${card.temporaryYaml !== undefined ? '<div class="temporary-config"><span>已应用临时配置 · 未保存</span><button type="button" class="text-button" data-action="reset-config">恢复保存版本</button></div>' : ""}
          <span class="gpu-label">运行设备（可多选）</span>
          <div class="gpu-options">${devices}</div>
        </div>`;
    }).join("");
    renderReferences();
    updateSummary();
  }

  function handlePipelineCardEvent(event) {
    const target = event.target;
    const cardElement = target.closest(".pipeline-card");
    if (!cardElement) return;
    const card = state.pipelineCards.find((item) => item.id === Number(cardElement.dataset.cardId));
    if (!card) return;
    const action = target.dataset.action;
    if (event.type === "click" && ["pipeline", "config", "gpu"].includes(action)) return;
    if (event.type === "change" && ["remove", "yaml"].includes(action)) return;
    if (action === "remove") {
      if (state.activeJobId || state.submitting || state.importingReferences) {
        toast("请先等待或取消本工作区的任务，再移除工作区", true);
        return;
      }
      callbacks.remove?.();
    } else if (action === "pipeline") {
      card.pipeline = target.value;
      card.config = pipelineInfo(card.pipeline)?.configs?.[0] || "";
      delete card.temporaryYaml;
      delete card.temporaryStatus;
      applyPipelineDefaults(card.pipeline);
      renderPipelineCards();
    } else if (action === "config") {
      card.config = target.value;
      delete card.temporaryYaml;
      delete card.temporaryStatus;
      renderPipelineCards();
    } else if (action === "reset-config" && event.type === "click") {
      delete card.temporaryYaml;
      delete card.temporaryStatus;
      renderPipelineCards();
    } else if (action === "gpu") {
      target.checked ? card.gpus.add(target.dataset.gpu) : card.gpus.delete(target.dataset.gpu);
      updateSummary();
    } else if (action === "yaml") {
      openYamlEditor(card);
    }
  }

  async function openYamlEditor(card) {
    state.yamlCardId = card.id;
    $("#yamlPipelineLabel").textContent = pipelineInfo(card.pipeline)?.display_name || card.pipeline;
    $("#yamlConfigName").value = card.config || "new-config";
    $("#yamlEditor").value = "";
    $("#yamlError").textContent = "";
    if (card.temporaryYaml !== undefined) {
      $("#yamlEditor").value = card.temporaryYaml;
    } else if (card.config) {
      try {
        const result = await api(`/api/pipelines/${encodeURIComponent(card.pipeline)}/configs/${encodeURIComponent(card.config)}`);
        $("#yamlEditor").value = result.content;
      } catch (error) {
        $("#yamlError").textContent = error.message;
      }
    }
    $("#yamlDialog").showModal();
  }

  async function saveYaml() {
    const card = state.pipelineCards.find((item) => item.id === state.yamlCardId);
    if (!card) return;
    const configName = $("#yamlConfigName").value.trim();
    const content = $("#yamlEditor").value;
    $("#yamlError").textContent = "";
    try {
      const result = await api(`/api/pipelines/${encodeURIComponent(card.pipeline)}/configs/${encodeURIComponent(configName)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content }),
      });
      const info = pipelineInfo(card.pipeline);
      if (!info.configs.includes(configName)) info.configs.push(configName);
      info.configs.sort();
      info.config_statuses[configName] = result.status;
      card.config = configName;
      delete card.temporaryYaml;
      delete card.temporaryStatus;
      $("#yamlDialog").close();
      renderPipelineCards();
      toast(`配置 ${configName} 已保存`);
    } catch (error) {
      $("#yamlError").textContent = error.message;
    }
  }

  async function applyTemporaryYaml() {
    const card = state.pipelineCards.find((item) => item.id === state.yamlCardId);
    if (!card) return;
    const content = $("#yamlEditor").value;
    $("#yamlError").textContent = "";
    try {
      const result = await api(`/api/pipelines/${encodeURIComponent(card.pipeline)}/validate-config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content }),
      });
      card.temporaryYaml = content;
      card.temporaryStatus = result.status;
      $("#yamlDialog").close();
      renderPipelineCards();
      toast("已应用到当前卡片，配置文件未改动");
    } catch (error) {
      $("#yamlError").textContent = error.message;
    }
  }

  function setMode(mode) {
    state.mode = mode;
    $$(".mode-btn").forEach((button) => {
      button.classList.toggle("active", button.dataset.mode === mode);
      button.setAttribute("aria-pressed", String(button.dataset.mode === mode));
    });
    $("#benchmarkMode").classList.toggle("hidden", mode !== "benchmark");
    $("#customMode").classList.toggle("hidden", mode !== "custom");
    $("#inputFields").scrollTop = 0;
    updateSummary();
  }

  function updateTaskInputs() {
    const task = $("#taskType").value;
    const field = $("#imageUploadField");
    field.classList.toggle("hidden", task === "T2I");
    $("#imageInput").multiple = task === "MultiReference";
    $("#uploadLabel").textContent = task === "MultiReference" ? "参考图像（可多选）" : "输入图像";
    renderReferences();
  }

  function referenceLimit() {
    if (["TI2I", "Outpainting"].includes($("#taskType").value)) return 1;
    const info = pipelineInfo(state.pipelineCards[0]?.pipeline);
    return info?.supported_tasks?.includes("MultiReference") ? info.max_reference_images : null;
  }

  function renderReferences() {
    const task = $("#taskType").value;
    const limit = referenceLimit();
    const count = state.references.length;
    const slots = limit === null
      ? Math.max(6, Math.ceil((count + 1) / 3) * 3)
      : Math.max(limit, count);
    state.activeReferenceIndex = Math.min(state.activeReferenceIndex, Math.max(0, count - 1));
    const active = state.references[state.activeReferenceIndex];
    $("#referenceSlots").innerHTML = Array.from({ length: slots }, (_, index) => {
      const ref = state.references[index];
      if (ref) return `<button type="button" class="reference-slot filled ${index === state.activeReferenceIndex ? "selected" : ""}" data-reference-index="${index}" aria-label="预览参考图 ${index + 1}" aria-pressed="${index === state.activeReferenceIndex}"><img src="${escapeHtml(ref.url)}" alt="${escapeHtml(ref.file.name)}" /><span class="reference-index">${index + 1}</span></button>`;
      return `<button type="button" class="reference-slot add" data-add-reference aria-label="添加参考图"><strong>＋</strong><span>添加图片</span></button>`;
    }).join("");
    $("#referenceCount").textContent = `已上传 ${count}${limit === null ? "" : "/" + limit} 张参考图`;
    $("#clearImagesBtn").disabled = !count || state.submitting || state.importingReferences;
    $("#referencePreview").innerHTML = active
      ? `<img src="${escapeHtml(active.url)}" alt="参考图 ${state.activeReferenceIndex + 1} 预览" /><span class="reference-index">${state.activeReferenceIndex + 1}</span><button type="button" class="delete-reference" data-remove-reference="${state.activeReferenceIndex}" aria-label="删除当前参考图">删除</button><span class="reference-filename">${escapeHtml(active.file.name)}</span>`
      : '<button type="button" class="preview-empty" data-add-reference>＋<span>添加参考图<br />也可拖拽到此处</span></button>';
    $("#imageDropZone").classList.toggle("single-reference", ["TI2I", "Outpainting"].includes(task));
  }

  function applyReferenceRatio() {
    if (!$("#autoRatioInput").checked || $("#taskType").value === "T2I") return;
    const ref = state.references[state.activeReferenceIndex];
    if (!ref) return;
    if ($("#taskType").value === "Outpainting") {
      $("#widthInput").value = Math.min(4096, Math.max(ref.width + 16, Math.ceil(ref.width * 1.5 / 16) * 16));
      $("#heightInput").value = Math.min(4096, Math.max(ref.height + 16, Math.ceil(ref.height * 1.5 / 16) * 16));
    } else {
      const scale = 1024 / Math.max(ref.width, ref.height);
      $("#widthInput").value = Math.max(64, Math.round(ref.width * scale / 16) * 16);
      $("#heightInput").value = Math.max(64, Math.round(ref.height * scale / 16) * 16);
    }
  }

  async function addReferenceFiles(files) {
    if (state.submitting || state.importingReferences) return;
    state.importingReferences = true;
    updateSummary();
    const limit = referenceLimit();
    for (const file of files) {
      if (limit !== null && state.references.length >= limit) {
        toast(`当前任务最多 ${limit} 张参考图，请先删除已有图片再添加`, true);
        break;
      }
      if (!/\.(png|jpe?g|webp|bmp)$/i.test(file.name) || file.size > 50 * 1024 * 1024) {
        toast(`${file.name}：请使用不超过 50MB 的 PNG/JPG/WEBP/BMP 图片`, true);
        continue;
      }
      const url = URL.createObjectURL(file);
      try {
        const image = new Image();
        image.src = url;
        await image.decode();
        if (state.disposed) {
          URL.revokeObjectURL(url);
          return;
        }
        const currentLimit = referenceLimit();
        if (currentLimit !== null && state.references.length >= currentLimit) {
          URL.revokeObjectURL(url);
          break;
        }
        state.references.push({ file, url, width: image.naturalWidth, height: image.naturalHeight });
        state.activeReferenceIndex = state.references.length - 1;
      } catch {
        URL.revokeObjectURL(url);
        toast(`无法读取图片：${file.name}`, true);
      }
    }
    $("#imageInput").value = "";
    state.importingReferences = false;
    renderReferences();
    applyReferenceRatio();
    updateSummary();
  }

  function clearReferences() {
    if (state.submitting || state.importingReferences) return;
    for (const ref of state.references) URL.revokeObjectURL(ref.url);
    state.references = [];
    state.activeReferenceIndex = 0;
    renderReferences();
    updateSummary();
  }

  function sampleCount() {
    return state.mode === "benchmark" ? benchmarkPicker.count() : 1;
  }

  function requestedTaskTypes() {
    if (state.mode === "custom") return new Set([$("#taskType").value]);
    return new Set(benchmarkPicker.selectedSamples().map((sample) => sample.task_type));
  }

  function updateSummary() {
    const pipelines = state.pipelineCards.filter((card) => card.pipeline && card.config && card.gpus.size);
    const samples = sampleCount();
    const requestedTasks = requestedTaskTypes();
    const incompatible = pipelines
      .map((card) => {
        const supported = new Set(pipelineInfo(card.pipeline)?.supported_tasks || []);
        const missing = [...requestedTasks].filter((task) => !supported.has(task));
        return missing.length ? `${card.pipeline} 不支持 ${missing.join("/")}` : null;
      })
      .filter(Boolean);
    const unavailable = pipelines
      .map((card) => {
        const status = configStatus(card);
        return status && !status.available
          ? `${pipelineInfo(card.pipeline)?.display_name || card.pipeline}: ${status.issues.join("；")}`
          : null;
      })
      .filter(Boolean);
    const unsupportedDevices = pipelines
      .map((card) => {
        const supported = new Set(pipelineInfo(card.pipeline)?.supported_devices || []);
        const selected = [...card.gpus].map((id) => id === "cpu" ? "cpu" : "cuda");
        const missing = [...new Set(selected.filter((device) => !supported.has(device)))];
        return missing.length ? `${card.pipeline} 不支持 ${missing.join("/")}` : null;
      })
      .filter(Boolean);
    const referenceLimits = pipelines
      .map((card) => {
        const limit = pipelineInfo(card.pipeline)?.max_reference_images;
        if (limit === null || limit === undefined) return null;
        const selected = state.mode === "benchmark"
          ? benchmarkPicker.selectedSamples()
          : [{ id: "自定义样本", input_images: $("#taskType").value === "T2I" ? [] : state.references }];
        const exceeded = selected.filter((sample) => (sample.input_images || []).length > limit);
        return exceeded.length ? `${card.pipeline} 最多 ${limit} 张参考图（${exceeded[0].id} 已超限）` : null;
      })
      .filter(Boolean);
    const customBlockers = [];
    if (state.mode === "benchmark") {
      customBlockers.push(...benchmarkPicker.blockers());
    }
    if (state.mode === "custom") {
      if (!$("#promptInput").value.trim()) customBlockers.push("请输入 Prompt");
      if ($("#taskType").value === "TI2I" && state.references.length !== 1) customBlockers.push("单图任务需要且只能使用 1 张参考图");
      if ($("#taskType").value === "MultiReference" && state.references.length < 2) customBlockers.push("多参考图任务至少需要 2 张图片");
      if ($("#taskType").value === "Outpainting") {
        if (state.references.length !== 1) customBlockers.push("扩图任务需要且只能使用 1 张输入图");
        const ref = state.references[0];
        if (ref && (Number($("#widthInput").value) < ref.width || Number($("#heightInput").value) < ref.height)) customBlockers.push("扩图目标宽高不能小于输入图");
      }
    }
    const blockers = [...incompatible, ...unsupportedDevices, ...referenceLimits, ...unavailable, ...customBlockers];
    $("#pipelineTotal").textContent = pipelines.length;
    $("#sampleTotal").textContent = samples;
    $("#unitTotal").textContent = pipelines.length * samples;
    const valid = pipelines.length === state.pipelineCards.length && pipelines.length > 0 && samples > 0 && !blockers.length;
    $("#runBtn").disabled = !valid || Boolean(state.activeJobId) || state.submitting || state.importingReferences;
    const busy = Boolean(state.activeJobId) || state.submitting || state.importingReferences;
    $("#runBtn").title = blockers.join("；");
    const selectedBenchmarks = benchmarkPicker.selections().map((item) => item.name);
    $("#outputLocation").textContent = `output/gen/${state.pipelineCards[0]?.pipeline || "{pipeline}"}/${state.mode === "custom" ? "users" : selectedBenchmarks.join(", ") || "{benchmark}"}/`;
    $("#outputLocationField").hidden = !$("#saveImagesInput").checked;
    $("#saveImagesInput").disabled = busy;
    $("#pipelineFields").disabled = busy;
    $("#inputFields").disabled = busy;
    benchmarkPicker.setDisabled(busy);
    $$(".mode-btn").forEach((button) => { button.disabled = busy; });
    $("#newPipelineBtn").disabled = busy;
    callbacks.changed?.();
  }

  async function uploadCustomImages() {
    const task = $("#taskType").value;
    if (task === "T2I") return [];
    const files = state.references.map((ref) => ref.file);
    if (!files.length) throw new Error("当前任务类型需要输入图像");
    if (["TI2I", "Outpainting"].includes(task) && files.length !== 1) throw new Error(`${task} 需要且只能选择一张图像`);
    if (task === "MultiReference" && files.length < 2) throw new Error("多参考图任务至少需要两张图片");
    const uploaded = [];
    for (const file of files) {
      const body = new FormData();
      body.append("file", file);
      uploaded.push(await api("/api/uploads", { method: "POST", body }));
    }
    return uploaded.map((item) => item.path);
  }

  async function startJob() {
    if (state.submitting || state.activeJobId) return;
    state.submitting = true;
    updateSummary();
    try {
      const pipelines = state.pipelineCards.map((card) => {
        if (!card.pipeline || !card.config || !card.gpus.size) throw new Error("Pipeline 配置不完整");
        const selection = { pipeline: card.pipeline, config_name: card.config, gpu_ids: [...card.gpus] };
        if (card.temporaryYaml !== undefined) selection.config_yaml = card.temporaryYaml;
        return selection;
      });
      const payload = {
        input_mode: state.mode,
        save_images: $("#saveImagesInput").checked,
        benchmarks: state.mode === "benchmark" ? benchmarkPicker.selections() : [],
        custom_samples: [],
        pipelines,
      };
      if (state.mode === "custom") {
        const prompt = $("#promptInput").value.trim();
        if (!prompt) throw new Error("请输入 Prompt");
        const task = $("#taskType").value;
        const reference = state.references[0];
        const width = Number($("#widthInput").value);
        const height = Number($("#heightInput").value);
        payload.custom_samples = [{
          id: "user-0001",
          task_type: task,
          prompt,
          negative_prompt: $("#negativePromptInput").value.trim(),
          input_images: await uploadCustomImages(),
          seed: $("#seedInput").value === "" ? null : Number($("#seedInput").value),
          width,
          height,
          num_inference_steps: Number($("#stepsInput").value),
          extra: task === "Outpainting" && reference ? { input_box: [Math.floor((width - reference.width) / 2), Math.floor((height - reference.height) / 2), Math.floor((width - reference.width) / 2) + reference.width, Math.floor((height - reference.height) / 2) + reference.height] } : {},
        }];
      }
      const job = await api("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      state.activeJobId = job.id;
      state.lastJobId = job.id;
      state.lastJobPipeline = payload.pipelines[0].pipeline;
      state.lastJobConfig = payload.pipelines[0].config_name;
      state.displayedImages.clear();
      state.resultGroups.clear();
      state.manuallySavedImages.clear();
      resultViewer.close();
      $("#resultGallery").innerHTML = "";
      $("#progressSection").classList.remove("hidden");
      $("#resultsSection").classList.add("hidden");
      resultsHost.hidden = true;
      renderJob(job);
      if (terminalStatuses.has(job.status)) state.activeJobId = null;
      else subscribeToJob(job.id);
      $("#progressSection").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
      toast(error.message, true);
    } finally {
      state.submitting = false;
      updateSummary();
    }
  }

  function subscribeToJob(jobId) {
    state.eventSource?.close();
    const events = new EventSource(`/api/jobs/${encodeURIComponent(jobId)}/events`);
    state.eventSource = events;
    events.onmessage = (event) => {
      if (state.disposed || state.activeJobId !== jobId) return;
      const job = JSON.parse(event.data);
      renderJob(job);
      if (terminalStatuses.has(job.status)) {
        events.close();
        state.activeJobId = null;
        updateSummary();
      }
    };
    events.onerror = () => {
      if (state.activeJobId) toast("进度连接已断开，可刷新页面查看已有任务", true);
    };
  }

  function renderJob(job) {
    state.lastJob = job;
    outputHost.hidden = false;
    $("#progressSection").classList.remove("hidden");
    $("#progressSection").dataset.jobId = job.id;
    const jobPipeline = state.lastJobPipeline || job.results?.[0]?.pipeline || job.results?.[0]?.unit_id?.split(":")[0]
      || Object.values(job.active || {})[0]?.pipeline;
    if (jobPipeline) state.lastJobPipeline = jobPipeline;
    if (!state.lastJobConfig && job.results?.[0]?.config_name) state.lastJobConfig = job.results[0].config_name;
    const title = outputLabel();
    $("#workspaceOutputTitle").textContent = title;
    $("#modelResultsTitle").textContent = title;
    const percent = Math.round(job.progress * 100);
    $("#sampleProgressText").textContent = `${job.completed} / ${job.total} · ${percent}%`;
    $("#sampleProgressBar").style.width = `${percent}%`;
    $("#sampleProgressBar").parentElement.setAttribute("aria-valuenow", String(percent));
    $("#sampleAverage").textContent = averageSampleLabel(job.average_sample_seconds);
    $("#sampleEta").textContent = etaLabel(job.eta_seconds, job.status, job.eta_phase);
    $("#jobStatus").textContent = job.status;
    $("#jobStatus").title = job.message || "";
    const active = Object.entries(job.active || {});
    $("#gpuProgressList").innerHTML = active.length
      ? active.map(([gpu, item]) => {
          const progress = Math.round((item.progress || 0) * 100);
          const title = [gpu === "cpu" ? "CPU" : `GPU ${gpu}`, item.pipeline, item.sample_id].filter(Boolean).join(" · ");
          return `<div class="gpu-progress-row"><div class="progress-title"><span>${escapeHtml(title)}</span><div class="progress-metrics"><strong>${item.step || 0} / ${item.total_steps || "—"}</strong><span class="progress-eta" data-denoise-eta>${escapeHtml(etaLabel(item.eta_seconds, job.status, item.eta_phase))}</span></div></div><div class="progress-track"><div class="progress-fill" style="width:${progress}%"></div></div></div>`;
        }).join("")
      : terminalStatuses.has(job.status) ? "" : `<div class="gpu-progress-row" aria-label="去噪进度"><div class="progress-title"><div class="progress-metrics"><strong>0 / —</strong><span class="progress-eta" data-denoise-eta>${escapeHtml(etaLabel(null, job.status, job.eta_phase))}</span></div></div><div class="progress-track"><div class="progress-fill"></div></div></div>`;
    if (job.errors?.length) {
      $("#errorDetails").classList.remove("hidden");
      $("#errorText").textContent = job.errors.map((item) => `${item.unit_id || `GPU ${item.gpu_id}`}: ${item.error}`).join("\n");
    } else {
      $("#errorDetails").classList.add("hidden");
      $("#errorText").textContent = "";
    }
    renderResults(job.results || []);
    callbacks.outputChanged?.();
  }

  function outputLabel() {
    return `Model ${String(initial.number || 1).padStart(2, "0")} · ${pipelineInfo(state.lastJobPipeline)?.display_name || state.lastJobPipeline || "未运行"}${state.lastJobConfig ? ` / ${state.lastJobConfig}` : ""}`;
  }

  function etaLabel(seconds, status, phase) {
    if (["completed", "completed_with_errors"].includes(status)) return "剩余 0秒";
    if (status === "cancelled") return "已取消";
    if (["failed", "interrupted"].includes(status)) return "已结束";
    if (status === "cancelling") return "正在停止";
    const known = typeof seconds === "number" && Number.isFinite(seconds) && seconds >= 0;
    if (!known) {
      if (status === "paused") return "已暂停";
      return phase === "waiting" ? "等待设备" : phase === "loading" ? "加载模型中" : "估算中";
    }
    return `${status === "paused" ? "已暂停 · " : ""}预计剩余 ${durationLabel(seconds)}`;
  }

  function durationLabel(seconds) {
    const total = Math.max(0, Math.ceil(seconds));
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor(total % 3600 / 60);
    const remainder = total % 60;
    return hours ? `${hours}小时${minutes}分` : minutes ? `${minutes}分${remainder}秒` : total ? `${total}秒` : "不足1秒";
  }

  function averageSampleLabel(seconds) {
    const known = typeof seconds === "number" && Number.isFinite(seconds) && seconds >= 0;
    return `平均每样本 ${known ? durationLabel(seconds) : "—"}`;
  }

  function renderResults(results) {
    for (const result of results) {
      const outpainting = result.request?.task_type === "Outpainting";
      let resultContainer = $("#resultGallery");
      if (outpainting) {
        const groupKey = `${state.lastJob?.id || state.lastJobId}:${result.unit_id}`;
        if (!state.resultGroups.has(groupKey)) {
          const group = document.createElement("section");
          group.className = "outpainting-result-group";
          const inputUrl = result.request?.input_image_urls?.[0] || "";
          const targetUrl = result.request?.target_image_url || "";
          const box = result.metadata?.input_box || result.request?.extra?.input_box;
          const canvasWidth = Number(result.metadata?.canvas_width || result.metadata?.inference?.canvas_width || result.request?.width || 1);
          const canvasHeight = Number(result.metadata?.canvas_height || result.metadata?.inference?.canvas_height || result.request?.height || 1);
          const targetFrame = Array.isArray(box) && box.length === 4
            ? `<span class="outpainting-input-frame" style="left:${100 * box[0] / canvasWidth}%;top:${100 * box[1] / canvasHeight}%;width:${100 * (box[2] - box[0]) / canvasWidth}%;height:${100 * (box[3] - box[1]) / canvasHeight}%"></span>`
            : "";
          group.innerHTML = `<div class="outpainting-result-input"><div class="outpainting-source-pair">${inputUrl ? `<figure><figcaption>输入图</figcaption><img src="${escapeHtml(inputUrl)}" alt="扩图输入" loading="lazy" /></figure>` : ""}${targetUrl ? `<figure><figcaption>GT</figcaption><span class="outpainting-target-image" style="aspect-ratio:${canvasWidth}/${canvasHeight}"><img src="${escapeHtml(targetUrl)}" alt="扩图 GT" loading="lazy" />${targetFrame}</span></figure>` : ""}</div><p>${escapeHtml(result.request?.prompt || "")}</p></div><div class="outpainting-generated-grid"></div>`;
          state.resultGroups.set(groupKey, group);
          $("#resultGallery").append(group);
        }
        resultContainer = state.resultGroups.get(groupKey).querySelector(".outpainting-generated-grid");
      }
      for (const [index, url] of (result.image_urls || []).entries()) {
        const key = `${state.lastJob?.id || state.lastJobId}:${result.unit_id}:${index}`;
        const legacyParts = String(result.unit_id || "").split(":");
        const sampleId = result.sample_id || legacyParts.slice(1, -1).join(":") || result.unit_id;
        const sample = [result.benchmark, sampleId].filter(Boolean).join(" / ");
        const label = `${sample}${(result.image_urls || []).length > 1 ? ` · ${index + 1}` : ""}`;
        let record = state.displayedImages.get(key);
        const savedUrl = state.manuallySavedImages.get(key);
        const saved = Boolean(savedUrl || !result.preview_id || result.saved_images?.[String(index)]);
        if (!record) {
          const element = document.createElement("article");
          element.className = `result-card${outpainting ? " outpainting-result-card" : ""}`;
          const box = result.metadata?.input_box || result.request?.extra?.input_box;
          const canvasWidth = Number(result.metadata?.canvas_width || result.metadata?.inference?.canvas_width || result.request?.width || 1);
          const canvasHeight = Number(result.metadata?.canvas_height || result.metadata?.inference?.canvas_height || result.request?.height || 1);
          const frame = outpainting && Array.isArray(box) && box.length === 4
            ? `<span class="outpainting-input-frame" style="left:${100 * box[0] / canvasWidth}%;top:${100 * box[1] / canvasHeight}%;width:${100 * (box[2] - box[0]) / canvasWidth}%;height:${100 * (box[3] - box[1]) / canvasHeight}%"></span>`
            : "";
          element.innerHTML = `<button type="button" class="result-open" aria-label="${escapeHtml(`查看 ${label}`)}"${outpainting ? ` style="aspect-ratio:${canvasWidth}/${canvasHeight}"` : ""}><img loading="lazy" alt="${escapeHtml(sample)}" />${frame}</button><div><strong>${escapeHtml(label)}</strong></div>`;
          record = { key, element };
          element.querySelector("button").addEventListener("click", () => resultViewer.open(record));
          state.displayedImages.set(key, record);
          resultContainer.append(element);
        }
        Object.assign(record, { url: savedUrl || url, previewId: result.preview_id, index, saved,
          label, filename: `${state.lastJobPipeline || "image"}-${sampleId || "sample"}-${index + 1}.png`.replace(/[\\/:*?"<>|]/g, "_") });
        const image = record.element.querySelector("img");
        if (image.getAttribute("src") !== record.url) image.src = record.url;
        record.element.querySelector("button").title = saved ? result.output_dir || "查看图像" : "临时预览";
      }
    }
    if (state.displayedImages.size) {
      resultsHost.hidden = false;
      $("#resultsSection").classList.remove("hidden");
      $("#resultCount").textContent = `${state.displayedImages.size} 张图像`;
    }
    resultViewer.refresh();
  }

  async function controlJob(action) {
    if (!state.activeJobId) return;
    try {
      const job = await api(`/api/jobs/${encodeURIComponent(state.activeJobId)}/${action}`, { method: "POST" });
      renderJob(job);
      if (terminalStatuses.has(job.status)) {
        state.eventSource?.close();
        state.activeJobId = null;
      }
      updateSummary();
    } catch (error) {
      toast(error.message, true);
    }
  }

  function bindEvents() {
    root.addEventListener("input", () => updateSummary());
    root.addEventListener("change", () => callbacks.changed?.());
    $("#pipelineCards").addEventListener("click", handlePipelineCardEvent);
    $("#pipelineCards").addEventListener("change", handlePipelineCardEvent);
    $$(".mode-btn").forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
    $("#taskType").addEventListener("change", () => { updateTaskInputs(); applyReferenceRatio(); updateSummary(); });
    $("#imageInput").addEventListener("change", (event) => addReferenceFiles([...event.target.files]));
    $("#imageUploadField").addEventListener("click", (event) => {
      if (state.submitting || state.importingReferences) return;
      if (event.target.closest("[data-add-reference]")) $("#imageInput").click();
      const thumbnail = event.target.closest("[data-reference-index]");
      if (thumbnail) {
        state.activeReferenceIndex = Number(thumbnail.dataset.referenceIndex);
        renderReferences();
        applyReferenceRatio();
      }
      const remove = event.target.closest("[data-remove-reference]");
      if (remove) {
        const [ref] = state.references.splice(Number(remove.dataset.removeReference), 1);
        if (ref) URL.revokeObjectURL(ref.url);
        renderReferences();
        applyReferenceRatio();
        updateSummary();
      }
    });
    $("#clearImagesBtn").addEventListener("click", clearReferences);
    $("#autoRatioInput").addEventListener("change", applyReferenceRatio);
    $("#imageDropZone").addEventListener("dragover", (event) => {
      event.preventDefault();
      event.currentTarget.classList.add("drag-over");
    });
    $("#imageDropZone").addEventListener("dragleave", (event) => {
      if (!event.currentTarget.contains(event.relatedTarget)) event.currentTarget.classList.remove("drag-over");
    });
    $("#imageDropZone").addEventListener("drop", (event) => {
      event.preventDefault();
      event.currentTarget.classList.remove("drag-over");
      addReferenceFiles([...event.dataTransfer.files]);
    });
    $("#promptInput").addEventListener("input", updateSummary);
    $("#clearPromptBtn").addEventListener("click", () => {
      $("#promptInput").value = "";
      updateSummary();
    });
    $("#expandPromptBtn").addEventListener("click", () => {
      $("#expandedPromptInput").value = $("#promptInput").value;
      $("#promptDialog").showModal();
    });
    $("#applyPromptBtn").addEventListener("click", () => {
      $("#promptInput").value = $("#expandedPromptInput").value;
      $("#promptDialog").close();
      updateSummary();
    });
    $("#runBtn").addEventListener("click", startJob);
    $("#newPipelineBtn").addEventListener("click", () => callbacks.newPipeline?.());
    $("#saveYamlBtn").addEventListener("click", saveYaml);
    $("#applyYamlBtn").addEventListener("click", applyTemporaryYaml);
  }

  async function initialize() {
    bindEvents();
    if (initial.jobId) state.activeJobId = initial.jobId;
    for (const [name, value] of Object.entries(initial.fields || {})) {
      const element = $("#" + name);
      if (!element || !["INPUT", "SELECT", "TEXTAREA"].includes(element.tagName)) continue;
      if (element.type === "checkbox") element.checked = Boolean(value);
      else element.value = String(value);
    }
    addPipelineCard();
    updateTaskInputs();
    setMode(initial.mode === "benchmark" ? "benchmark" : "custom");
    try {
      await benchmarkPicker.ready;
      if (state.disposed) return;
      if (initial.referenceFiles?.length) {
        await addReferenceFiles(initial.referenceFiles);
        // Preserve explicitly entered dimensions even when auto-ratio is enabled.
        for (const name of ["widthInput", "heightInput"]) {
          if (initial.fields?.[name] !== undefined) $("#" + name).value = initial.fields[name];
        }
      }
      if (initial.jobId) {
        const job = state.bootstrap.jobs?.find((item) => item.id === initial.jobId)
          || await api(`/api/jobs/${encodeURIComponent(initial.jobId)}`);
        if (state.disposed) return;
        renderJob(job);
        if (!terminalStatuses.has(job.status)) {
          state.activeJobId = job.id;
          subscribeToJob(job.id);
        } else state.activeJobId = null;
      }
    } catch (error) {
      state.activeJobId = null;
      toast(error.message, true);
    } finally {
      if (!state.disposed) updateSummary();
    }
  }

  const ready = initialize();
  return {
    ready,
    outputState() {
      return { label: outputLabel(), status: state.lastJob?.status, hasResults: state.displayedImages.size > 0 };
    },
    async control(action) {
      const status = state.lastJob?.status;
      if (action === "pause" && status !== "running") return;
      if (action === "resume" && status !== "paused") return;
      if (action === "cancel" && (!status || terminalStatuses.has(status) || status === "cancelling")) return;
      await controlJob(action);
    },
    cloneInputs() {
      const { mode, benchmarks, fields } = this.snapshot();
      return { mode, benchmarks, fields, referenceFiles: state.references.map((ref) => ref.file) };
    },
    updatePipelines(pipelines, selectedName) {
      state.bootstrap = { ...state.bootstrap, pipelines };
      if (selectedName && !state.activeJobId && !state.submitting) {
        const card = state.pipelineCards[0];
        card.pipeline = selectedName;
        card.config = pipelineInfo(selectedName)?.configs?.[0] || "";
        delete card.temporaryYaml;
        delete card.temporaryStatus;
      }
      renderPipelineCards();
    },
    snapshot() {
      const card = state.pipelineCards[0];
      if (!card) return null;
      const fields = {};
      for (const name of ["taskType", "seedInput", "promptInput", "negativePromptInput", "widthInput", "heightInput", "stepsInput", "autoRatioInput", "saveImagesInput"]) {
        const element = $("#" + name);
        fields[name] = element.type === "checkbox" ? element.checked : element.value;
      }
      return {
        id: initial.id,
        number: initial.number,
        pipeline: card.pipeline,
        config: card.config,
        gpus: [...card.gpus],
        mode: state.mode,
        benchmarks: benchmarkPicker.snapshot(),
        jobId: state.lastJobId,
        jobPipeline: state.lastJobPipeline,
        jobConfig: state.lastJobConfig,
        fields,
      };
    },
    dispose() {
      state.disposed = true;
      benchmarkPicker.dispose();
      state.eventSource?.close();
      resultViewer.dispose();
      clearTimeout(toastTimer);
      for (const ref of state.references) URL.revokeObjectURL(ref.url);
    },
  };
  }

