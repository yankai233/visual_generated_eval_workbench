# Visual Generation Evaluation Workbench

用于统一接入图像生成 Pipeline、选择 Benchmark 或自定义输入、分配多张 GPU，并在浏览器中实时观察样本与去噪进度。

## 启动

```bash
git clone git@github.com:yankai233/-visual_generated_eval_workbench.git
cd -- -visual_generated_eval_workbench
python -m pip install -e .
python -m src.main --server-name 0.0.0.0 --server-port 7861
```

启动日志会打印服务器可访问 IP。不要在浏览器中使用 `0.0.0.0`，应打开 `http://<server-ip>:7861`。

仓库包含源码、默认配置、测试与 `assets/GenBench` 示例数据；模型权重、生成结果、上传文件、日志和网页编辑器口令保留在本机，不纳入版本管理。首次验证可以使用无需模型权重的 Demo Pipeline。实际模型的配置保留了开发服务器的本地路径，在其他机器运行前，请按实际位置修改 `configs/{pipeline}/default/config.yaml` 中的模型、Python 环境和外部方法仓库路径；扩图数据入口 `assets/GenBench/outpainting_wikiart/outpainting-source.json` 也需要适配本地数据路径。

> `0.0.0.0` 会将服务暴露到可达网络。本项目当前不包含鉴权，处理敏感数据时请使用内网、防火墙或反向代理访问控制。

## 内置 Pipeline

| Pipeline | 任务 | 规模 | 参考图 |
|---|---|---|---|
| Qwen-Image-2.1 | T2I / TI2I / MultiReference | 7B 图像生成器，另含约 8.8B 文本编码器 | 最多 10 张 |
| FLUX.3 Image Klein Base | T2I / TI2I / MultiReference | 5.6B 图像 Transformer | 最多 10 张 |
| FLUX.1-dev | T2I | 12B 图像 Transformer | 不支持 |
| Qwen-Image-Edit-2511 | TI2I / MultiReference | 约 20.4B 图像 Transformer，另含约 8.3B 文本编码器 | 工作台限制 3 张；官方未声明硬上限 |
| Demo Pipeline | 全部 | 无权重 | 不限 |

默认配置位于 `configs/{pipeline}/default/config.yaml`。普通本地目录直接传给 Diffusers；`~/.cache/huggingface/hub/models--...` 形式的 cache 根目录会自动读取 `refs/main` 并解析到实际 snapshot，因此不需要手工填写哈希目录。

Pipeline 选择器按 `FLUX / Qwen / Outpainting / Other / Test` 分组；新插件可以通过类属性 `pipeline_group` 指定分组，默认为 `Other`。

点击三栏下方的“添加 model”新增一整行独立的模型配置、输入和 Run，并复制上一行的输入模式、Benchmark 子集与语言、文本参数和参考图。复制后的图片管理互不影响；不会继承上一行的任务或结果。同一 Pipeline 可添加多行，分别选择配置或 GPU。一行的任务不会禁用其他行；运行期间锁定本行配置与输入。所有模型共用一个运行状态区域，可选择“全部模型”或某个模型执行暂停、继续、取消；结果按 Model 编号分组，即使 Pipeline 相同也不会混淆。Run 显示输出目录，完整图像路径可悬停结果查看。

单图或多参考图任务支持槽位式上传、分批追加、拖拽、缩略图切换、大图预览、单张删除和全部清空。参考图按编号顺序传入模型；上限仅由当前工作区的 Pipeline 决定。切换任务或 Pipeline 保留当前行参考图，数量不匹配时禁止开始生成。开启“比例自动识别”时，以当前预览图的比例更新输出尺寸，长边为 1024，宽高对齐到 16 像素。

顶部仅显示“服务可用”和“查看 GPU 使用情况”入口。监控弹窗以物理 GPU 编号展示显存已用/总量（GiB）、显存占比、GPU 利用率、温度和功耗，每 2 秒读取一次 `/api/system/gpus`；关闭弹窗或隐藏浏览器标签页即停止请求。后端只读调用 `nvidia-smi`，快照缓存 1 秒以合并并发访问，不加载模型、不初始化 CUDA，也不修改 GPU 状态。缺少驱动、超时和不支持的字段分别显示错误或 `—`，不会伪造为零；错误后仍自动重试。驱动版本和采集时间放在弹窗标题的悬停信息中，不另加页面说明小字。

当前浏览器标签页会保留各工作区的文本参数、Benchmark 选择和任务关联，刷新后恢复各行任务进度。浏览器文件选择与临时 YAML 不跨刷新保存；未提交的参考图需要重新选择。移除工作区会关闭对应的进度连接，运行中的工作区必须先结束或取消任务才能移除。

原生前端的页面装配位于 `src/web/static/app.js`；每行的独立状态与交互位于 `src/web/static/workspace.js`，使用浏览器原生 Shadow DOM 隔离控件。多工作区浏览器回归测试位于 `tests/test_workspace_browser.py`，安装 Playwright 与 Chromium 后，设置 `WORKBENCH_BROWSER_URL=http://127.0.0.1:17861` 即可运行；测试会拦截运行请求，不启动真实推理。

YAML 编辑支持“仅本次使用”和“保存配置”。前者只保留在当前页面的卡片中，刷新即失效，不修改 `configs/`；每次运行仍会在结果目录记录实际使用的配置快照。可以点击“恢复保存版本”丢弃临时配置。Prompt 支持清空和放大编辑。

Run 中的“保存图像”默认开启；关闭时仅返回内存预览，不创建该批次的生成结果目录，也不归档输入到 `.cases/`。上传参考图仍需暂存到上传目录，任务摘要仍保留在 `.jobs/`。点击任意生成结果会打开原生图像查看窗口，支持“保存到项目”“下载”和关闭（也支持 Esc / 点击遮罩）。保存到项目只保存选中的图像，并补齐 request/config/result 及对比索引所需的输入归档；已保存图像显示“已保存”，重复点击不会另存副本。下载仅交给浏览器，不会触发项目保存；关闭查看也不会保存。

临时预览由 `src/runtime/preview_store.py` 管理：图像数据保留在服务进程内存中，默认最多 512 MiB、生成后最多 1 小时，容量不足时淘汰最早的预览。刷新网页可恢复仍有效的内存预览；服务重启、过期或容量回收后，未保存图像不可恢复，也不会进入结果对比页。需要长期保留时请及时保存或下载。自定义 Pipeline 自己写出的文件不受此开关控制。原生查看窗口逻辑位于 `src/web/static/result-viewer.js`；关闭“保存图像”的 API 参数为 `save_images: false`，不传则保持原有自动保存行为。

页面会分别展示模型能力和当前配置的可运行状态。依赖版本不足、权重分片缺失、模型目录错误或 PyTorch CUDA 不可用时，Pipeline 仍可浏览和编辑 YAML，但“开始生成”会被禁用并显示具体原因。这个检查不会加载模型或占用显存。

FLUX.3 的本地模型卡要求 `ModularPipeline.from_pretrained()` 后调用 `load_components()`，工作台已使用该专用加载路径。它还建议使用独立的 9B Prompt Upsampler；默认配置不自动下载或加载该额外模型，输入较短的 Prompt 时效果可能受影响。

## 接入 Pipeline

Pipeline 面板内的“添加 Pipeline”打开网页 Python/YAML 编辑器。新建时提供接口模板，必须实现 `from_pretrained(config, device)` 和 `__call__(request, *, progress_callback, control)`。保存后写入 `src/pipelines/{名称}/pipeline.py`、自动生成 `__init__.py` 并将 YAML 写入 `configs/{名称}/default/config.yaml`，注册成功立即出现在选择器中。已有网页插件可以载入编辑；内置/手动创建的插件不会被网页覆盖。保存检查语法、接口、占位代码与版本冲突，注册失败恢复原文件。注册不调用 `from_pretrained`，但会执行模块顶层代码；请勿在顶层加载模型。运行任务期间不允许注册代码。

**安全边界：Python 插件不是沙箱。** 首次打开编辑器时生成 `configs/.pipeline-editor-token`（权限 0600，不通过静态文件接口暴露），从服务器读取后输入编辑器的“保存口令”；也可在启动服务前设置 `WORKBENCH_PIPELINE_TOKEN`。口令不在 bootstrap 中返回，不保存在浏览器持久存储。仅在可信网络或 SSH 隧道使用此功能，不要通过不可信的明文 HTTP 网络传输口令。服务仍只适合受信任用户的本地工作台。

在 `src/pipelines/{pipeline_name}/` 中创建包，并从 `__init__.py` 导出 `PIPELINE_CLASS`：

```python
from .pipeline import MyPipeline

PIPELINE_CLASS = MyPipeline
```

实现类继承 `BaseGenerationPipeline`：

```python
from src.pipelines.base import BaseGenerationPipeline, GenerationResult

class MyPipeline(BaseGenerationPipeline):
    pipeline_name = "my_pipeline"
    display_name = "My Pipeline"
    supported_tasks = ("T2I", "TI2I", "MultiReference")

    @classmethod
    def from_pretrained(cls, config, device):
        return cls(...)

    def __call__(self, request, *, progress_callback=None, control=None):
        # 在每个去噪步骤调用 control.checkpoint()，即可支持原位暂停和取消。
        # 每步结束后调用 progress_callback(step, total_steps)。
        return GenerationResult(images=[image], metadata={})
```

对应配置放在：

```text
configs/{pipeline_name}/{config_name}/config.yaml
```

`from_pretrained` 接收 YAML 解析后的字典和设备。GPU 子进程会先设置 `CUDA_VISIBLE_DEVICES`，随后传入 `device="cuda:0"`。

## Benchmark 格式

每个 Benchmark 位于 `assets/GenBench/{benchmark_name}/`。加载器同时支持统一 manifest 和项目当前使用的原生目录格式。

统一格式可使用 `manifest.jsonl` 或 `manifest.json`。JSONL 每行一个样本：

```json
{
  "id": "000001",
  "task_type": "T2I",
  "prompt": "a cat near a window",
  "negative_prompt": "",
  "input_images": [],
  "seed": 42,
  "width": 1024,
  "height": 1024,
  "num_inference_steps": 28,
  "extra": {}
}
```

`input_images` 使用相对于当前 Benchmark 目录的路径。任务类型支持 `T2I`、`TI2I`、`MultiReference` 和 `Outpainting`。

当前原生格式也可直接使用：

```text
assets/GenBench/
├── t2i/
│   ├── 1.json
│   └── 2.json
├── ti2i/
│   └── 6/
│       ├── info.json
│       └── reference.png
└── multi_reference/
    └── 10/
        ├── info.json
        ├── reference_1.png
        └── reference_2.png
```

原生格式中的 `t2i`、`ti2i`、`multi_reference` 会自动规范化为统一任务类型；`prompt.zh` 和 `prompt.en` 可在页面切换。

## 多 Benchmark 运行

输入区切换到 Benchmark 后，可勾选多个数据集。每项分别选择 Prompt 语言，显示“已选 / 全部”数量；点击“筛选样本”打开独立弹窗，支持 ID/Prompt 搜索、任务类型过滤、样本勾选及每页 50 条的分页。全选/清空作用于当前筛选结果（含其他页）；“取消”不修改选择，“应用”才生效。各工作区的选择互不影响，刷新可恢复；清空某个数据集不会自动重新选中全部。

提交格式中的 `benchmarks` 为 `[{"name": "t2i", "language": "zh", "sample_ids": ["1", "2"]}, ...]`；`sample_ids: null` 或省略表示全部，`[]` 明确表示零个。旧的 `benchmark_name/benchmark_language/sample_ids` 单 Benchmark 接口仍可用（旧接口 `sample_ids: []` 维持全部的原语义），不要混用两种参数。

同一任务的样本统一调度、多 GPU 分配、暂停/继续；输出仍保存为 `output/gen/{pipeline}/{各自benchmark}/{id}-{同一批次时间}/`。不同数据集允许同名样本 ID，对比索引也保留真实 Benchmark。单个数据集在同一任务中只能选一种 Prompt 语言，避免同目录同 ID 覆盖。

## 辅助工具

主页与对比页的“辅助工具”会在当前界面打开浮动窗口，不会离开生成页面。浮窗可拖动标题栏移动，通过右下角缩放，也可折叠或关闭；关闭只隐藏窗口，再次打开仍保留当前翻译和 Markdown 草稿。窗口位置与尺寸保存在当前浏览器中，切换翻译/Markdown 不改变浮窗大小。`/tools` 路由仍作为浮窗的内部内容入口保留。

- 翻译：输入停止约 420 ms 后自动请求 Google 并显示译文，修改语言会立即重新翻译；快速连续输入会取消旧请求并丢弃过期响应。配置 `GOOGLE_TRANSLATE_API_KEY` 时使用 Google Cloud Translation，否则使用 Google 网页翻译接口。支持语言交换、复制、清空与浏览器草稿。
- 若需要在工具页直接显示译文，在服务进程环境中设置 `GOOGLE_TRANSLATE_API_KEY` 并重启服务。后端调用 [Google Cloud Translation Basic v2](https://docs.cloud.google.com/translate/docs/reference/rest/v2/translate)，密钥不返回浏览器。请启用相应 Google Cloud API、检查配额，并为密钥配置访问限制。未配置时明确显示“Google 网页模式”，不会伪造译文或使用非官方接口。
- Markdown：新建、导入本地 UTF-8 `.md/.markdown`、编辑、实时预览、下载、打开及保存服务器文件。草稿保存在当前浏览器，项目保存位置为 `output/tools/markdown/`。
- 保存同名文件需要匹配读取时的版本，其他页面已修改时拒绝覆盖；另存为新名称时不能直接覆盖已有文件。历史内容保留在 `output/tools/markdown/.history/{文件名}/`。不提供任意服务器路径读取、写入或删除。
- Markdown 预览支持标题、列表、代码块、表格等；禁用原始 HTML，预览框使用无脚本沙箱及 CSP，不加载远程图片。

工具逻辑位于 `src/tools/`，API 位于 `src/web/auxiliary.py`，原生页面位于 `src/web/templates/tools.html` 与 `src/web/static/tools.{js,css}`。生产依赖新增 `markdown-it-py` 与 `httpx`。

## 生成与对比输出

主页的“结果对比”进入 `/compare`。对比表行是样本、列是 Pipeline；每列可选择配置与生成批次，允许同一个 Pipeline 添加多列。支持样本勾选、搜索、仅共同样本、分页、历史结果切换、多图切换和放大查看；缺少结果显示 `—`。

Benchmark 按 `benchmark + sample_id` 对齐，输入指纹不同会标记“输入不同”（例如中英文 Prompt 版本）。自定义输入使用稳定的 SHA-256 指纹：任务类型、规范化 Prompt/Negative Prompt、有序参考图像素内容。上传文件路径、文件名、Pipeline、CFG、seed、步数与尺寸不参与输入身份，因此独立工作区上传同一输入也能对齐。不同 Prompt 或参考图顺序属于不同样本。

### 扩图任务与外部结果

内置 `ProOut`、`BlueOut`、`PQDiff` 三个扩图 Pipeline，模型代码来自相邻的 `Image-Generate-Filed/Outpainting/research`，通过独立的 `yankai_outpainting` Conda 环境加载，生成结果仍由工作台写入 `output/gen/`。`PQDiff` 当前检查到的 checkpoint 只有约 2.5 MiB，因此会明确显示不可运行，避免把不完整权重当成有效模型。扩图输入必须是一张图；自定义模式按原图居中自动扩展目标画布，Benchmark 模式提供外部 WikiArt 的 800 个样本及其原始定位框。

生成结果中的扩图样本按“输入图 + Prompt + 生成图”分组，生成图上的红色虚线框表示原图保留区域，框外是扩图区域。点击“扩图结果对比”进入 `/outpainting-compare`，页面会只读扫描 `Image-Generate-Filed/Outpainting/output/eval`，自动列出可用方法/批次，按样本 ID 对齐多列结果，支持多选方法、搜索、分页、多生成图与带红框放大查看。外部目录不会被复制或修改。

新结果采用 schema version 2，输入统一归档到 `output/gen/.cases/{完整输入指纹}/`，包含 `case.json` 和有序参考图。`request.json` 与 `result.json` 的 `comparison` 字段保存样本身份和输入清单路径。自定义输出路径的样本 ID 为 `u-{输入指纹}-{原始样本ID摘要}`，避免一个批次重复输入时互相覆盖；对比页仍按完整输入指纹对齐。

旧结果兼容读取，无需批量迁移。自定义旧结果只有输入完整时才自动匹配，无法读取参考图或缺少输入记录时保持独立并标记“待分组”。对比页的“自定义分组”允许将选中的结果合为一行，也可用不同分组名拆开原先自动匹配的结果。关联保存在 `output/gen/.comparison/groups.json`，不会重写生成图、原始请求或结果；“恢复自动分组”可撤销关联。

```text
output/gen/{pipeline}/{benchmark-or-users}/{sample-id}-{batch-start-time}/
├── generated.png
├── request.json
├── config.yaml
└── result.json
```

同一批任务使用同一个批次开始时间。取消任务不会删除已完成结果。

## 运行语义

样本进度旁显示整批预计剩余时间，去噪进度旁显示各 GPU 当前图像的预计剩余时间。去噪使用最近 7 个回调点的步间耗时估算，不将首步前的加载/编码时间当作单步耗时；整批按每张卡的剩余队列预测并取最大值，根据已完成样本的实际生成耗时平滑校准，并按分辨率与步数调整待生成样本的估计。模型初次加载不混入后续样本均速，暂停等待不计入测速；等待设备、模型加载、数据不足时显示相应状态，未知 Pipeline/配置不会凭空给出时间。估计会随实际速度和后处理开销修正，并非精确承诺；无需页面一直打开，刷新后从后台快照恢复。

- 同一物理 GPU 上只有一个工作进程，任务串行执行，避免不同 Pipeline 争抢显存。
- 多选 GPU 时样本按 GPU 轮转分配并行生成。
- 支持去噪回调的 Pipeline 会在去噪步边界暂停并原位继续。
- 不调用 `control.checkpoint()` 的 Pipeline 只能在当前样本结束后响应暂停或取消。
- 活动进程控制保存在服务内存中，任务摘要、结果与运行元数据会持久化到输出目录。
- 浏览器刷新后会自动恢复当前或最近一次任务视图。
- 多个工作区分别创建独立任务；若选用相同 GPU，则在该设备上排队，使用不同 GPU 可并行运行。
- 桌面端每组 Pipeline、输入样本、Run 面板固定等高 360px，输入区标题固定、内容内部滚动；长参考图列表和 Benchmark 不再撑高面板。“添加 model”始终位于配置组下方、共享状态区上方。主页紧凑主题位于 `src/web/static/workbench.css`，不改变对比页和辅助工具的尺寸。
- 任务历史持久化在 `output/gen/.jobs/`；服务异常退出时仍处于运行态的历史任务会标记为 `interrupted`。

