# Visual Generation Evaluation Workbench

图像生成可视化评测台，支持多模型运行、Benchmark 批量生成和结果对比。

## 功能

- 支持文生图、图像编辑、多参考图生成和扩图。
- 接入 Qwen、FLUX、UNO、ProOut、BlueOut、PQDiff 等 Pipeline。
- 多 GPU 调度，查看进度，暂停、继续或取消任务。
- 按样本对比生成结果，查看 GPU 状态及「模型 × Benchmark」平均耗时表。

## 安装与启动

需要 Python 3.10+；实际模型所需的 PyTorch、Diffusers 或独立环境需另行安装。

```bash
git clone git@github.com:yankai233/visual_generated_eval_workbench.git
cd visual_generated_eval_workbench
python -m pip install -e .
python -m src.main --server-name 0.0.0.0 --server-port 7861
```

浏览器打开 `http://<服务器IP>:7861`。服务未配置登录鉴权，请在可信网络使用。

## 使用

1. 修改 `configs/{pipeline}/default/config.yaml` 中的模型、环境和外部仓库路径；首次体验可选择无需权重的 **Demo Pipeline**。
2. 在页面选择模型，填写自定义输入或选择 Benchmark，分配 GPU 后开始生成。
3. 点击「结果对比」查看图像，点击「耗时统计」查看平均生成耗时，缺失数据显示 `-`。

## 目录

| 路径 | 内容 |
| --- | --- |
| `src/pipelines/` | 模型接入代码，可参考 `demo/` 新增 Pipeline |
| `configs/` | 模型配置 |
| `assets/GenBench/` | Benchmark 数据与示例 |
| `src/web/` | 前端与 API |
| `output/gen/` | 生成图像、请求参数与运行记录 |
| `tests/` | 测试 |

模型权重、运行输出、日志和密钥不包含在仓库中。扩图功能还需配置外部方法与数据路径。
