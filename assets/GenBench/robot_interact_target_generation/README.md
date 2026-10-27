# RobotInteract Target Generation Benchmark

该 Benchmark 用于比较不同多参考图生成模型是否适合构造 RobotInteraction 数据集的 Target 图像。它不是最终论文 Benchmark，而是 teacher model selection pilot。

## 输入与目标

每条样本输入两张参考图：

1. 人物身份与服装参考；
2. 机器人身份与身体结构参考。

Prompt 控制：

- human–robot interaction relation；
- 人物的六类目标情绪；
- 面部与非面部情绪线索；
- 人与机器人的左右布局、朝向和接触部位；
- 机器人结构保持要求。

输出应为一张写实、全身可见的人–机器人交互 Target 候选。

## 样本设计

共 8 条样本：

| ID | Relation | Emotion | Robot morphology | Difficulty |
|---|---|---|---|---|
| ri-001 | high five | happy | biped | easy |
| ri-002 | high five | surprised | biped | medium |
| ri-003 | shoulder touch | sad | biped | medium |
| ri-004 | shoulder touch | angry | biped | medium |
| ri-005 | handshake | fearful | wheeled | hard |
| ri-006 | handshake | disgusted | wheeled | hard |
| ri-007 | hug | happy | biped | very hard |
| ri-008 | push/block | angry | wheeled | very hard |

前三组通过固定主体、relation 和场景，只改变情绪，分别检查：

- happy vs. surprised high five；
- sad vs. angry shoulder touch；
- fearful vs. disgusted handshake。

另外两条检查拥抱遮挡与阻挡接触。

## 模型选择指标

不要只按画质选模型。每张输出应审核：

1. Human Identity Preservation；
2. Clothing Preservation；
3. Robot Identity Preservation；
4. Robot Structure Preservation；
5. Relation Correctness；
6. Contact Correctness；
7. Observed Emotion；
8. Face Emotion Evidence；
9. Body/Interaction Emotion Evidence；
10. Image Quality and Artifact Rate。

`prompt_emotion` 只是生成意图。审核时必须从输出图重新标注 `observed_emotion`，并统计 prompt-to-observed confusion matrix。

推荐同时统计：

```text
Target pass@1
Target pass@K
accepted targets / GPU hour
accepted complete tuples / GPU hour
```

模型只有在 relation、emotion、人物身份、机器人结构和基本图像质量同时通过时，才算一个有效 Target。

## 运行建议

- 对所有 Pipeline 使用相同样本、参考图和 seed；
- 每个模型至少运行 4 个候选 seed，不能只比较单张随机输出；
- 分别报告简单、中等、困难和极困难样本；
- Qwen-Image-2.1、FLUX 3 Klein 和后续 UNO Pipeline 使用同一套人工审核表；
- 构造标签的视觉模型不得作为唯一最终评价器。

参考图由图像生成工具专门为本地研究 pilot 创建，不对应现实中特定个人或商业机器人产品。
