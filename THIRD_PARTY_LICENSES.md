# 第三方组件与许可声明

本项目（声析 Auralis 鸿蒙版）使用了以下第三方模型与组件。**全部为宽松型许可（Apache-2.0 / MIT），
允许商业使用、修改与再分发**，但需履行相应的署名与声明义务。

---

## 一、模型

### 1. laion/Majestrino-1.00 — Apache-2.0

- **来源**：https://huggingface.co/laion/Majestrino-1.00
- **许可**：Apache License 2.0
- **用途**：音频编码器（`audio_encoder.*`）与投影层（`audio_proj.*`）
- **本项目所做修改**（Apache-2.0 第 4(b) 条要求声明）：
  1. 仅导出**音频分支**，丢弃文本编码器（`text_model.*`）与对比温度（`logit_scale`）；
  2. 由 PyTorch 导出为 ONNX，再转换为 **MindSpore Lite** 格式（`encoder.ms`）；
  3. 为适配端侧运行时，将 `LayerNormalization`、`Softmax` 拆解为基础算子，`Clip` 改写为 `Max`；
  4. 采用 **fp16 序列化权重**（计算仍为 fp32）。

### 2. openai/whisper-small — Apache-2.0

- **来源**：https://huggingface.co/openai/whisper-small
- **许可**：Apache License 2.0
- **用途**：Majestrino 的音频编码器基座（由上者微调得到），以及 log-mel 特征提取的滤波器参数
  （`mel_filters.f32` 取自 `transformers` 的 `WhisperFeatureExtractor`）
- **本项目所做修改**：同上（经微调、导出与算子改写）

### 3. laion/Majestrino-1.00-voice-experts — MIT

- **来源**：https://huggingface.co/laion/Majestrino-1.00-voice-experts
- **许可**：MIT License
- **用途**：57 项嗓音专家头（`heads.ms`）
- **本项目所做修改**：导出为 ONNX → MindSpore Lite，fp16 序列化权重

> ⚠️ **模型权重（`encoder.ms` / `heads.ms`）未包含在本仓库中**，原因见 README「模型文件」一节。

---

## 二、运行时与原生库

### 4. MindSpore Lite（华为）— Apache-2.0

- **来源**：https://gitee.com/mindspore/mindspore-lite
- **许可**：Apache License 2.0
- **用途**：端侧推理引擎
- **分发方式**：**未随应用打包**。本项目以系统库 `libmindspore_lite_ndk.so` 的形式**动态链接**
  HarmonyOS 系统自带的 MindSpore Lite，因此不构成再分发。

### 5. libc++（LLVM）— Apache-2.0 with LLVM Exception

- **许可**：Apache License 2.0 with LLVM Exceptions
- **用途**：C++ 运行时，由 DevEco Studio 构建时打包为 `libc++_shared.so`

### 6. scikit-learn（仅训练阶段）— BSD-3-Clause

- **许可**：BSD 3-Clause
- **用途**：训练嗓音健康判定 GBDT（300 棵树，深度 3）
- **分发方式**：**未随应用打包**。应用内以自研 C++/ArkTS 实现前向推理（`attrib.bin`），
  不包含 sklearn 代码。

---

## 三、许可义务清单

| 义务 | 状态 |
|---|---|
| 保留 Apache-2.0 许可文本 | 见本目录 `LICENSE-APACHE-2.0.txt` |
| 保留 MIT 许可文本 | 见本目录 `LICENSE-MIT.txt` |
| 声明对原模型的修改（Apache-2.0 §4(b)） | ✅ 已在本文档第一、二节逐项列明 |
| 保留原始版权声明 | ✅ 本节列明来源与许可 |
| 未使用「仅限研究/非商业」条款的模型 | ✅ 已核对：三个模型均允许商业使用 |

---

## 四、需要留意的一点

本项目的**界面视觉设计**参考了合作者提供的安卓版实现（`声析-UI更新版-v1.3.apk`）。
若该设计或代码由第三方作者创作，**请在公开发布前取得其明确授权**，并在本文件中补充署名。
纯视觉风格（配色、排版）通常不受版权保护，但**具体代码与美术素材**受保护。
