# 声析 Auralis · 鸿蒙版（HarmonyOS / ArkTS + MindSpore Lite）

把嗓音健康推理**完整搬到鸿蒙手机上本地运行**，音频不出设备。与安卓版同一套模型与算法，
但端侧推理引擎换成 **MindSpore Lite**（HarmonyOS 系统内置的轻量化 AI 引擎）。

## 快速开始

### 环境要求

| 项 | 版本 |
|---|---|
| DevEco Studio | ≥ 6.0.0（本仓库用 26.0.0.851 验证） |
| HarmonyOS SDK | API 12 起可装（`compatibleSdkVersion = 5.0.0(12)`），编译用 API 26 |
| 设备 | HarmonyOS NEXT（真机验证：Mate 70 Pro+ API 26、MatePad Pro 13.2 API 24） |

### 步骤

```bash
git clone <本仓库地址>
cd VoiceHealth
```

1. **放置模型文件**（见下方「模型文件」，仓库未包含）
2. **配置签名**：DevEco Studio → `File > Project Structure > Signing Configs`，勾选自动生成签名
3. **构建**：`hvigorw --mode module -p module=entry@default assembleHap`
   （或在 DevEco Studio 里 `Build > Build Hap(s)`）
4. **安装**：`hdc install -r entry/build/default/outputs/default/entry-default-signed.hap`
## 产物

```
entry/build/default/outputs/default/entry-default-unsigned.hap   (402.1 MB, 未签名)
```

HAP 内已包含全部模型（`resources/rawfile/`）：

| 资源 | 大小 | 用途 |
|---|---|---|
| `encoder.ms` | 348.4 MB | Whisper-small 编码器 + 投影 |
| `heads.ms` | 51.9 MB | 57 专家 + 语音检测 |
| `attrib.bin` | 0.1 MB | GBDT 判定 + 家族归因（原生 C++ 用） |
| `mel_filters.f32` | 0.1 MB | mel 滤波器组 |

**首次启动会自动把两个 `.ms` 释放到应用沙箱**（界面显示「正在释放模型 …（仅首次）」），
之后直接从沙箱加载。释放采用 **4 MB 分块流式拷贝**：`getRawFd()` 拿到 HAP 内偏移，
`fs.lseek` 定位后循环 `readSync/writeSync`——避免把 348 MB 一次性读进 ArkTS 堆。
文件大小与 rawfile 不一致时会自动重新释放（升级模型无需手动清理）。

## 结构

```
麦克风 16kHz 单声道 PCM
   └─ 原生 C++ log-mel 前端（NAPI，与 WhisperFeatureExtractor 误差 ~6e-6）
        └─ encoder.ms   (Whisper-small 编码器 + 投影 → 768 维嵌入)   ┐
             └─ heads.ms (57 专家 + 语音检测 → 135 维 + 语音概率)   ├ MindSpore Lite (ArkTS)
                  └─ 57 项指标后处理（argmax / round+clip）        │
                       ├─ GBDT 判定（原生 C++）→ 不健康概率 + 判定 ┘
                       └─ 原生 C++ 精确 tree-SHAP → 8 大家族贡献
```

- `entry/src/main/cpp/` — mel 前端、GBDT 判定、家族 SHAP 归因（全部为已验证的 C++ 代码）
- `entry/src/main/ets/pages/Index.ets` — 四屏 UI + 录音 + MindSpore Lite 推理
- `entry/src/main/resources/rawfile/` — `attrib.bin`（判定+归因模型，104 KB）、`mel_filters.f32`

## 构建

```powershell
powershell -File E:\medicalminiprogramdev\work\build_harmony.ps1
```
内部使用 DevEco Studio 自带的 hvigor（`tools/hvigor/bin/hvigorw.js`）与 SDK（`DEVECO_SDK_HOME`）。
`compatibleSdkVersion` / `targetSdkVersion` 为 **26.0.0**（API 26）。

## 模型文件（**未包含在本仓库**）

`encoder.ms`（约 174 MB）与 `heads.ms`（约 26 MB）由训练产物导出，**超过 GitHub 单文件 100 MB 上限**，
因此已在 `.gitignore` 中排除。放到 `entry/src/main/resources/rawfile/` 下即可参与打包：

```
entry/src/main/resources/rawfile/
├── encoder.ms         # Whisper-small 编码器（fp16 序列化权重，174 MB）
├── heads.ms           # 57 项专家头（26 MB）
├── attrib.bin         # GBDT 判定 + 家族归因（106 KB，已在仓库内）
└── mel_filters.f32    # log-mel 滤波器（64 KB，已在仓库内）
```

> 想直接托管在仓库里，可用 Git LFS（免费额度 1 GB 存储）：
> ```bash
> git lfs install
> git lfs track "entry/src/main/resources/rawfile/*.ms"
> git add .gitattributes entry/src/main/resources/rawfile/*.ms
> git commit -m "add models via git lfs"
> ```


**无需任何手动操作**：装上即用。两个 `.ms` 图位于 HAP 的 `resources/rawfile/`，
首次启动自动释放到沙箱：

```
/data/app/el2/100/base/com.auralis.voicehealth/haps/entry/files/{encoder.ms,heads.ms}
```

设备上约占用 800 MB（HAP 内一份 + 沙箱释放后一份）。

如需**替换模型**（例如重新转换后的图），重新构建 HAP 即可；应用会比较文件大小，
不一致时自动覆盖。也可以仍用 hdc 手动推送覆盖沙箱内同名文件：

```powershell
powershell -File E:\medicalminiprogramdev\work\push_harmony_models.ps1
```

> 之所以要释放一份而不是直接 `loadModelFromFd`：rawfile 在 HAP 内是一个**带偏移**的区域
> （`RawFileDescriptor{fd, offset, length}`），MindSpore Lite 的 `loadModelFromFd` 只接受
> 从文件头开始的 fd，无法处理该偏移，因此必须先落到沙箱。

## 数值校验（全部实测）

| 环节 | 结果 |
|---|---|
| `encoder.ms` vs ONNX | **NaN 0**，嵌入 max\|diff\| **1.382e-06**，余弦相似度 **1.00000012** |
| `heads.ms` vs ONNX | expert_raw max\|diff\| **1.431e-06**，**57/57 指标一致** |
| 原生 GBDT 判定 vs ONNX | 概率 max\|diff\| **3.576e-07**，判定 **10/10** |
| 原生家族归因 vs Python 服务端 | 家族 1-10 分 **10/10 精确一致** |
| C++ mel 前端 vs 官方提取器 | **6.08e-06** |

## ⚠️ ONNX → MindSpore Lite 踩到的坑（重要）

`converter_lite` 直接转换 Whisper 编码器会**静默产出全 NaN 的图**（转换返回 SUCCESS、无任何警告）。
逐节点二分定位后确认是 **MindSpore Lite 融合算子内核的数值问题**，必须在转换前改写 ONNX：

| 问题 | 现象 | 处理 |
|---|---|---|
| 融合 **Softmax** | 最后一层注意力 head 3 在 argmax 位置产生 18 个 NaN → 污染整个嵌入 | 拆成 `ReduceMax/Sub/Exp/ReduceSum/Div` |
| 融合 **LayerNormalization** | 大激活值下方差计算不稳定 | 拆成 `ReduceMean/Sub/Mul/ReduceMean/Add/Sqrt/Div` |
| `Clip` 的 max 输入为空串 | 转换器解析失败 | `Clip(x,min,+inf)` → `Max(x,min)` |
| `Shape` + `Expand` 广播 | `Shape` 返回值错误 → Div 产生 NaN | 删掉（ONNX 的 Div 本来就会广播） |

改写脚本：`work/harmony_fix_encoder.py`（每步都用 ONNX Runtime 校验改写前后逐位一致）。

## 已按需求调整（最终形态）

| 项 | 说明 |
|---|---|
| **推理设备** | **自动选择，界面上没有切换入口**：优先 **NPU(NNRT)**，失败自动回退 **CPU**；CPU 内部优先 **fp16**（见下），再退回 fp32 |
| **GPU 档** | **已删除** |
| **UI 风格切换** | **已删除**，固定为小程序的专业风（`#254F7F` 主色） |
| **嗓音日历** | **新增**：每次评测自动存入本地档案，按月日历回看，点某天看当天记录、点记录回看完整报告。数据仅存 `filesDir/voice_history.json`，**不上云** |

### 为什么 CPU 优先用 fp16

30 秒输入下 Whisper 编码器有 1500 帧，**单层注意力矩阵 12×1500×1500×4B = 108 MB**，
加上权重 348 MB，fp32 峰值约 1 GB，**实测在设备上撞到应用内存墙**：

```
graphs=1  rss=934MB
encoder predict status=-1  rss=977MB      ← OH_AI_STATUS_LITE_ERROR，分配失败
```

fp16 让权重与激活**都减半**，因此自动阶梯把 fp16 放在 fp32 之前。
状态栏的 `dev=` 会显示实际生效的设备（`nnrt:xxx` / `cpu-fp16(省内存)` / `cpu`）。


| 问题 | 根因 | 处理 |
|---|---|---|
| 两套 UI 切换无反应 | `pal`（调色板）是普通字段而非 `@State`，重新赋值不会触发 ArkUI 重渲染 | 改为 `@State pal: Palette`，赋值新实例即刷新 |
| 录音结束没有报告页、直接回首页 | `analyze()` 的异常被 catch 后静默 `screen='home'`，看不到原因 | 新增**失败页**：显示失败步骤、错误信息、中间量（pcm/mel 长度、嵌入模长、raw 范围、语音概率）；`procView` 也显示当前阶段 |
| 雷达图第二次分析不更新 | `Canvas.onReady` 只触发一次 | 抽出 `draw()`，用 `@Watch('redraw')` 在 `values` 变化时重绘 |

## NPU

MindSpore Lite 的 `Context` 支持 `target: ['nnrt']` + `nnrt.deviceID`，即通过 **NNRT**
（Neural Network Runtime）下沉到专用加速器：

```ts
const devices = mindSporeLite.getAllNNRTDeviceDescriptions();
// deviceType() === NNRTDeviceType.NNRTDEVICE_ACCELERATOR 即为 NPU
const ctx: mindSporeLite.Context = {
  target: ['nnrt'],
  nnrt: { deviceID: d.deviceID(),
          performanceMode: mindSporeLite.PerformanceMode.PERFORMANCE_HIGH,
          priority: mindSporeLite.Priority.PRIORITY_HIGH }
};
```

应用首页有 **CPU / NPU(NNRT)** 切换：选 NPU 时会枚举设备、优先取
`NNRTDEVICE_ACCELERATOR`，取不到则自动回退 CPU 并在状态栏说明原因。

**注意**：本机没有鸿蒙设备，NPU 路径**未真机验证**——代码已就绪，需要你在真机上点一下
「NPU (NNRT)」看状态栏显示的是实际设备名还是「未检出 NPU 设备，已回退 CPU」。

## 许可与致谢

本项目使用的模型与运行时**全部为宽松型许可（Apache-2.0 / MIT），允许商业使用**：

| 组件 | 来源 | 许可 |
|---|---|---|
| 音频编码器 + 投影层 | [laion/Majestrino-1.00](https://huggingface.co/laion/Majestrino-1.00) | Apache-2.0 |
| 编码器基座 + mel 滤波器 | [openai/whisper-small](https://huggingface.co/openai/whisper-small) | Apache-2.0 |
| 57 项嗓音专家头 | [laion/Majestrino-1.00-voice-experts](https://huggingface.co/laion/Majestrino-1.00-voice-experts) | MIT |
| 端侧推理引擎 | [MindSpore Lite](https://gitee.com/mindspore/mindspore-lite)（系统库，未随包分发） | Apache-2.0 |

**对原模型的修改**（Apache-2.0 §4(b) 要求声明）：仅导出音频分支、算子改写以适配 MindSpore Lite、
fp16 序列化权重——详见 [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)。

许可原文：[LICENSE-APACHE-2.0.txt](LICENSE-APACHE-2.0.txt) · [LICENSE-MIT.txt](LICENSE-MIT.txt)