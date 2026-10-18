// HarmonyOS NAPI module: the WHOLE on-device pipeline in native code.
//
//   mic PCM (Float32Array) -> log-mel -> encoder.ms -> heads.ms (MindSpore Lite C API)
//                          -> 57-metric post-processing -> GBDT judgement -> tree-SHAP families
//
// MindSpore Lite is driven through the NDK C API (OH_AI_*) instead of the ArkTS wrapper: the
// ArkTS `model.getInputs()` returned undefined on device, and keeping everything native also
// removes the 960 KB mel round-trip through the JS heap.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <hilog/log.h>
#include <malloc.h>
#include <unistd.h>
#include <vector>

#include "napi/native_api.h"

#include "mindspore/context.h"
#include "mindspore/model.h"
#include "mindspore/tensor.h"
#include "mindspore/types.h"

#include "attrib.h"
#include "mel_frontend.h"
#include "voice_health_constants.h"

namespace {

vh::AttribModel g_attrib;
bool g_attribReady = false;
std::vector<float> g_melFilters;

OH_AI_ModelHandle g_enc = nullptr;
OH_AI_ModelHandle g_heads = nullptr;
OH_AI_ContextHandle g_ctxEnc = nullptr;
OH_AI_ContextHandle g_ctxHeads = nullptr;

// remembered so a failed run can rebuild the graphs and retry (self-healing)
std::string g_encPath;
std::string g_headsPath;
int g_activeMode = 2;
bool g_graphsLoaded = false;   // 模型当前是否驻留
std::vector<char> g_modelBuf;   // 内存兜底缓冲区（destroyGraphs 会释放其容量）
std::string g_device = "cpu";
std::string g_lastError;

// ---------------------------------------------------------------- helpers
bool getFloat32Array(napi_env env, napi_value value, std::vector<float> &out) {
    napi_typedarray_type type;
    size_t length = 0;
    void *data = nullptr;
    napi_value arraybuffer = nullptr;
    size_t byteOffset = 0;
    if (napi_get_typedarray_info(env, value, &type, &length, &data, &arraybuffer, &byteOffset) != napi_ok) {
        return false;
    }
    if (type != napi_float32_array || data == nullptr) return false;

    // 设备实测：本机 napi_get_typedarray_info 返回的 length 是「字节数」而不是元素个数
    // （480000 个 float 得到 1920000）。按元素个数使用会多拷贝 4 倍、读到数组末尾之外
    // 的堆内存，那些垃圾值（NaN / 1e21 / FLT_MAX）会让 log-mel 整片变成 NaN，
    // 最终表现为 MindSpore 的 "encoder predict status=-1"，且时好时坏。
    // 这里用 ArrayBuffer 的真实字节长度交叉校验，得到元素个数，不依赖该语义。
    size_t elems = length;
    void *abData = nullptr;
    size_t abLen = 0;
    if (arraybuffer != nullptr &&
        napi_get_arraybuffer_info(env, arraybuffer, &abData, &abLen) == napi_ok &&
        abLen >= byteOffset) {
        const size_t availBytes = abLen - byteOffset;
        if (length * sizeof(float) > availBytes) {
            elems = availBytes / sizeof(float);   // length 被当成字节数了
        }
    }
    out.assign(static_cast<float *>(data), static_cast<float *>(data) + elems);
    return true;
}

bool getString(napi_env env, napi_value value, std::string &out) {
    size_t len = 0;
    if (napi_get_value_string_utf8(env, value, nullptr, 0, &len) != napi_ok) return false;
    out.resize(len + 1);
    size_t copied = 0;
    if (napi_get_value_string_utf8(env, value, out.data(), len + 1, &copied) != napi_ok) return false;
    out.resize(copied);
    return true;
}

napi_value makeBool(napi_env env, bool v) {
    napi_value r = nullptr;
    napi_get_boolean(env, v, &r);
    return r;
}

napi_value makeUndefined(napi_env env) {
    napi_value r = nullptr;
    napi_get_undefined(env, &r);
    return r;
}

// ---- hilog：让设备端能看到每一步的详情（tag: VoiceHealth） ----
#define VH_DOMAIN 0xD001100
#define VH_TAG "VoiceHealth"
#define LOGI(...) OH_LOG_Print(LOG_APP, LOG_INFO, VH_DOMAIN, VH_TAG, __VA_ARGS__)
#define LOGE(...) OH_LOG_Print(LOG_APP, LOG_ERROR, VH_DOMAIN, VH_TAG, __VA_ARGS__)

/** 真实内存占用：Pss + SwapPss，来自 /proc/self/smaps_rollup（RSS 会把共享页全额计入） */
int readPssMb() {
    FILE *f = fopen("/proc/self/smaps_rollup", "r");
    if (f == nullptr) return -1;
    char line[256];
    long kb = 0;
    while (fgets(line, sizeof(line), f) != nullptr) {
        if (strncmp(line, "Pss:", 4) == 0 || strncmp(line, "SwapPss:", 8) == 0) {
            kb += atol(line + (line[4] == ':' ? 4 : 8));
        }
    }
    fclose(f);
    return (int)(kb / 1024);
}

napi_value makeString(napi_env env, const std::string &s) {
    napi_value r = nullptr;
    napi_create_string_utf8(env, s.c_str(), s.size(), &r);
    return r;
}

/** Resident set size in MB, straight from /proc — tells us whether we are hitting OOM. */
int readRssMb() {
    FILE *f = fopen("/proc/self/statm", "r");
    if (f == nullptr) return -1;
    long total = 0;
    long resident = 0;
    if (fscanf(f, "%ld %ld", &total, &resident) != 2) {
        fclose(f);
        return -1;
    }
    fclose(f);
    return (int)(resident * (long)sysconf(_SC_PAGESIZE) / (1024 * 1024));
}

bool loadFiltersFromFile(const std::string &path) {
    FILE *f = fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    const long want = (long)vh::kNFreqs * vh::kNMels * (long)sizeof(float);
    if (sz != want) {
        fclose(f);
        return false;
    }
    g_melFilters.resize((size_t)vh::kNFreqs * vh::kNMels);
    const size_t got = fread(g_melFilters.data(), 1, (size_t)sz, f);
    fclose(f);
    if (got != (size_t)sz) {
        g_melFilters.clear();
        return false;
    }
    return true;
}

void destroyGraphs() {
    // 释放内存兜底缓冲区：assign 之后 capacity 会一直占着（heads 54MB 正好在阈值内）
    g_modelBuf.clear();
    g_modelBuf.shrink_to_fit();
    if (g_enc != nullptr) {
        OH_AI_ModelDestroy(&g_enc);
        g_enc = nullptr;
    }
    if (g_heads != nullptr) {
        OH_AI_ModelDestroy(&g_heads);
        g_heads = nullptr;
    }
    if (g_ctxEnc != nullptr) {
        OH_AI_ContextDestroy(&g_ctxEnc);
        g_ctxEnc = nullptr;
    }
    if (g_ctxHeads != nullptr) {
        OH_AI_ContextDestroy(&g_ctxHeads);
        g_ctxHeads = nullptr;
    }
#ifdef __GLIBC__
    // 把空闲内存归还系统，缓解反复构建模型造成的堆碎片
    malloc_trim(0);
#endif
}

/** Cheap probe: is there an NNRT *accelerator*? Loading a 348 MB graph into NNRT and failing
 *  leaves the allocator fragmented, which then makes the CPU fallback fail much earlier
 *  (observed: the predict failure threshold dropped from 977 MB to 778 MB). */
bool nnrtAcceleratorAvailable() {
    size_t num = 0;
    NNRTDeviceDesc *descs = OH_AI_GetAllNNRTDeviceDescs(&num);
    bool found = false;
    if (descs != nullptr && num > 0) {
        for (size_t i = 0; i < num; i++) {
            NNRTDeviceDesc *d = OH_AI_GetElementOfNNRTDeviceDescs(descs, i);
            if (d != nullptr && OH_AI_GetTypeFromNNRTDeviceDesc(d) == OH_AI_NNRTDEVICE_ACCELERATOR) {
                found = true;
                break;
            }
        }
    }
    if (descs != nullptr) {
        OH_AI_DestroyAllNNRTDeviceDescs(&descs);
    }
    return found;
}

/** Automatic device priority: NPU (only when an NNRT accelerator really exists), then
 *  CPU fp16, then CPU fp32.
 *
 *  fp16 comes first among the CPU modes because it is the only configuration verified to work
 *  on device: fp32 needs ~1 GB of activations for the 30 s Whisper encoder and dies with
 *  `encoder predict status=-1` around 977 MB. Trying fp32 first would also fragment the
 *  allocator before fp16 ever gets a chance.
 *  mode: 0 = CPU fp32, 1 = NPU (NNRT), 2 = CPU fp16. */
OH_AI_ContextHandle makeContext(int mode) {
    OH_AI_ContextHandle ctx = OH_AI_ContextCreate();
    if (ctx == nullptr) return nullptr;
    OH_AI_ContextSetThreadNum(ctx, 4);   // 回退：2 线程未带来收益
    // parallel execution duplicates intermediate buffers; the encoder is memory-bound on device
    OH_AI_ContextSetEnableParallel(ctx, false);

    // NNRT accelerator FIRST so that every operator it can take is offloaded, then CPU as the
    // fallback for the rest. Registering NNRT alone (as before) demands that the WHOLE graph
    // compile for the NPU, and a single unsupported op fails the entire build.
    if (mode == 1) {
        size_t num = 0;
        NNRTDeviceDesc *descs = OH_AI_GetAllNNRTDeviceDescs(&num);
        OH_AI_DeviceInfoHandle accel = nullptr;
        if (descs != nullptr && num > 0) {
            for (size_t i = 0; i < num; i++) {
                NNRTDeviceDesc *d = OH_AI_GetElementOfNNRTDeviceDescs(descs, i);
                if (d != nullptr &&
                    OH_AI_GetTypeFromNNRTDeviceDesc(d) == OH_AI_NNRTDEVICE_ACCELERATOR) {
                    const char *nm = OH_AI_GetNameFromNNRTDeviceDesc(d);
                    accel = OH_AI_CreateNNRTDeviceInfoByName(nm);
                    if (accel != nullptr) {
                        OH_AI_DeviceInfoSetPerformanceMode(accel, OH_AI_PERFORMANCE_HIGH);
                        OH_AI_DeviceInfoSetPriority(accel, OH_AI_PRIORITY_HIGH);
                        OH_AI_ContextAddDeviceInfo(ctx, accel);
                        g_device = std::string("nnrt+fp16cpu:") + (nm != nullptr ? nm : "accel");
                    }
                    break;
                }
            }
        }
        if (descs != nullptr) {
            OH_AI_DestroyAllNNRTDeviceDescs(&descs);
        }
    }

    OH_AI_DeviceInfoHandle cpu = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_CPU);
    if (cpu != nullptr) {
        // mode 0 = fp32 (best accuracy); modes 1/2 = fp16 (half the activation memory)
        OH_AI_DeviceInfoSetEnableFP16(cpu, mode != 0);
        OH_AI_ContextAddDeviceInfo(ctx, cpu);
    }
    if (mode == 1 && g_device.rfind("nnrt", 0) != 0) {
        g_device = "cpu-fp16 (NNRT 未检出加速器)";
    } else if (mode == 2) {
        g_device = "cpu-fp16(省内存)";
    } else if (mode == 0) {
        g_device = "cpu";
    }
    return ctx;
}

/** Build one .ms graph.
 *  Tries the file path first, then falls back to slurping the file into memory and using
 *  OH_AI_ModelBuild. `diag` collects an actionable trace (open result, size, statuses). */

OH_AI_ModelHandle tryBuild(const std::string &path, OH_AI_ContextHandle ctx, OH_AI_Status &st,
                           std::string &diag) {
    OH_AI_ModelHandle m = OH_AI_ModelCreate();
    if (m == nullptr) {
        st = OH_AI_STATUS_CORE_FAILED;
        diag += "ModelCreate=null;";
        return nullptr;
    }
    st = OH_AI_ModelBuildFromFile(m, path.c_str(), OH_AI_MODELTYPE_MINDIR, ctx);
    if (st == OH_AI_STATUS_SUCCESS) {
        diag += "fileBuild=ok;";
        return m;
    }
    const int stFile = (int)st;
    OH_AI_ModelDestroy(&m);

    // probe the file so the message tells us whether the native layer can even see it
    FILE *f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        diag += "fileBuild=" + std::to_string(stFile) + " fopen=FAIL;";
        return nullptr;
    }
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    diag += "fileBuild=" + std::to_string(stFile) + " fopen=ok size=" + std::to_string(sz) + ";";
    if (sz <= 0) {
        fclose(f);
        return nullptr;
    }
    // Only slurp small graphs: keeping a 348 MB buffer alive on top of the model's own weights
    // doubles the footprint and made the device run out of memory.
    if (sz > 64L * 1024 * 1024) {
        fclose(f);
        diag += "skipBufFallback(sizeTooLarge);";
        return nullptr;
    }
    g_modelBuf.assign((size_t)sz, 0);
    const size_t got = fread(g_modelBuf.data(), 1, (size_t)sz, f);
    fclose(f);
    if (got != (size_t)sz) {
        diag += "read=" + std::to_string(got) + ";";
        g_modelBuf.clear();
        return nullptr;
    }
    m = OH_AI_ModelCreate();
    st = OH_AI_ModelBuild(m, g_modelBuf.data(), g_modelBuf.size(), OH_AI_MODELTYPE_MINDIR, ctx);
    if (st == OH_AI_STATUS_SUCCESS) {
        diag += "bufBuild=ok;";
        return m;
    }
    diag += "bufBuild=" + std::to_string((int)st) + ";";
    OH_AI_ModelDestroy(&m);
    g_modelBuf.clear();
    return nullptr;
}

}  // namespace

// ---------------------------------------------------------------- initMel(path)
static napi_value InitMel(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1] = {nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    if (argc < 1) return makeBool(env, false);
    std::string path;
    if (!getString(env, args[0], path)) return makeBool(env, false);
    return makeBool(env, loadFiltersFromFile(path));
}

// ---------------------------------------------------------------- initModel(path)  (GBDT + SHAP)
static napi_value InitModel(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1] = {nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    if (argc < 1) return makeBool(env, false);
    std::string path;
    if (!getString(env, args[0], path)) return makeBool(env, false);
    g_attribReady = g_attrib.load(path);
    return makeBool(env, g_attribReady);
}

// ---------------------------------------------------------------- loadGraphs(encPath, headsPath, useNpu) -> note
namespace {
/** Build encoder + heads, one context per model (a shared context broke the 2nd build). */
bool buildBoth(const std::string &encPath, const std::string &headsPath, int mode,
               std::string &diag) {
    OH_AI_Status st = OH_AI_STATUS_CORE_FAILED;
    g_ctxEnc = makeContext(mode);
    if (g_ctxEnc == nullptr) {
        diag += "ContextCreate=null;";
        return false;
    }
    diag += "enc[" + encPath + "] rss=" + std::to_string(readRssMb()) + "MB ";
    g_enc = tryBuild(encPath, g_ctxEnc, st, diag);
    LOGI("buildBoth mode=%{public}d encoder=%{public}s pss=%{public}dMB rss=%{public}dMB",
         mode, (g_enc != nullptr ? "OK" : "FAIL"), readPssMb(), readRssMb());
    if (g_enc == nullptr) {
        return false;
    }
    g_ctxHeads = makeContext(mode);
    if (g_ctxHeads == nullptr) {
        diag += "headsCtx=null;";
        return false;
    }
    diag += "heads[" + headsPath + "] rss=" + std::to_string(readRssMb()) + "MB ";
    g_heads = tryBuild(headsPath, g_ctxHeads, st, diag);
    LOGI("buildBoth mode=%{public}d heads=%{public}s pss=%{public}dMB rss=%{public}dMB",
         mode, (g_heads != nullptr ? "OK" : "FAIL"), readPssMb(), readRssMb());
    if (g_heads != nullptr) {
        g_activeMode = mode;
    }
    return g_heads != nullptr;
}
}  // namespace

static napi_value LoadGraphs(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2] = {nullptr, nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    std::string encPath;
    std::string headsPath;
    if (argc < 2 || !getString(env, args[0], encPath) || !getString(env, args[1], headsPath)) {
        return makeString(env, "参数错误");
    }

    destroyGraphs();
    g_lastError.clear();

    // NPU (NNRT) first, in HYBRID form: the accelerator takes every operator it supports and the
    // CPU covers the rest, so one unsupported op no longer fails the whole build.
    // Then CPU fp16 (the configuration already verified on device).
    // 设备实测：失败发生在 PSS=503MB，而成功峰值 605MB，且系统还有 2.9GB 可用
    // —— 内存不是约束。因此优先 fp32（精度最好），fp16 仅作兜底。
    // 怀疑之前的 status=-1 是 fp16 模式的运行时缺陷（失败仅耗时 0.2s，未进入计算）。
    std::vector<std::vector<int>> passes;
    passes.push_back({0});      // CPU fp32
    passes.push_back({2});      // CPU fp16 兜底

    std::string diag;
    bool ok = false;
    for (size_t p = 0; p < passes.size() && !ok; p++) {
        if (p > 0) {
            diag += " || 以纯 CPU 重新尝试: ";
        }
        for (size_t i = 0; i < passes[p].size() && !ok; i++) {
            destroyGraphs();
            diag += "[dev=" + std::to_string(passes[p][i]) + "] ";
            ok = buildBoth(encPath, headsPath, passes[p][i], diag);
        }
    }

    if (!ok) {
        const std::string msg = "模型构建失败: " + diag;
        g_lastError = msg;
        LOGE("LoadGraphs FAILED diag=%{public}s", diag.c_str());
        destroyGraphs();
        return makeString(env, msg);
    }
    g_encPath = encPath;    // 供失败自愈时重建
    g_headsPath = headsPath;
    g_graphsLoaded = true;
    LOGI("LoadGraphs OK device=%{public}s pss=%{public}dMB", g_device.c_str(), readPssMb());
    return makeString(env, "");
}

// ---------------------------------------------------------------- currentRss() -> number
static napi_value CurrentRss(napi_env env, napi_callback_info info) {
    napi_value v = nullptr;
    napi_create_int32(env, readRssMb(), &v);
    return v;
}

// ---------------------------------------------------------------- status()
static napi_value Status(napi_env env, napi_callback_info info) {
    std::string s = "filters=" + std::to_string(g_melFilters.size()) +
                    " attrib=" + (g_attribReady ? "1" : "0") +
                    " graphs=" + ((g_enc != nullptr && g_heads != nullptr) ? "1" : "0") +
                    " rss=" + std::to_string(readRssMb()) + "MB" +
                    " dev=" + g_device;
    if (!g_lastError.empty()) s += " err=" + g_lastError;
    return makeString(env, s);
}

// ---------------------------------------------------------------- pipeline (no JS calls, thread-safe)
struct PipelineOut {
    int metrics[VH_N_CODES] = {0};
    double prob = 0;
    int label = 0;
    double speech = 0;
    double embNorm = 0;
    float fam[8] = {0};
    float famShap[8] = {0};
    std::string error;
    // diagnostics / self-healing
    int rssMel = 0;
    int rssEnc = 0;
    int rssHeads = 0;
    bool needsRebuild = false;
    bool rebuilt = false;
};


bool computePipeline(const std::vector<float> &audio, PipelineOut &out) {
    if (g_melFilters.empty()) {
        out.error = "MEL_FILTERS_EMPTY";
        return false;
    }
    if (g_enc == nullptr || g_heads == nullptr) {
        out.error = "GRAPHS_NOT_LOADED";
        return false;
    }

    // ---- 1) log-mel ----
    std::vector<float> mel((size_t)vh::kNMels * vh::kNFrames);
    vh::computeLogMel(audio.empty() ? nullptr : audio.data(), (int)audio.size(),
                      g_melFilters.data(), mel.data());
    out.rssMel = readRssMb();
    // 检查 mel 是否含 NaN/Inf —— 输入脏数据会让 MindSpore predict 秒失败
    {
        double mn = 1e30, mx = -1e30;
        int bad = 0;
        for (size_t i = 0; i < mel.size(); ++i) {
            const float v = mel[i];
            if (!std::isfinite(v)) { bad++; } else { if (v < mn) mn = v; if (v > mx) mx = v; }
        }
        LOGI("stage=mel done pcm=%{public}d n=%{public}d bad=%{public}d min=%{public}f max=%{public}f pss=%{public}dMB",
             (int)audio.size(), (int)mel.size(), bad, mn, mx, readPssMb());
    }

    // ---- 2) encoder ----
    OH_AI_TensorHandleArray encIn = OH_AI_ModelGetInputs(g_enc);
    if (encIn.handle_num < 1 || encIn.handle_list == nullptr) {
        out.error = "ENCODER_NO_INPUT";
        return false;
    }
    void *encInData = OH_AI_TensorGetMutableData(encIn.handle_list[0]);
    const size_t encInBytes = (size_t)OH_AI_TensorGetDataSize(encIn.handle_list[0]);
    if (encInData == nullptr || encInBytes != mel.size() * sizeof(float)) {
        out.error = "ENCODER_INPUT_SIZE_MISMATCH";
        return false;
    }
    std::memcpy(encInData, mel.data(), encInBytes);
    LOGI("stage=enc-input ptr=%{public}p bytes=%{public}d", encInData, (int)encInBytes);

    OH_AI_TensorHandleArray encOut;
    std::memset(&encOut, 0, sizeof(encOut));
    OH_AI_Status st = OH_AI_ModelPredict(g_enc, encIn, &encOut, nullptr, nullptr);
    if (st != OH_AI_STATUS_SUCCESS) {
        // one retry: a transient allocation failure often succeeds on the second attempt
        std::memset(&encOut, 0, sizeof(encOut));
        st = OH_AI_ModelPredict(g_enc, encIn, &encOut, nullptr, nullptr);
    }
    if (st != OH_AI_STATUS_SUCCESS || encOut.handle_num < 1 || encOut.handle_list == nullptr) {
        out.needsRebuild = true;   // 让上层重建模型再试（第二次运行内存不足时的自愈）
        LOGE("stage=encoder FAILED status=%{public}d pss=%{public}dMB rss=%{public}dMB",
             (int)st, readPssMb(), readRssMb());
        out.error = "encoder predict status=" + std::to_string((int)st) +
                    " rss=" + std::to_string(readRssMb()) + "MB" +
                    " (mel后=" + std::to_string(out.rssMel) + "MB)";
        return false;
    }
    out.rssEnc = readRssMb();
    LOGI("stage=encoder done pss=%{public}dMB", readPssMb());
    const float *emb = static_cast<const float *>(OH_AI_TensorGetData(encOut.handle_list[0]));
    const int64_t embN = OH_AI_TensorGetElementNum(encOut.handle_list[0]);
    if (emb == nullptr || embN <= 0) {
        out.error = "ENCODER_OUTPUT_EMPTY";
        return false;
    }
    std::vector<float> embCopy(emb, emb + embN);

    // ---- 3) heads ----
    OH_AI_TensorHandleArray hIn = OH_AI_ModelGetInputs(g_heads);
    if (hIn.handle_num < 1 || hIn.handle_list == nullptr) {
        out.error = "HEADS_NO_INPUT";
        return false;
    }
    void *hInData = OH_AI_TensorGetMutableData(hIn.handle_list[0]);
    const size_t hInBytes = (size_t)OH_AI_TensorGetDataSize(hIn.handle_list[0]);
    if (hInData == nullptr || hInBytes != embCopy.size() * sizeof(float)) {
        out.error = "HEADS_INPUT_SIZE_MISMATCH";
        return false;
    }
    std::memcpy(hInData, embCopy.data(), hInBytes);

    OH_AI_TensorHandleArray hOut;
    std::memset(&hOut, 0, sizeof(hOut));
    st = OH_AI_ModelPredict(g_heads, hIn, &hOut, nullptr, nullptr);
    if (st != OH_AI_STATUS_SUCCESS || hOut.handle_num < 2 || hOut.handle_list == nullptr) {
        out.error = "heads predict status=" + std::to_string((int)st);
        return false;
    }
    const float *raw = static_cast<const float *>(OH_AI_TensorGetData(hOut.handle_list[0]));
    const int64_t rawN = OH_AI_TensorGetElementNum(hOut.handle_list[0]);
    const float *spP = static_cast<const float *>(OH_AI_TensorGetData(hOut.handle_list[1]));
    if (raw == nullptr || spP == nullptr || rawN != VH_N_RAW) {
        out.error = "HEADS_OUTPUT_INVALID";
        return false;
    }
    const double speech = (double)spP[0];

    // guard: a broken graph would surface as NaN, which must not become "all metrics = 0"
    for (int64_t i = 0; i < rawN; ++i) {
        if (!std::isfinite(raw[i])) {
            out.error = "MODEL_OUTPUT_NAN";
            return false;
        }
    }
    if (!std::isfinite(speech)) {
        out.error = "SPEECH_PROB_NAN";
        return false;
    }

    // ---- 4) 135 raw head outputs -> 57 integer metrics ----
    for (int i = 0; i < VH_N_CODES; ++i) {
        const int off = VH_OFFSETS[i];
        if (VH_KINDS[i] == 'c') {
            int best = 0;
            for (int j = 1; j < 7; ++j) {
                if (raw[off + j] > raw[off + best]) best = j;
            }
            out.metrics[i] = best;
        } else {
            float v = std::round(raw[off]);
            if (v < 0.f) v = 0.f;
            if (v > 6.f) v = 6.f;
            out.metrics[i] = static_cast<int>(v);
        }
    }

    // ---- 5) GBDT judgement + family attribution ----
    float feats[VH_N_FEATURES];
    for (int i = 0; i < VH_N_FEATURES; ++i) {
        const int c = VH_FEATURE_CODE[i];
        feats[i] = (c < 0) ? (float)speech : (float)out.metrics[c];
    }
    float prob = 0.f;
    int label = 0;
    if (g_attribReady) {
        g_attrib.judge(feats, &prob, &label);
        g_attrib.score(feats, out.fam, out.famShap, nullptr);
    }
    out.prob = prob;
    out.label = label;
    out.speech = speech;
    double n2 = 0;
    for (float v : embCopy) n2 += (double)v * v;
    out.embNorm = std::sqrt(n2);
    out.rssHeads = readRssMb();
    LOGI("stage=heads done pss=%{public}dMB prob=%{public}f label=%{public}d",
         readPssMb(), (double)out.prob, out.label);
    return true;
}

/** Build the JS result object — must run on the JS thread. */
napi_value pipelineToObject(napi_env env, const PipelineOut &o) {
    napi_value obj = nullptr;
    napi_create_object(env, &obj);
    napi_value arr = nullptr;
    napi_create_array_with_length(env, VH_N_CODES, &arr);
    for (int i = 0; i < VH_N_CODES; ++i) {
        napi_value v = nullptr;
        napi_create_int32(env, o.metrics[i], &v);
        napi_set_element(env, arr, i, v);
    }
    napi_set_named_property(env, obj, "metrics", arr);

    napi_value p = nullptr;
    napi_create_double(env, o.prob, &p);
    napi_set_named_property(env, obj, "probability", p);
    napi_value l = nullptr;
    napi_create_int32(env, o.label, &l);
    napi_set_named_property(env, obj, "label", l);
    napi_value sp = nullptr;
    napi_create_double(env, o.speech, &sp);
    napi_set_named_property(env, obj, "speech", sp);

    napi_value famArr = nullptr;
    napi_create_array_with_length(env, 8, &famArr);
    napi_value shapArr = nullptr;
    napi_create_array_with_length(env, 8, &shapArr);
    for (int i = 0; i < 8; ++i) {
        napi_value a = nullptr;
        napi_create_int32(env, (int)o.fam[i], &a);
        napi_set_element(env, famArr, i, a);
        napi_value b = nullptr;
        napi_create_double(env, o.famShap[i], &b);
        napi_set_element(env, shapArr, i, b);
    }
    napi_set_named_property(env, obj, "families", famArr);
    napi_set_named_property(env, obj, "familyShap", shapArr);
    napi_value en = nullptr;
    napi_create_double(env, o.embNorm, &en);
    napi_set_named_property(env, obj, "embeddingNorm", en);

    napi_value r1 = nullptr;
    napi_create_int32(env, o.rssMel, &r1);
    napi_set_named_property(env, obj, "rssMel", r1);
    napi_value r2 = nullptr;
    napi_create_int32(env, o.rssEnc, &r2);
    napi_set_named_property(env, obj, "rssEnc", r2);
    napi_value r3 = nullptr;
    napi_create_int32(env, o.rssHeads, &r3);
    napi_set_named_property(env, obj, "rssHeads", r3);
    napi_value rb = nullptr;
    napi_get_boolean(env, o.rebuilt, &rb);
    napi_set_named_property(env, obj, "rebuilt", rb);
    return obj;
}

// ---------------------------------------------------------------- analyzeAudio(audio) -> result   [sync]
static napi_value AnalyzeAudio(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1] = {nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    std::vector<float> audio;
    if (argc < 1 || !getFloat32Array(env, args[0], audio)) {
        napi_throw_error(env, nullptr, "AUDIO_TYPED_ARRAY_INVALID");
        return makeUndefined(env);
    }
    PipelineOut out;
    if (!computePipeline(audio, out)) {
        g_lastError = out.error;
        napi_throw_error(env, nullptr, out.error.c_str());
        return makeUndefined(env);
    }
    return pipelineToObject(env, out);
}

// ---------------------------------------------------------------- analyzeAudioAsync(audio) -> Promise
// The whole pipeline runs on a NAPI worker thread so the UI keeps animating while the
// encoder works (a synchronous call froze the interface for seconds on device).
struct AsyncCtx {
    napi_async_work work = nullptr;
    napi_deferred deferred = nullptr;
    std::vector<float> audio;
    PipelineOut out;
};

void AsyncExecute(napi_env env, void *data) {
    AsyncCtx *c = static_cast<AsyncCtx *>(data);

    // 上一次分析结束后已释放模型：本次先重建，让每次分析的内存起点一致。
    // （设备上推理缓冲不会真正归还，连续分析会一次次抬高基线，
    //   第 2 次靠自愈能过、第 3 次就不行了。）
    if (!g_graphsLoaded) {
        if (g_encPath.empty()) {
            c->out.error = "GRAPHS_NOT_LOADED";
            return;
        }
        std::string d;
        if (!buildBoth(g_encPath, g_headsPath, g_activeMode, d)) {
            c->out.error = "重建模型失败: " + d;
            return;
        }
        g_graphsLoaded = true;
    }

    if (computePipeline(c->audio, c->out)) {
        return;
    }
    // Self-healing: on device the *second* analysis in a session fails with
    // `encoder predict status=-1` because the previous run's activation buffers are still
    // resident. Rebuild the graphs and retry once.
    if (c->out.needsRebuild && !g_encPath.empty()) {
        std::string d;
        destroyGraphs();
        if (buildBoth(g_encPath, g_headsPath, g_activeMode, d)) {
            PipelineOut retry;
            const bool ok = computePipeline(c->audio, retry);
            retry.rebuilt = true;
            if (!ok) {
                retry.error += " | 重建后仍失败 rss=" + std::to_string(readRssMb()) + "MB";
            }
            c->out = retry;
        }
    }
}

void AsyncComplete(napi_env env, napi_status status, void *data) {
    AsyncCtx *c = static_cast<AsyncCtx *>(data);
    if (c->out.error.empty()) {
        napi_value obj = pipelineToObject(env, c->out);
        napi_resolve_deferred(env, c->deferred, obj);
    } else {
        g_lastError = c->out.error;
        napi_value msg = nullptr;
        napi_create_string_utf8(env, c->out.error.c_str(), c->out.error.size(), &msg);
        napi_value err = nullptr;
        napi_create_error(env, nullptr, msg, &err);
        napi_reject_deferred(env, c->deferred, err);
    }
    napi_delete_async_work(env, c->work);
    delete c;

    // 主动释放模型，让下一次分析从干净的内存状态开始
    destroyGraphs();
    g_graphsLoaded = false;
}

static napi_value AnalyzeAudioAsync(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1] = {nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    AsyncCtx *c = new AsyncCtx();
    if (argc < 1 || !getFloat32Array(env, args[0], c->audio)) {
        delete c;
        napi_throw_error(env, nullptr, "AUDIO_TYPED_ARRAY_INVALID");
        return makeUndefined(env);
    }
    napi_value promise = nullptr;
    napi_create_promise(env, &c->deferred, &promise);
    napi_value resName = nullptr;
    napi_create_string_utf8(env, "voiceHealthAnalyze", NAPI_AUTO_LENGTH, &resName);
    napi_create_async_work(env, nullptr, resName, AsyncExecute, AsyncComplete, c, &c->work);
    napi_queue_async_work(env, c->work);
    return promise;
}

// ---------------------------------------------------------------- module registration
EXTERN_C_START
static napi_value Init(napi_env env, napi_value exports) {
    napi_property_descriptor desc[] = {
        {"initMel", nullptr, InitMel, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"initModel", nullptr, InitModel, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"loadGraphs", nullptr, LoadGraphs, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"analyzeAudio", nullptr, AnalyzeAudio, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"analyzeAudioAsync", nullptr, AnalyzeAudioAsync, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"status", nullptr, Status, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"currentRss", nullptr, CurrentRss, nullptr, nullptr, nullptr, napi_default, nullptr},
    };
    napi_define_properties(env, exports, sizeof(desc) / sizeof(desc[0]), desc);
    return exports;
}
EXTERN_C_END

static napi_module voiceHealthModule = {
    .nm_version = 1,
    .nm_flags = 0,
    .nm_filename = nullptr,
    .nm_register_func = Init,
    .nm_modname = "voicehealth",
    .nm_priv = nullptr,
    .reserved = {0},
};

extern "C" __attribute__((constructor)) void RegisterVoiceHealthModule(void) {
    napi_module_register(&voiceHealthModule);
}
