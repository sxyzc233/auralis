/**
 * 声析 Auralis · HarmonyOS 本地推理版
 *
 * 音频全程在本机处理：log-mel（C++/NAPI）→ encoder.ms → heads.ms（MindSpore Lite）
 * → 57 项指标后处理 + GBDT 判定 + 家族归因（同一份 C++/NAPI）→ 四屏报告。
 */
import { audio } from '@kit.AudioKit';
import { fileIo as fs } from '@kit.CoreFileKit';
import { abilityAccessCtrl, common, Permissions } from '@kit.AbilityKit';
import { BusinessError } from '@kit.BasicServicesKit';
import voicehealth from 'libvoicehealth.so';
import { hidebug } from '@kit.PerformanceAnalysisKit';
import { bundleManager, appMemoryOptimizer } from '@kit.AbilityKit';
import { deviceInfo } from '@kit.BasicServicesKit';
import { HistoryStore, Rec, recordsOfDay, dayKey, hhmm, monthInfo } from '../common/HistoryStore';
import { TH, READING_TEXT, scoreTone } from '../common/Theme';
import { BrandMark, VoiceArt, TimerRing, LiveWave, TrackView } from '../common/AuralisViz';

// ---------------------------------------------------------------- native binding
interface AnalyzeResult {
  metrics: number[];
  probability: number;
  label: number;
  speech: number;
  families: number[];
  familyShap: number[];
  embeddingNorm: number;
  rssMel: number;
  rssEnc: number;
  rssHeads: number;
  rebuilt: boolean;
}

interface NativeApi {
  initMel(path: string): boolean;
  initModel(path: string): boolean;
  /** builds both .ms graphs; returns "" on success or a failure note */
  loadGraphs(encPath: string, headsPath: string): string;
  /** log-mel -> encoder -> heads -> 57 metrics -> GBDT judgement -> family SHAP, all native */
  analyzeAudio(audio: Float32Array): AnalyzeResult;
  analyzeAudioAsync(audio: Float32Array): Promise<AnalyzeResult>;
  status(): string;
  currentRss(): number;
}
const native: NativeApi = voicehealth as NativeApi;

// ---------------------------------------------------------------- constants
const THRESHOLD: number = 0.30177322;
const MEL_LEN: number = 80 * 3000;

const CODES: string[] = [
  'AGEV', 'AROU', 'ARSH', 'ATCK', 'BKGN', 'BRGT', 'CHNK', 'CLRT', 'COGL', 'DARC',
  'DFLU', 'EMPH', 'ESTH', 'EXPL', 'FOCS', 'FULL', 'GEND', 'HARM', 'METL', 'R_CHST',
  'R_HEAD', 'R_MASK', 'R_MIXD', 'R_NASL', 'R_ORAL', 'R_THRT', 'RANG', 'RCQL', 'REGS', 'RESP',
  'ROUG', 'S_ASMR', 'S_AUTH', 'S_CART', 'S_CASU', 'S_CONV', 'S_DRAM', 'S_FORM', 'S_MONO', 'S_NARR',
  'S_NEWS', 'S_PLAY', 'S_RANT', 'S_STRY', 'S_TECH', 'S_WHIS', 'SMTH', 'STNC', 'STRU', 'TEMP',
  'TENS', 'VALN', 'VALS', 'VFLX', 'VOLT', 'VULN', 'WARM'
];
const FAMILIES: string[] = ['音质', '共鸣', '清晰流畅', '情绪张力', '表达节奏', '语音含量', '风格内容', '录音条件'];

// 专业风 (ui-新) 与 手绘贴纸风 (ui2-global.wxss) 两套配色
/**
 * 调色板：全部对齐安卓版新 UI（AuralisUi.kt）的 42 个色 token。
 * 字段名沿用旧版，避免大范围改动；nav 由"深色导航栏"改为新设计的"浅色顶栏"。
 */
class Palette {
  primary: string = TH.brandGreen;        // #477660 主操作色
  primaryDark: string = TH.barDark;       // #426850
  teal: string = TH.ringFill;             // #83A58E 辅助绿
  amber: string = TH.scoreTextWarn;       // #83632A
  rose: string = TH.errorFg;              // #9B4F35
  blue: string = TH.metricValue;          // #4C6957
  bg: string = TH.paper;                  // #F8FAF8 纸感底
  card: string = '#FFFFFF';
  border: string = TH.line;               // #E1E9E3
  text: string = TH.ink;                  // #19383D
  sub: string = TH.secondary;             // #5B7062
  muted: string = TH.brandCaption;        // #86928B
  track: string = TH.trackBg;             // #E0E8DF
  /** 新设计顶栏是浅色的（纸底墨字），不再是深色色块 */
  navBg: string = TH.paper;
  navText: string = TH.ink;

  // ---- 新设计专用语义色 ----
  scorePanelBg: string = TH.scorePanelBg;
  scorePanelWarn: string = TH.scorePanelWarn;
  scorePanelBad: string = TH.scorePanelBad;
  scoreGreen: string = TH.scoreGreen;
  scoreUnit: string = TH.scoreUnit;
  readingBg: string = TH.readingBg;
  readingInk: string = TH.readingInk;
  adviceIconBg: string = TH.adviceIconBg;
  adviceIconFg: string = TH.adviceIconFg;
  adviceText: string = TH.adviceText;
  haloBorder: string = TH.haloBorder;
  iconButtonBg: string = TH.iconButtonBg;
  iconButtonFg: string = TH.iconButtonFg;
  secondaryButtonBg: string = TH.secondaryButtonBg;
  secondaryButtonFg: string = TH.secondaryButtonFg;
  errorBg: string = TH.errorBg;
  eyebrow: string = TH.eyebrowColor;
  howToBorder: string = TH.howToBorder;
  howToNumber: string = TH.howToNumber;
  stepAccent: string = TH.stepAccent;
  stepLabel: string = TH.stepLabel;
  metaDivider: string = TH.metaDivider;

  static pro(): Palette {
    return new Palette();
  }

  static sticker(): Palette {
    return new Palette();
  }
}

interface Bar {
  name: string;
  value: number;
  text: string;
  color: string;
}

interface Narration {
  risk: string;
  summary: string;
  suggestions: string[];
  warnings: string[];
}

function narrate(prob: number, shap: number[]): Narration {
  const margin: number = prob - THRESHOLD;
  let risk: string = '';
  if (prob >= THRESHOLD && margin >= 0.15) {
    risk = '中等偏不健康';
  } else if (prob >= THRESHOLD) {
    risk = '临界，略偏向不健康';
  } else if (margin > -0.15) {
    risk = '临界，略偏向健康';
  } else {
    risk = '偏健康，风险较低';
  }
  let lead: string = '';
  if (risk.indexOf('偏健康') === 0) {
    lead = '这次录音整体看起来比较健康，我们的判定是「健康」方向。';
  } else if (risk.indexOf('临界') === 0) {
    lead = '这次录音处于临界区间，属于边界情况，建议结合前后的变化一起看。';
  } else {
    lead = '这次录音的信号偏「不健康」，标注了主要让判定上升的因素。';
  }
  lead += ' 模型给出的不健康概率约 ' + Math.round(prob * 100).toString() + '%。';

  const patho: boolean[] = [true, true, true, true, true, false, false, false];
  const advice: string[] = [
    '近期控制用嗓强度，多喝温水、保证睡眠，避免烟酒与大声喊叫。',
    '练习腹式呼吸与放松喉部，多做哼鸣、头腔共鸣练习，避免声带硬挤。',
    '留意因疲劳、压力导致的口齿不清，必要时放慢语速。',
    '用嗓前放松肩颈与喉部，减少压力和喊叫，让声音更松弛。',
    '放慢语速、学会控制呼气，让气息更稳。',
    '', '', ''
  ];
  const confNote: string[] = [
    '', '', '', '', '',
    '',   // 语音含量：按要求不再提示
    '',   // 风格内容：按要求不再提示
    '录音条件：多为录音环境或个体差异，请勿直接当作病变，属混淆因素。'
  ];
  const suggestions: string[] = [];
  const warnings: string[] = [];
  for (let i = 0; i < FAMILIES.length; i++) {
    const s: number = (i < shap.length) ? shap[i] : 0;
    if (s <= 0) {
      continue;
    }
    if (patho[i]) {
      if (advice[i].length > 0) {
        suggestions.push(advice[i]);
      }
    } else if (confNote[i].length > 0) {
      warnings.push(confNote[i]);
    }
  }
  if (suggestions.length === 0) {
    suggestions.push('目前判断更接近健康，建议保持规律用嗓、多补水、注意休息即可。');
  }
  warnings.push('以上为嗓音数据的初筛参考，判别力中等（AUROC≈0.69），不能代替专业医学诊断；如有持续嘶哑/疼痛请及时就医。');
  return { risk: risk, summary: lead, suggestions: suggestions, warnings: warnings };
}

function metric4(metrics: number[], code: string, flip: boolean): number {
  const i: number = CODES.indexOf(code);
  if (i < 0) {
    return 0;
  }
  const v: number = metrics[i] / 6 * 4;
  return flip ? (4 - v) : v;
}

@Entry
@Component
struct Index {
  @State screen: string = 'home';
  @State status: string = '正在初始化…';
  @State liveRss: number = 0;
  @State memText: string = '读取中…';
  private apiVersion: number = 0;
  private rssTimer: number = -1;
  @State ready: boolean = false;
  // 界面切换的横向位移动画
  @State offsetX: number = 0;
  @State reportSource: string = 'home';
  // 历史档案
  @State history: Rec[] = [];  @State calYear: number = new Date().getFullYear();
  @State calMonth: number = new Date().getMonth() + 1;
  @State selDay: number = new Date().getDate();
  @State npuNote: string = '';

  // recording
  @State left: number = 30;
  @State progress: number = 0;
  @State db: string = '--';
  @State bars: number[] = new Array<number>(24).fill(4);

  // report
  @State healthScore: number = 0;
  @State healthyPct: number = 0;
  @State riskPct: number = 0;
  @State speechPct: number = 0;
  @State radar: number[] = [0, 0, 0, 0, 0];
  @State narration: Narration = { risk: '', summary: '', suggestions: [], warnings: [] };
  @State families: number[] = new Array<number>(8).fill(0);
  @State elapsedMs: number = 0;
  @State diag: string = '';

  // error reporting
  @State stage: string = '';
  @State errorMsg: string = '';

  private ctx: common.UIAbilityContext = getContext(this) as common.UIAbilityContext;
  // must be @State: reassigning a plain field would not trigger a re-render (style switch did nothing)
  private pal: Palette = Palette.pro();
  private capturer: audio.AudioCapturer | null = null;
  private store: HistoryStore | null = null;
  /** 录音起始时刻：倒计时按真实时间走，不依赖是否收到音频 */
  private recStartMs: number = 0;
  /** 麦克风启动失败标记：在录音页给出明确提示 */
  @State micFailed: boolean = false;
  private pcm: number[] = [];
  private recording: boolean = false;
  private timer: number = -1;

  aboutToAppear(): void {
    try {
      this.apiVersion = deviceInfo.sdkApiVersion;   // 用于按系统版本降级
      console.info('[VH] sdkApiVersion=' + this.apiVersion.toString());
    } catch (e) {
      this.apiVersion = 0;
    }
    // 首页实时显示真实内存占用（每 1.5 秒取一次原生进程 RSS）
    this.rssTimer = setInterval(() => {
      try {
        this.liveRss = native.currentRss();
      } catch (e) {
        this.liveRss = -1;
      }
      this.memText = this.readMem();
    }, 1500);
    this.pal = Palette.pro();
    this.prepare();
  }

  aboutToDisappear(): void {
    if (this.rssTimer >= 0) {
      clearInterval(this.rssTimer);
      this.rssTimer = -1;
    }
    this.stopCapture();
  }

  /**
   * 释放 HAP 的文件页缓存。
   *
   * prepare() 会把 encoder.ms(174MB) + heads.ms(26MB) 从 rawfile 解压到沙箱，
   * 这个读取过程把整个 HAP 的文件页拉进了进程。设备实测 hidumper：
   *     .hap 段 PSS = 205184 kB（≈200.4MB，正好等于两个模型之和），且是 Private Clean
   * 模型已经落在沙箱，HAP 里这份映射毫无用处，却是实打实的 200MB 占用，
   * 还招来系统的 memory.force_shrink_all。API 26 起可主动驱逐。
   */
  private async evictHapPages(): Promise<void> {
    // appMemoryOptimizer 是 API 26 新增（HarmonyOS 6.1 = API 23 等设备没有）。
    // 版本不足时直接跳过：该优化只是锦上添花，缺了不影响推理正确性。
    if (this.apiVersion > 0 && this.apiVersion < 26) {
      return;
    }
    try {
      const flag: number = bundleManager.BundleFlag.GET_BUNDLE_INFO_WITH_HAP_MODULE;
      const info = bundleManager.getBundleInfoForSelfSync(flag);
      const paths: string[] = [];
      const mods = info.hapModulesInfo;
      for (let i = 0; i < mods.length; i++) {
        paths.push(mods[i].codePath);
      }
      if (paths.length > 0) {
        await appMemoryOptimizer.evictFilePages(paths);
        console.info('[VH] evictFilePages ok: ' + JSON.stringify(paths));
      }
    } catch (e) {
      const err = e as BusinessError;
      console.warn('[VH] evictFilePages failed: ' + (err.message !== undefined ? err.message : JSON.stringify(e)));
    }
  }

  /** 读取系统真实内存指标：PSS 才是实际占用，RSS 会把共享文件页全额计入。 */
  private readMem(): string {
    try {
      const pssMb: number = Number(hidebug.getPss()) / 1024;
      const rssMb: number = this.liveRss;
      const sys = hidebug.getSystemMemInfo();
      const totalGb: number = Number(sys.totalMem) / 1048576;
      const availGb: number = Number(sys.availableMem) / 1048576;
      const lim = hidebug.getAppMemoryLimit();
      const rlMb: number = Number(lim.rssLimit) / 1024;
      const vmMb: number = Number(lim.vmHeapLimit) / 1024;
      let gm = '';
      try {
        gm = ' · 图形 ' + (hidebug.getGraphicsMemorySync() / 1048576).toFixed(0) + 'MB';
      } catch (e2) {
        gm = '';
      }
      return '实际占用 PSS ' + pssMb.toFixed(0) + 'MB（RSS ' + rssMb.toString() + 'MB）'
        + ' · 系统可用 ' + availGb.toFixed(1) + '/' + totalGb.toFixed(1) + 'GB'
        + ' · 进程上限 ' + (rlMb > 0 ? rlMb.toFixed(0) + 'MB' : '未限制')
        + ' · VM堆 ' + (vmMb > 0 ? vmMb.toFixed(0) + 'MB' : '—') + gm;
    } catch (e) {
      return '内存指标不可用（API 版本较低？）';
    }
  }

  /** 系统返回键：非首页逐级回退，首页交还系统（回到桌面） */
  onBackPress(): boolean {
    if (this.screen === 'home') {
      return false;
    }
    if (this.screen === 'proc') {
      return true;            // 正在分析，忽略返回
    }
    if (this.screen === 'rec') {
      this.cancelCapture();   // 取消录音并回到首页
      return true;
    }
    if (this.screen === 'report') {
      this.goto(this.reportSource, false);
      return true;
    }
    this.goto('home', false); // history / error
    return true;
  }

  /** 界面切换：旧页向左/右平移出场，新页从另一侧平移入场 */
  private goto(next: string, forward: boolean): void {
    if (this.screen === next) {
      return;
    }
    const w: number = 360;
    const dir: number = forward ? 1 : -1;
    animateTo({
      duration: 170,
      curve: Curve.EaseIn,
      onFinish: () => {
        this.screen = next;
        this.offsetX = dir * w;
        animateTo({ duration: 230, curve: Curve.EaseOut }, () => {
          this.offsetX = 0;
        });
      }
    }, () => {
      this.offsetX = -dir * w;
    });
  }

  // ---------------------------------------------------------------- setup
  private async prepare(): Promise<void> {
    try {
      const resMgr = this.ctx.resourceManager;
      if (this.store === null) {
        this.store = new HistoryStore(this.ctx.filesDir);
      }
      this.history = this.store.load();

      // mel filter bank (63 KB): copy to the sandbox and let the native side read it.
      // Passing it as a Float32Array across the NAPI boundary silently failed on device.
      const mf: Uint8Array = await resMgr.getRawFileContent('mel_filters.f32');
      const mfPath: string = this.ctx.filesDir + '/mel_filters.f32';
      if (this.fileSize(mfPath) !== mf.byteLength) {
        const mfBytes: ArrayBuffer = mf.buffer.slice(mf.byteOffset, mf.byteOffset + mf.byteLength);
        const mfDst = fs.openSync(mfPath, fs.OpenMode.READ_WRITE | fs.OpenMode.CREATE | fs.OpenMode.TRUNC);
        fs.writeSync(mfDst.fd, mfBytes);
        fs.closeSync(mfDst);
      }
      const melOk: boolean = native.initMel(mfPath);
      if (!melOk) {
        throw new Error('mel 滤波器加载失败 (' + mf.byteLength + ' 字节)');
      }

      // attribution / judgement model (104 KB) must be a real file for fopen()
      const ab: Uint8Array = await resMgr.getRawFileContent('attrib.bin');
      const abPath: string = this.ctx.filesDir + '/attrib.bin';
      const af = fs.openSync(abPath, fs.OpenMode.READ_WRITE | fs.OpenMode.CREATE | fs.OpenMode.TRUNC);
      fs.writeSync(af.fd, ab.buffer);
      fs.closeSync(af);
      const ok: boolean = native.initModel(abPath);

      // neural graphs ship inside the HAP and are released into the sandbox on first launch
      const encPath: string = this.ctx.filesDir + '/encoder.ms';
      const headsPath: string = this.ctx.filesDir + '/heads.ms';
      const encRaw = await resMgr.getRawFd('encoder.ms');
      if (this.fileSize(encPath) !== encRaw.length) {
        this.status = '正在释放模型 encoder.ms（约 348 MB，仅首次）…';
        await this.extractRawfile('encoder.ms', encPath, encRaw.fd, encRaw.offset, encRaw.length);
      }
      const headsRaw = await resMgr.getRawFd('heads.ms');
      if (this.fileSize(headsPath) !== headsRaw.length) {
        this.status = '正在释放模型 heads.ms（约 52 MB，仅首次）…';
        await this.extractRawfile('heads.ms', headsPath, headsRaw.fd, headsRaw.offset, headsRaw.length);
      }
      this.status = '正在加载模型…';
      const note: string = native.loadGraphs(encPath, headsPath);
      if (note.length > 0) {
        throw new Error(note);
      }
      this.ready = true;
      // 模型已在沙箱，HAP 里那 200MB 映射页现在没有用了，立刻让系统回收
      await this.evictHapPages();
      this.status = ok ? '模型已就绪' : '归因层未加载';
      this.npuNote = native.status();
      this.diag = 'native: ' + this.npuNote;
    } catch (e) {
      const err = e as BusinessError;
      this.status = '初始化失败: ' + err.message;
      this.ready = false;
    }
  }

  /** Size of a sandbox file, or -1 when it does not exist yet. */
  private fileSize(path: string): number {
    try {
      return fs.statSync(path).size;
    } catch (e) {
      return -1;
    }
  }

  /**
   * Copy a rawfile out of the HAP in 4 MB chunks.
   * Reading a ~350 MB rawfile with getRawFileContent would put the whole thing on the JS heap,
   * so instead we seek the rawfile fd to the resource offset and stream it to the sandbox.
   */
  private async extractRawfile(name: string, destPath: string, fd: number,
                               offset: number, length: number): Promise<void> {
    const dst = fs.openSync(destPath, fs.OpenMode.READ_WRITE | fs.OpenMode.CREATE | fs.OpenMode.TRUNC);
    const CHUNK: number = 4 * 1024 * 1024;
    const buf: ArrayBuffer = new ArrayBuffer(CHUNK);
    fs.lseek(fd, offset, fs.WhenceType.SEEK_SET);
    let done: number = 0;
    while (done < length) {
      const want: number = Math.min(CHUNK, length - done);
      const n: number = fs.readSync(fd, buf, { length: want });
      if (n <= 0) {
        break;
      }
      fs.writeSync(dst.fd, buf, { length: n });
      done += n;
    }
    fs.closeSync(dst);
    if (this.fileSize(destPath) !== length) {
      throw new Error('释放 ' + name + ' 失败');
    }
  }

  /** Switch the inference device and rebuild both graphs (device selection is native now). */


  // ---------------------------------------------------------------- recording
  private async startCapture(): Promise<void> {
    try {
      await this.doStartCapture();
    } catch (e) {
      const err = e as BusinessError;
      this.status = '启动录音失败: ' + (err.message !== undefined ? err.message : JSON.stringify(e));
      this.goto('home', false);
    }
  }

  private async doStartCapture(): Promise<void> {
    const atManager = abilityAccessCtrl.createAtManager();
    const perms: Permissions[] = ['ohos.permission.MICROPHONE'];
    try {
      const res = await atManager.requestPermissionsFromUser(this.ctx, perms);
      if (res.authResults.length > 0 && res.authResults[0] !== 0) {
        this.status = '需要麦克风权限';
        return;
      }
    } catch (e) {
      this.status = '麦克风授权失败';
      return;
    }
    const streamInfo: audio.AudioStreamInfo = {
      samplingRate: audio.AudioSamplingRate.SAMPLE_RATE_16000,
      channels: audio.AudioChannel.CHANNEL_1,
      sampleFormat: audio.AudioSampleFormat.SAMPLE_FORMAT_S16LE,
      encodingType: audio.AudioEncodingType.ENCODING_TYPE_RAW
    };
    const capturerInfo: audio.AudioCapturerInfo = {
      source: audio.SourceType.SOURCE_TYPE_MIC,
      capturerFlags: 0
    };
    const options: audio.AudioCapturerOptions = { streamInfo: streamInfo, capturerInfo: capturerInfo };
    // 若上一次的采集器还在，先彻底停掉：否则会有多个采集器同时往 this.pcm 推数据，
    // 采样数翻倍（实测出现过 1920000 = 4×480000），且释放后的缓冲区会推入垃圾浮点。
    this.stopCapture();
    this.capturer = await audio.createAudioCapturer(options);
    this.pcm = [];
    this.recording = true;
    this.left = 30;
    this.progress = 0;
    this.goto('rec', true);

    const cap = this.capturer;
    cap.on('readData', (buffer: ArrayBuffer) => {
      if (!this.recording) {
        return;
      }
      const i16 = new Int16Array(buffer);
      let sum = 0;
      const heights: number[] = new Array<number>(24).fill(4);
      const lim: number = 16000 * 30;
      for (let i = 0; i < i16.length; i++) {
        // 硬上限：绝不让数组越过 30 秒，避免多次录音叠加与越界读取
        if (this.pcm.length >= lim) {
          break;
        }
        const v = i16[i] / 32768;
        this.pcm.push(v);
        sum += v * v;
      }
      const rms = Math.sqrt(sum / Math.max(1, i16.length));
      const dB = rms > 0.000001 ? 20 * Math.log10(rms) : -90;
      this.db = dB.toFixed(0);
      for (let b = 0; b < 24; b++) {
        const idx = Math.min(i16.length - 1, Math.floor(b / 24 * i16.length));
        heights[b] = Math.max(4, Math.min(100, Math.abs(i16[idx] / 32768) * 100));
      }
      this.bars = heights;
      if (this.pcm.length >= 16000 * 30) {
        this.finishCapture();
      }
    });
    // 倒计时按"真实经过时间"计算，并且必须在 cap.start() 之前启动：
    // 麦克风被其他应用占用时 cap.start() 会长时间等待甚至失败，
    // 若定时器写在它后面，界面就会永远停在 30 秒不动。
    this.recStartMs = Date.now();
    this.timer = setInterval(() => {
      const seconds = (Date.now() - this.recStartMs) / 1000;
      this.progress = Math.min(1, seconds / 30);
      this.left = Math.max(0, Math.ceil(30 - seconds));
      if (seconds >= 30) {
        this.finishCapture();
      }
    }, 200);

    // 采集器启动失败（典型原因：麦克风被微信/录音机等占用）时给出可读提示，
    // 而不是让用户面对一个不动的倒计时。
    try {
      await cap.start();
    } catch (e) {
      const err = e as BusinessError;
      this.status = '麦克风启动失败：' + (err.message !== undefined ? err.message : JSON.stringify(e))
        + '。请确认没有其他应用正在使用麦克风。';
      this.micFailed = true;
    }
  }

  private stopCapture(): void {
    this.recording = false;
    if (this.timer >= 0) {
      clearInterval(this.timer);
      this.timer = -1;
    }
    if (this.capturer !== null) {
      this.capturer.stop();
      this.capturer.release();
      this.capturer = null;
    }
  }

  private cancelCapture(): void {
    this.stopCapture();
    this.pcm = [];
    this.goto('home', false);
  }

  private finishCapture(): void {
    if (!this.recording) {
      return;
    }
    this.stopCapture();
    this.goto('proc', true);
    // 等过渡动画走完再发起推理，保证「本地模型分析中」先渲染出来
    setTimeout(() => {
      this.analyze();
    }, 260);
  }

  // ---------------------------------------------------------------- inference
  private async analyze(): Promise<void> {
    const t0 = Date.now();
    try {
      // 推理前再驱逐一次 HAP 文件页（保险，成本极低）
      await this.evictHapPages();
      // 只取前 30 秒，并把非有限值（NaN/Inf，来自释放后的采集缓冲区）清零
      const raw = Float32Array.from(this.pcm);
      this.pcm = [];
      // 空/过短音频：直接给出可读原因，避免把空数组送进原生层报 AUDIO_TYPED_ARRAY_INVALID
      if (raw.length < 1600) {
        throw new Error('没有采集到声音（仅 ' + raw.length.toString() + ' 个采样，约 '
          + (raw.length / 16000).toFixed(2) + ' 秒）。请确认麦克风没有被其他应用（如微信、录音机）占用，然后重试。');
      }
      const cap30: number = Math.min(raw.length, 16000 * 30);
      const pcm = new Float32Array(cap30);
      let dirty = 0;
      for (let i = 0; i < cap30; i++) {
        const v = raw[i];
        if (v === v && v > -1e6 && v < 1e6) {
          pcm[i] = v;
        } else {
          pcm[i] = 0;
          dirty++;
        }
      }
      console.info('[VH] audio raw=' + raw.length.toString() + ' cap=' + cap30.toString()
        + ' out=' + pcm.length.toString() + ' dirty=' + dirty.toString());
      this.diag = native.status() + ' pcm=' + pcm.length;

      this.stage = 'log-mel → encoder → heads → 判定 → 归因';
      const res: AnalyzeResult = await native.analyzeAudioAsync(pcm);
      if (res === undefined || res === null) {
        throw new Error('原生推理返回空');
      }
      this.elapsedMs = Date.now() - t0;
      this.diag += ' prob=' + res.probability.toFixed(4) + ' sp=' + res.speech.toFixed(3) +
        ' |emb|=' + res.embeddingNorm.toFixed(3) +
        ' rss mel/enc/heads=' + res.rssMel + '/' + res.rssEnc + '/' + res.rssHeads + 'MB' +
        (res.rebuilt ? ' (已重建模型)' : '');

      this.healthScore = Math.max(0, Math.min(100, Math.round((1 - res.probability) * 100)));
      this.riskPct = Math.round(res.probability * 100);
      this.healthyPct = 100 - this.riskPct;
      this.speechPct = Math.round(res.speech * 100);
      this.families = res.families;
      this.radar = [
        metric4(res.metrics, 'ROUG', true),
        metric4(res.metrics, 'HARM', false),
        metric4(res.metrics, 'TENS', true),
        metric4(res.metrics, 'RESP', false),
        metric4(res.metrics, 'RCQL', false)
      ];
      this.narration = narrate(res.probability, res.familyShap);
      this.stage = '';

      // 存入本地嗓音档案（日历记录），不上云
      const now: number = Date.now();
      const rec: Rec = {
        id: now,
        ts: now,
        score: this.healthScore,
        prob: res.probability,
        label: res.label,
        speech: res.speech,
        radar: this.radar,
        risk: this.narration.risk,
        summary: this.narration.summary,
        suggestions: this.narration.suggestions,
        warnings: this.narration.warnings,
        families: res.families
      };
      if (this.store !== null) {
        this.history = this.store.add(rec);
      }
      this.reportSource = 'home';   // 新评测的报告，返回时回首页
      this.goto('report', true);
    } catch (e) {
      const err = e as BusinessError;
      this.errorMsg = (err.message !== undefined ? err.message : JSON.stringify(e));
      this.goto('error', true);
    }
  }

  // ---------------------------------------------------------------- local archive
  private shiftMonth(delta: number): void {
    let y = this.calYear;
    let m = this.calMonth + delta;
    if (m < 1) {
      m = 12;
      y -= 1;
    } else if (m > 12) {
      m = 1;
      y += 1;
    }
    this.calYear = y;
    this.calMonth = m;
    this.selDay = 1;
  }

  /** 日历格子：0 表示空位 */
  private calCells(): number[] {
    const info: number[] = monthInfo(this.calYear, this.calMonth);
    const cells: number[] = [];
    for (let i = 0; i < info[0]; i++) {
      cells.push(0);
    }
    for (let d = 1; d <= info[1]; d++) {
      cells.push(d);
    }
    while (cells.length % 7 !== 0) {
      cells.push(0);
    }
    return cells;
  }

  /** 当天平均健康分，-1 表示没有记录 */
  private scoreOfDay(d: number): number {
    const recs: Rec[] = recordsOfDay(this.history, this.calYear, this.calMonth, d);
    if (recs.length === 0) {
      return -1;
    }
    let sum = 0;
    for (let i = 0; i < recs.length; i++) {
      sum += recs[i].score;
    }
    return Math.round(sum / recs.length);
  }

  private recsOfSelDay(): Rec[] {
    return recordsOfDay(this.history, this.calYear, this.calMonth, this.selDay);
  }

  private toneColor(score: number): string {
    if (score >= 70) {
      return this.pal.teal;
    }
    if (score >= 40) {
      return this.pal.amber;
    }
    return this.pal.rose;
  }

  /** 从本地档案打开一份历史报告 */
  private openRec(r: Rec): void {
    this.healthScore = r.score;
    this.riskPct = Math.round(r.prob * 100);
    this.healthyPct = 100 - this.riskPct;
    this.speechPct = Math.round(r.speech * 100);
    this.radar = r.radar;
    this.families = r.families;
    this.narration = {
      risk: r.risk,
      summary: r.summary,
      suggestions: r.suggestions,
      warnings: r.warnings
    };
    this.elapsedMs = 0;
    this.reportSource = 'history';
    this.diag = '历史档案 · ' + dayKey(r.ts) + ' ' + hhmm(r.ts);
    this.goto('report', true);
  }

  // ---------------------------------------------------------------- ui helpers
  @Builder
  cardHead(icon: string, title: string, chip: string) {
    Row() {
      Text(icon).fontSize(16)
      Text(title).fontSize(15).fontWeight(FontWeight.Bold).fontColor(this.pal.text).margin({ left: 6 })
      Blank()
      if (chip.length > 0) {
        Text(chip).fontSize(10).fontColor(this.pal.muted)
          .backgroundColor(this.pal.track).borderRadius(20).padding({ left: 8, right: 8, top: 3, bottom: 3 })
      }
    }.width('100%').margin({ bottom: 10 })
  }

  @Builder
  card(icon: string, title: string, chip: string) {
    Column() {
      this.cardHead(icon, title, chip)
    }
  }

  @Builder
  bar(name: string, text: string, pct: number, color: string) {
    Column() {
      Row() {
        Text(name).fontSize(13).fontColor(this.pal.sub).layoutWeight(1)
        Text(text).fontSize(13).fontWeight(FontWeight.Bold).fontColor(color)
      }.width('100%')
      Row() {
        Row().width(Math.max(0, Math.min(100, pct)) + '%').height(7)
          .backgroundColor(color).borderRadius(4)
      }.width('100%').height(7).backgroundColor(this.pal.track).borderRadius(4).margin({ top: 5 })
    }.width('100%').margin({ bottom: 10 })
  }

  @Builder
  statusCard() {
    Column() {
      Row() {
        Text('📋').fontSize(15)
        Text('评测说明').fontSize(14).fontWeight(FontWeight.Bold).fontColor(this.pal.text).margin({ left: 6 })
      }.width('100%')
      Text('· 请在安静环境下，用平时自然的语速朗读').fontSize(12).fontColor(this.pal.sub).margin({ top: 6 })
      Text('· 录音最长 30 秒，读完即可点「提前完成并分析」').fontSize(12).fontColor(this.pal.sub).margin({ top: 4 })
      Text('· 每次评测结果会自动记入嗓音日历，可随时回看').fontSize(12).fontColor(this.pal.sub).margin({ top: 4 })
      Text('· 模型可选 NPU 加速（NNRT）').fontSize(12).fontColor(this.pal.sub).margin({ top: 4 })
      Text('状态：' + this.status).fontSize(11).fontColor(this.pal.muted).margin({ top: 8 })
      Text(this.memText).fontSize(11).fontColor(this.pal.muted).margin({ top: 4 })
      if (this.npuNote.length > 0) {
        Text(this.npuNote).fontSize(11).fontColor(this.pal.muted).margin({ top: 4 })
      }
    }
    .width('100%').padding(14).backgroundColor(this.pal.card)
    .borderRadius(12).border({ width: 1, color: this.pal.border }).margin({ top: 28, bottom: 12 })
  }

  /** 采集元信息一行 */
  @Builder
  captureMeta() {
    Row() {
      Text('最长 30 秒').fontSize(10).fontColor(this.pal.muted)
      Text('·').fontSize(10).fontColor(this.pal.metaDivider).margin({ left: 8, right: 8 })
      Text('无需登录').fontSize(10).fontColor(this.pal.muted)
      Text('·').fontSize(10).fontColor(this.pal.metaDivider).margin({ left: 8, right: 8 })
      Text('离线分析').fontSize(10).fontColor(this.pal.muted)
    }
    .width('100%').justifyContent(FlexAlign.Center).margin({ top: 19 })
  }

  /** 评测说明：三步 */
  @Builder
  howTo() {
    Column() {
      Text('评测说明').fontSize(11).fontWeight(FontWeight.Bold)
        .fontColor(this.pal.eyebrow).letterSpacing(1.2).width('100%')
      this.stepRow('01', '自然朗读', '在安静环境中朗读一段短文')
      this.stepRow('02', '本机分析', '音频不上传，全部在设备端完成')
      this.stepRow('03', '查看结果', '得到嗓音健康分与五维声学雷达')
    }
    .width('100%').padding(18)
    .border({ width: 1, color: this.pal.howToBorder }).borderRadius(16)
  }

  @Builder
  stepRow(num: string, title: string, desc: string) {
    Row() {
      Text(num).fontSize(11).fontWeight(FontWeight.Bold)
        .fontColor(this.pal.howToNumber).width(26)
      Column() {
        Text(title).fontSize(13).fontWeight(FontWeight.Bold).fontColor(this.pal.text)
        Text(desc).fontSize(11).fontColor(this.pal.sub).margin({ top: 2 })
      }.alignItems(HorizontalAlign.Start).layoutWeight(1)
    }
    .width('100%').margin({ top: 13 })
  }

  /** 隐私与免责说明 */
  @Builder
  privacy() {
    Text('音频仅在本机处理，不会上传或分享。结果用于嗓音状态初筛，不能代替医学诊断。')
      .fontSize(10).fontColor(this.pal.muted).lineHeight(18)
      .textAlign(TextAlign.Center).width('100%')
  }

  @Builder
  homeView() {
    Scroll() {
      Column() {
        // ---- 品牌条 ----
        Row() {
          BrandMark()
          Text('声析').fontSize(15).fontWeight(FontWeight.Bold)
            .fontColor(this.pal.text).margin({ left: 8 })
          Blank()
          Text('本地推理').fontSize(10).fontColor(this.pal.muted)
        }
        .width('100%').alignItems(VerticalAlign.Center)

        // ---- eyebrow 小标 ----
        Text('关照每一次发声').fontSize(11).fontWeight(FontWeight.Bold)
          .fontColor(this.pal.eyebrow).letterSpacing(1.98)
          .width('100%').margin({ top: 22 })

        // ---- 主标题 ----
        Text('今天的声音，\n感觉怎么样？')
          .fontSize(29).fontWeight(FontWeight.Bold).fontColor(this.pal.text)
          .lineHeight(45).letterSpacing(0.2)
          .width('100%').margin({ top: 8 })

        // ---- 副标题 ----
        Text('用一段自然朗读，了解你的嗓音状态。')
          .fontSize(13).fontColor(this.pal.sub).lineHeight(23)
          .width('100%').margin({ top: 4 })

        // ---- 声学插画 ----
        VoiceArt().width('100%').margin({ top: 16 })

        // ---- 主操作按钮 ----
        Row() {
          Text('🎙').fontSize(16).fontColor(TH.primaryFg)
          Text(this.ready ? '开始测评' : '正在准备分析模型…')
            .fontSize(15).fontWeight(FontWeight.Bold).fontColor(TH.primaryFg)
            .margin({ left: 8 }).layoutWeight(1).textAlign(TextAlign.Center)
          Text('→').fontSize(17).fontColor(TH.primaryFg)
        }
        .width('100%').height(54).padding({ left: 18, right: 18 })
        .backgroundColor(this.pal.primary).borderRadius(27)
        .opacity(this.ready ? 1.0 : 0.45)
        .margin({ top: 18 })
        .onClick(() => {
          if (this.ready) {
            this.startCapture();
          } else {
            this.status = '模型未就绪，无法开始录制。原因见下方红框：' + this.status;
          }
        })

        // 初始化失败时用醒目横幅显示原因，避免"点了没反应"
        if (!this.ready) {
          Column() {
            Text('⚠️ 模型未就绪').fontSize(14).fontWeight(FontWeight.Bold).fontColor(this.pal.rose)
            Text(this.status).fontSize(12).fontColor(this.pal.text).margin({ top: 6 })
            Text(this.diag).fontSize(10).fontColor(this.pal.muted).margin({ top: 6 })
          }
          .width('100%').padding(12).margin({ top: 12 })
          .backgroundColor(this.pal.errorBg).borderRadius(10)
          .border({ width: 2, color: this.pal.rose })
        } else if (this.status.length > 0) {
          Text(this.status).fontSize(11).fontColor(this.pal.sub)
            .textAlign(TextAlign.Center).width('100%').margin({ top: 12 })
        }

        Divider().color(this.pal.border).margin({ top: 26, bottom: 0 })

        Column() { this.howTo() }.width('100%').margin({ top: 22 })

        Blank().layoutWeight(1)

        // （按要求移除隐私说明）

        // 嗓音日历入口（新设计里没有这个页面，按同一套视觉新做）
        Row() {
          Text('📅').fontSize(13)
          Text('嗓音日历').fontSize(13).fontWeight(FontWeight.Bold)
            .fontColor(this.pal.text).layoutWeight(1).margin({ left: 8 })
          Text(this.history.length.toString() + ' 条').fontSize(11).fontColor(this.pal.muted)
          Text('›').fontSize(16).fontColor(this.pal.muted).margin({ left: 6 })
        }
        .width('100%').height(50).padding({ left: 16, right: 16 }).margin({ top: 18 })
        .backgroundColor(this.pal.card).borderRadius(14)
        .border({ width: 1, color: this.pal.border })
        .onClick(() => { this.goto('history', true); })
      }
      .width('100%').padding({ left: 18, right: 18, top: 40, bottom: 24 })
    }
    .width('100%').height('100%').backgroundColor(this.pal.bg).constraintSize({ maxWidth: 640 })
  }

  /** 顶栏：返回 + 标题 + 右侧徽标（新设计的 appbar 是浅色纸底墨字） */
  @Builder
  appbar(title: string) {
    Row() {
      Text('←').fontSize(20).fontColor(this.pal.text).width(34)
        .onClick(() => this.cancelCapture())
      Text(title).fontSize(15).fontWeight(FontWeight.Bold).fontColor(this.pal.text)
        .layoutWeight(1).textAlign(TextAlign.Center)
      Row() {
        Circle().width(6).height(6).fill(TH.badgeDot)
        Text('录音中').fontSize(11).fontColor(TH.badgeFg).margin({ left: 5 })
      }.width(34)
    }
    .width('100%').height(44).alignItems(VerticalAlign.Center)
  }

  @Builder
  recView() {
    Column() {
      this.appbar('嗓音测评')

      if (this.micFailed) {
        Text(this.status).fontSize(11).fontColor(this.pal.rose)
          .width('100%').padding(10).backgroundColor(this.pal.errorBg)
          .borderRadius(10).margin({ top: 8 })
      }

      Text('像平时一样说话').fontSize(20).fontWeight(FontWeight.Bold).fontColor(this.pal.text)
        .textAlign(TextAlign.Center).width('100%').margin({ top: 9 })
      Text('靠近手机，保持自然语速').fontSize(12).fontColor(this.pal.sub)
        .textAlign(TextAlign.Center).width('100%').margin({ top: 6 })

      // 倒计时环
      Stack() {
        TimerRing({ progress: this.progress })
          .width(150).height(150)
        Column() {
          Text(this.left.toString()).fontSize(47).fontWeight(FontWeight.Normal)
            .fontColor(this.pal.text).letterSpacing(-2)
          Text('秒后自动完成').fontSize(11).fontColor(this.pal.sub)
        }
      }.margin({ top: 23 })

      // 朗读卡
      Column() {
        Text('请自然朗读').fontSize(10).fontColor(this.pal.sub)
          .letterSpacing(1.0).width('100%').margin({ bottom: 11 })
        Text(READING_TEXT).fontSize(16).fontColor(this.pal.readingInk)
          .lineHeight(31).width('100%')
      }
      .width('100%').padding(20).backgroundColor(this.pal.readingBg)
      .borderRadius(20).margin({ top: 18 })

      // 实时声波
      LiveWave({ levels: this.waveLevels() })
        .width('100%').margin({ top: 10 })


      Blank()

      Row() {
        Text('完成朗读，查看结果').fontSize(15).fontWeight(FontWeight.Bold)
          .fontColor(TH.primaryFg).layoutWeight(1).textAlign(TextAlign.Center)
      }
      .width('100%').height(54).backgroundColor(this.pal.primary).borderRadius(27)
      .onClick(() => this.finishCapture())

      Text('取消本次测评').fontSize(13).fontColor(TH.secondaryButtonFg)
        .textAlign(TextAlign.Center).width('100%').height(46)
        .backgroundColor(this.pal.secondaryButtonBg).borderRadius(23)
        .margin({ top: 10 })
        .onClick(() => this.cancelCapture())
    }
    .width('100%').height('100%').padding({ left: 22, right: 22, top: 14, bottom: 18 })
    .backgroundColor(this.pal.bg).constraintSize({ maxWidth: 640 })
  }

  /** 把 24 段柱高映射成 LiveWave 需要的 11 列 0~1 电平 */
  private waveLevels(): number[] {
    const out: number[] = [];
    const n = this.bars.length;
    for (let i = 0; i < 11; i++) {
      const a = Math.floor(i * n / 11);
      const b = Math.max(a + 1, Math.floor((i + 1) * n / 11));
      let mx = 0;
      for (let j = a; j < Math.min(b, n); j++) {
        if (this.bars[j] > mx) {
          mx = this.bars[j];
        }
      }
      out.push(Math.max(0, Math.min(1, (mx - 4) / 96)));
    }
    return out;
  }

  @Builder
  procView() {
    Column() {
      this.appbar('嗓音测评')

      Blank().layoutWeight(1)

      VoiceArt().width('100%').margin({ top: 33 })
      Text('听见你的声音').fontSize(23).fontWeight(FontWeight.Bold).fontColor(this.pal.text)
        .textAlign(TextAlign.Center).width('100%').margin({ top: 12 })
      Text('正在本机分析，请稍候').fontSize(12).fontColor(this.pal.sub).lineHeight(22)
        .textAlign(TextAlign.Center).width('100%')

      Column() {
        this.stepRowState(true, '声音采集完成')
        this.stepRowState(false, '正在分析嗓音特征')
      }.width('100%').alignItems(HorizontalAlign.Center).margin({ top: 32 })

      Blank().layoutWeight(1)

      // （按要求移除隐私说明）
    }
    .width('100%').height('100%').padding({ left: 22, right: 22, top: 14, bottom: 18 })
    .backgroundColor(this.pal.bg).constraintSize({ maxWidth: 640 })
  }

  /** 分析进度步骤行：完成打勾 / 进行中转圈 */
  @Builder
  stepRowState(done: boolean, label: string) {
    Row() {
      if (done) {
        Text('✓').fontSize(14).fontColor(this.pal.stepAccent).width(14).textAlign(TextAlign.Center)
      } else {
        LoadingProgress().width(14).height(14).color(this.pal.stepAccent)
      }
      Text(label).fontSize(12).fontColor(this.pal.stepLabel).margin({ left: 9 })
    }
    .alignItems(VerticalAlign.Center).margin({ top: 7 })
  }

  /** 嗓音日历：按月回看本地档案，点某天看当天记录，点记录看完整报告 */
  @Builder
  historyView() {
    Column() {
      Row() {
        Text('← 返回').fontSize(14).fontColor(this.pal.navText).layoutWeight(1)
          .onClick(() => { this.goto('home', false); })
        Text('嗓音日历').fontSize(16).fontWeight(FontWeight.Bold).fontColor(this.pal.navText)
        Blank().layoutWeight(1)
      }
      .width('100%').padding({ left: 14, right: 14, top: 26, bottom: 12 })
      .backgroundColor(this.pal.navBg)

      Scroll() {
        Column() {
          Row() {
            Text('‹').fontSize(24).fontColor(this.pal.primary).padding({ left: 8, right: 16 })
              .onClick(() => this.shiftMonth(-1))
            Text(this.calYear.toString() + ' 年 ' + this.calMonth.toString() + ' 月')
              .fontSize(16).fontWeight(FontWeight.Bold).fontColor(this.pal.text)
              .layoutWeight(1).textAlign(TextAlign.Center)
            Text('›').fontSize(24).fontColor(this.pal.primary).padding({ left: 16, right: 8 })
              .onClick(() => this.shiftMonth(1))
          }
          .width('100%').margin({ bottom: 8 })

          Row() {
            ForEach(['日', '一', '二', '三', '四', '五', '六'], (w: string) => {
              Text(w).fontSize(11).fontColor(this.pal.muted).textAlign(TextAlign.Center).layoutWeight(1)
            }, (w: string, i: number) => w + i.toString())
          }.width('100%').margin({ bottom: 4 })

          Flex({ wrap: FlexWrap.Wrap }) {
            ForEach(this.calCells(), (d: number, idx: number) => {
              Column() {
                if (d > 0) {
                  Text(d.toString()).fontSize(13)
                    .fontColor(d === this.selDay ? Color.White : this.pal.text)
                    .backgroundColor(d === this.selDay ? this.pal.primary : Color.Transparent)
                    .borderRadius(14).width(28).height(20).textAlign(TextAlign.Center)
                  if (this.scoreOfDay(d) >= 0) {
                    Text(this.scoreOfDay(d).toString()).fontSize(10).fontWeight(FontWeight.Bold)
                      .fontColor(Color.White).backgroundColor(this.toneColor(this.scoreOfDay(d)))
                      .borderRadius(8).padding({ left: 5, right: 5 }).margin({ top: 2 })
                  } else {
                    Text(' ').fontSize(10).margin({ top: 2 })
                  }
                }
              }
              .width('14.28%').height(46)
              .justifyContent(FlexAlign.Center)
              .onClick(() => {
                if (d > 0) {
                  this.selDay = d;
                }
              })
            }, (d: number, i: number) => i.toString())
          }.width('100%')

          Divider().color(this.pal.border).margin({ top: 8, bottom: 8 })

          Text(this.calMonth.toString() + ' 月 ' + this.selDay.toString() + ' 日 · '
            + this.recsOfSelDay().length.toString() + ' 次评测')
            .fontSize(13).fontWeight(FontWeight.Bold).fontColor(this.pal.text)
            .width('100%').margin({ bottom: 6 })

          if (this.recsOfSelDay().length === 0) {
            Text('这一天还没有评测记录').fontSize(12).fontColor(this.pal.muted)
              .width('100%').padding({ top: 10, bottom: 10 })
          }

          ForEach(this.recsOfSelDay(), (r: Rec) => {
            Row() {
              Column() {
                Text(r.score.toString() + ' 分').fontSize(18).fontWeight(FontWeight.Bold)
                  .fontColor(this.toneColor(r.score))
                Text(hhmm(r.ts)).fontSize(11).fontColor(this.pal.muted)
              }.width(70)

              Column() {
                Text(r.risk).fontSize(13).fontWeight(FontWeight.Bold).fontColor(this.pal.text)
                Text('不健康概率 ' + Math.round(r.prob * 100).toString() + '% · 语音 '
                  + Math.round(r.speech * 100).toString() + '%')
                  .fontSize(11).fontColor(this.pal.sub).margin({ top: 2 })
                Text(r.summary).fontSize(11).fontColor(this.pal.muted)
                  .maxLines(2).textOverflow({ overflow: TextOverflow.Ellipsis }).margin({ top: 2 })
              }.layoutWeight(1).alignItems(HorizontalAlign.Start)

              Text('›').fontSize(20).fontColor(this.pal.muted)
            }
            .width('100%').padding(10).margin({ bottom: 8 })
            .backgroundColor(this.pal.card).borderRadius(10)
            .border({ width: 1, color: this.pal.border })
            .onClick(() => this.openRec(r))
          }, (r: Rec, i: number) => r.id.toString() + '_' + i.toString())

          Text(' ').fontSize(10).margin({ top: 10, bottom: 30 })
        }
        .width('100%').padding({ left: 12, right: 12, top: 12 })
      }
      .layoutWeight(1).width('100%')
    }
    .width('100%').height('100%').backgroundColor(this.pal.bg).constraintSize({ maxWidth: 640 })
  }

  /** 失败时把「哪一步 + 什么错」摆出来，而不是静默回首页 */
  @Builder
  errorView() {
    Column() {
      Text('⚠️').fontSize(40)
      Text('分析失败').fontSize(17).fontWeight(FontWeight.Bold).fontColor(this.pal.rose).margin({ top: 10 })
      Text('失败步骤：' + this.stage).fontSize(13).fontColor(this.pal.sub)
        .margin({ top: 12 }).width('100%').textAlign(TextAlign.Center)
      Text(this.errorMsg).fontSize(12).fontColor(this.pal.text)
        .margin({ top: 8 }).width('100%').padding(12)
        .backgroundColor(this.pal.card).borderRadius(10)
        .border({ width: 1, color: this.pal.border })
      Text('中间量：' + this.diag).fontSize(10).fontColor(this.pal.muted)
        .margin({ top: 10 }).width('100%')
      Text('返回').fontSize(15).fontWeight(FontWeight.Bold).fontColor(Color.White)
        .textAlign(TextAlign.Center).width('70%').height(42).margin({ top: 20 })
        .linearGradient({ angle: 135, colors: [[this.pal.primary, 0.0], [this.pal.primaryDark, 1.0]] })
        .borderRadius(21).onClick(() => { this.goto('home', false); })
    }
    .width('100%').height('100%').padding(20)
    .justifyContent(FlexAlign.Center).backgroundColor(this.pal.bg).constraintSize({ maxWidth: 640 })
  }

  @Builder
  reportView() {
    Column() {
      Row() {
        Text('← 返回').fontSize(14).fontColor(this.pal.navText).layoutWeight(1)
          .onClick(() => { this.goto('home', false); })
        Text('声学分析报告').fontSize(16).fontWeight(FontWeight.Bold).fontColor(this.pal.navText)
        Blank().layoutWeight(1)
      }
      .width('100%').padding({ left: 14, right: 14, top: 26, bottom: 12 })
      .backgroundColor(this.pal.navBg)

      Scroll() {
        Column() {
          // 1) 嗓音健康判定
          Column() {
            this.cardHead('🟢', '嗓音健康判定', this.healthScore >= 70 ? '健康' : '需关注')
            Row() {
              Text(this.healthScore.toString()).fontSize(54).fontWeight(FontWeight.Bold)
                .fontColor(this.healthScore >= 70 ? this.pal.teal : (this.healthScore >= 40 ? this.pal.amber : this.pal.rose))
              Text(' /100').fontSize(15).fontColor(this.pal.muted).margin({ top: 22 })
            }
            Text(this.healthScore >= 70 ? '嗓音状态良好，继续保持' :
              (this.healthScore >= 40 ? '存在一定疲劳或代偿，建议养护' : '嗓音风险偏高，建议尽快复查'))
              .fontSize(13).fontColor(this.pal.sub)
          }
          .width('100%').padding(14).backgroundColor(this.pal.card).borderRadius(12)
          .border({ width: 1, color: this.pal.border }).margin({ bottom: 12 })

          // 2) 模型叙述
          Column() {
            this.cardHead('🗣️', '模型叙述', this.narration.risk)
            Text(this.narration.summary).fontSize(13).fontColor(this.pal.text).width('100%')
            Text('养护建议：').fontSize(13).fontWeight(FontWeight.Bold).fontColor(this.pal.text)
              .margin({ top: 10 })
            ForEach(this.narration.suggestions, (s: string) => {
              Text('• ' + s).fontSize(12).fontColor(this.pal.sub).width('100%').margin({ top: 4 })
            }, (s: string, i: number) => i.toString())
            Text('温馨提示（仅供参考）：').fontSize(13).fontWeight(FontWeight.Bold).fontColor(this.pal.text)
              .margin({ top: 10 })
            ForEach(this.narration.warnings, (w: string) => {
              Text('⚠️ ' + w).fontSize(12).fontColor(this.pal.amber).width('100%').margin({ top: 4 })
            }, (w: string, i: number) => i.toString())
          }
          .width('100%').padding(14).backgroundColor(this.pal.card).borderRadius(12)
          .border({ width: 1, color: this.pal.border }).margin({ bottom: 12 })

          // 3) 判定概率明细
          Column() {
            this.cardHead('📈', '判定概率明细', '健康分依据')
            this.bar('健康概率', this.healthyPct.toString() + '%', this.healthyPct, this.pal.teal)
            this.bar('风险概率', this.riskPct.toString() + '%', this.riskPct,
              this.riskPct > 40 ? this.pal.rose : (this.riskPct > 20 ? this.pal.amber : this.pal.teal))
            this.bar('语音活跃', this.speechPct.toString() + '%', this.speechPct, this.pal.blue)
          }
          .width('100%').padding(14).backgroundColor(this.pal.card).borderRadius(12)
          .border({ width: 1, color: this.pal.border }).margin({ bottom: 12 })

          // 4) 声学质量雷达
          Column() {
            this.cardHead('📊', '核心声学质量雷达', '真实推理输出')
            RadarChart({ values: this.radar, accent: this.pal.teal, ink: this.pal.sub })
              .width('100%').height(360)
          }
          .width('100%').padding(14).backgroundColor(this.pal.card).borderRadius(12)
          .border({ width: 1, color: this.pal.border }).margin({ bottom: 12 })

          Text('用时 ' + this.elapsedMs.toString() + ' ms · 本机推理')
            .fontSize(10).fontColor(this.pal.muted).margin({ bottom: 4 })
          Text('诊断：' + this.diag).fontSize(9).fontColor(this.pal.muted).margin({ bottom: 20 })
        }
        .width('100%').padding({ left: 12, right: 12, top: 12 })
      }
      .layoutWeight(1).width('100%')
    }
    .width('100%').height('100%').backgroundColor(this.pal.bg).constraintSize({ maxWidth: 640 })
  }

  build() {
    Stack() {
      if (this.screen === 'home') {
        this.homeView()
      } else if (this.screen === 'rec') {
        this.recView()
      } else if (this.screen === 'proc') {
        this.procView()
      } else if (this.screen === 'error') {
        this.errorView()
      } else if (this.screen === 'history') {
        this.historyView()
      } else {
        this.reportView()
      }
    }
    .width('100%').height('100%')
    .translate({ x: this.offsetX })
    .clip(true)
  }
}

/** 五维声学质量雷达图 */
@Component
struct RadarChart {
  @Prop @Watch('redraw') values: number[];
  @Prop accent: string;
  @Prop ink: string;

  private settings: RenderingContextSettings = new RenderingContextSettings(true);
  private ctx: CanvasRenderingContext2D = new CanvasRenderingContext2D(this.settings);
  private labels: string[] = ['抗粗糙度', '谐波丰富度', '抗紧张度', '气息支撑', '清晰度'];
  private ready: boolean = false;

  redraw(): void {
    if (this.ready) {
      this.draw();
    }
  }

  private draw(): void {
    const w = this.ctx.width;
    const h = this.ctx.height;
    const cx = w / 2;
    const cy = h / 2 + 4;
    const r = Math.min(w / 2 - 78, h / 2 - 40);   // 左右各留出标签的位置，否则两侧文字被裁
    const n = 5;
    this.ctx.clearRect(0, 0, w, h);
    this.ctx.lineWidth = 1;

    for (let ring = 1; ring <= 4; ring++) {
      this.ctx.strokeStyle = ring === 3 ? '#55254F7F' : '#1F86909C';
      this.ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const a = -Math.PI / 2 + 2 * Math.PI * i / n;
        const x = cx + (r * ring / 4) * Math.cos(a);
        const y = cy + (r * ring / 4) * Math.sin(a);
        if (i === 0) {
          this.ctx.moveTo(x, y);
        } else {
          this.ctx.lineTo(x, y);
        }
      }
      this.ctx.closePath();
      this.ctx.stroke();
    }
    this.ctx.strokeStyle = '#3386909C';
    for (let i = 0; i < n; i++) {
      const a = -Math.PI / 2 + 2 * Math.PI * i / n;
      this.ctx.beginPath();
      this.ctx.moveTo(cx, cy);
      this.ctx.lineTo(cx + r * Math.cos(a), cy + r * Math.sin(a));
      this.ctx.stroke();
    }

    this.ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const a = -Math.PI / 2 + 2 * Math.PI * i / n;
      const v = Math.max(0, Math.min(1, (i < this.values.length ? this.values[i] : 0) / 4));
      const x = cx + r * v * Math.cos(a);
      const y = cy + r * v * Math.sin(a);
      if (i === 0) {
        this.ctx.moveTo(x, y);
      } else {
        this.ctx.lineTo(x, y);
      }
    }
    this.ctx.closePath();
    this.ctx.fillStyle = '#5576C5AD';
    this.ctx.fill();
    this.ctx.strokeStyle = this.accent;
    this.ctx.lineWidth = 2;
    this.ctx.stroke();

    this.ctx.fillStyle = this.ink;
    // 38px 时左右两侧标签放不下，收到 28px 并贴近图形
    this.ctx.font = '28px sans-serif';
    for (let i = 0; i < n; i++) {
      const a = -Math.PI / 2 + 2 * Math.PI * i / n;
      const x = cx + (r + 30) * Math.cos(a);
      const y = cy + (r + 30) * Math.sin(a);
      this.ctx.textAlign = Math.abs(x - cx) < 10 ? 'center' : (x < cx ? 'right' : 'left');
      this.ctx.fillText(this.labels[i], x, y + 10);
    }
  }

  build() {
    Canvas(this.ctx)
      .width('100%').height('100%')
      .onReady(() => {
        this.ready = true;
        this.draw();
      })
  }
}
