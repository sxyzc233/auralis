/**
 * 声析 Auralis · 自绘控件（ArkUI Canvas 版）
 *
 * 逐一对齐安卓版 AuralisUi.kt 里的同名 View：
 *   BrandMark  4 根小竖条（品牌标）
 *   VoiceArt   声学插画：径向光晕 + 3 圈同心环 + 11 根声波条 + 2 个圆点
 *   TimerRing  倒计时环：底环 + 扇形进度 + 中心遮罩
 *   LiveWave   实时声波：11 列，按音频电平起伏
 *   TrackView  水平进度条
 *
 * 注意：ArkUI Canvas 的绘制坐标单位是 vp，而 font 的 px 是物理像素，
 * 所以字号统一用 vp2px() 换算，避免高分屏上字变得极小。
 */
import { TH } from './Theme';

const DASH: number[] = [0];  // 占位，避免空数组类型推断问题

/** 品牌标：4 根圆角竖条 */
@Component
export struct BrandMark {
  private st: RenderingContextSettings = new RenderingContextSettings(true);
  private ctx: CanvasRenderingContext2D = new CanvasRenderingContext2D(this.st);

  build() {
    Canvas(this.ctx)
      .width(30)
      .height(24)
      .onReady(() => this.draw())
  }

  private draw(): void {
    const h = this.ctx.height;
    const w = this.ctx.width;
    this.ctx.clearRect(0, 0, w, h);
    const barW = 3.2;
    const gap = 2.4;
    const heights: number[] = [14.0, 25.0, 20.0, 10.0];
    this.ctx.fillStyle = TH.brandGreen;
    let x = 0;
    for (let i = 0; i < heights.length; i++) {
      const bh = heights[i] * 0.8;
      const y = (h - bh) / 2;
      this.ctx.beginPath();
      this.ctx.moveTo(x + barW / 2, y);
      this.ctx.arcTo(x + barW, y, x + barW, y + bh, barW / 2);
      this.ctx.arcTo(x + barW, y + bh, x, y + bh, barW / 2);
      this.ctx.arcTo(x, y + bh, x, y, barW / 2);
      this.ctx.arcTo(x, y, x + barW, y, barW / 2);
      this.ctx.closePath();
      this.ctx.fill();
      x += barW + gap;
    }
  }
}

/** 声学插画 */
@Component
export struct VoiceArt {
  private st: RenderingContextSettings = new RenderingContextSettings(true);
  private ctx: CanvasRenderingContext2D = new CanvasRenderingContext2D(this.st);

  build() {
    Canvas(this.ctx)
      .width('100%')
      .height(188)
      .onReady(() => this.draw())
  }

  private draw(): void {
    const w = this.ctx.width;
    const h = this.ctx.height;
    this.ctx.clearRect(0, 0, w, h);
    const cx = w / 2;
    const cy = h / 2;
    const R = 90.5;

    // 径向光晕
    const g = this.ctx.createRadialGradient(cx, cy, 0, cx, cy, R);
    g.addColorStop(0.0, '#E2EEE5');
    g.addColorStop(0.537, '#E8F1E8');
    g.addColorStop(0.905, '#F1F5ED');
    g.addColorStop(0.919, 'rgba(241,245,237,0)');
    this.ctx.fillStyle = g;
    this.ctx.beginPath();
    this.ctx.arc(cx, cy, R, 0, Math.PI * 2);
    this.ctx.fill();

    // 三圈同心环
    this.ctx.lineWidth = 1;
    const rings: string[] = [TH.haloBorder, TH.haloRingOuter, TH.haloRingInner];
    const radii: number[] = [R - 0.5, R - 19, R - 43];
    for (let i = 0; i < rings.length; i++) {
      this.ctx.strokeStyle = rings[i];
      this.ctx.beginPath();
      this.ctx.arc(cx, cy, radii[i], 0, Math.PI * 2);
      this.ctx.stroke();
    }

    // 11 根声波条
    const bars: number[] = [13, 24, 38, 52, 66, 44, 29, 54, 40, 22, 13];
    const barW = 5;
    const gap = 5;
    const totalW = bars.length * barW + (bars.length - 1) * gap;
    let x = cx - totalW / 2;
    for (let i = 0; i < bars.length; i++) {
      const bh = Math.max(2, bars[i]);
      this.ctx.fillStyle = (i % 2 === 1) ? TH.barLight : TH.barDark;
      const y = cy - bh / 2;
      const r = Math.min(7, barW / 2);
      this.roundRect(x, y, barW, bh, r);
      x += barW + gap;
    }

    // 两个装饰圆点
    this.ctx.fillStyle = TH.artDotColor;
    this.ctx.beginPath();
    this.ctx.arc(58.5, 43.5, 3.5, 0, Math.PI * 2);
    this.ctx.fill();
    this.ctx.beginPath();
    this.ctx.arc(w - 55, 137, 2.0, 0, Math.PI * 2);
    this.ctx.fill();
  }

  private roundRect(x: number, y: number, w: number, h: number, r: number): void {
    this.ctx.beginPath();
    this.ctx.moveTo(x + r, y);
    this.ctx.arcTo(x + w, y, x + w, y + h, r);
    this.ctx.arcTo(x + w, y + h, x, y + h, r);
    this.ctx.arcTo(x, y + h, x, y, r);
    this.ctx.arcTo(x, y, x + w, y, r);
    this.ctx.closePath();
    this.ctx.fill();
  }
}

/** 倒计时环 */
@Component
export struct TimerRing {
  @Prop @Watch('redraw') progress: number = 0;
  private st: RenderingContextSettings = new RenderingContextSettings(true);
  private ctx: CanvasRenderingContext2D = new CanvasRenderingContext2D(this.st);

  build() {
    Canvas(this.ctx)
      .width('100%')
      .height('100%')
      .onReady(() => this.draw())
  }

  redraw(): void {
    this.draw();
  }

  private draw(): void {
    const w = this.ctx.width;
    const h = this.ctx.height;
    this.ctx.clearRect(0, 0, w, h);
    const cx = w / 2;
    const cy = h / 2;
    const r = Math.min(cx, cy);

    // 底环
    this.ctx.fillStyle = TH.ringTrack;
    this.ctx.beginPath();
    this.ctx.arc(cx, cy, r, 0, Math.PI * 2);
    this.ctx.fill();

    // 扇形进度（从 -90° 顺时针，填充到圆心，与安卓版一致）
    const p = Math.max(0, Math.min(1, this.progress));
    if (p > 0) {
      this.ctx.fillStyle = TH.ringFill;
      this.ctx.beginPath();
      this.ctx.moveTo(cx, cy);
      this.ctx.arc(cx, cy, r, -Math.PI / 2, -Math.PI / 2 + p * Math.PI * 2);
      this.ctx.closePath();
      this.ctx.fill();
    }

    // 中心遮罩
    const inner = Math.max(0, r - 4);
    this.ctx.fillStyle = TH.paper;
    this.ctx.beginPath();
    this.ctx.arc(cx, cy, inner, 0, Math.PI * 2);
    this.ctx.fill();
  }
}

/** 实时声波 */
@Component
export struct LiveWave {
  /** 11 列电平，取值 0~1 */
  @Prop @Watch('redraw') levels: number[] = [];
  private st: RenderingContextSettings = new RenderingContextSettings(true);
  private ctx: CanvasRenderingContext2D = new CanvasRenderingContext2D(this.st);
  private columns: number = 11;

  build() {
    Canvas(this.ctx)
      .width('100%')
      .height(36)
      .onReady(() => this.draw())
  }

  redraw(): void {
    this.draw();
  }

  private draw(): void {
    const w = this.ctx.width;
    const h = this.ctx.height;
    this.ctx.clearRect(0, 0, w, h);
    const barW = 3;
    const gap = 5;
    const totalW = this.columns * barW + (this.columns - 1) * gap;
    let x = (w - totalW) / 2;
    const cy = h / 2;
    for (let i = 0; i < this.columns; i++) {
      const lv = (i < this.levels.length) ? Math.max(0, Math.min(1, this.levels[i])) : 0;
      const bh = lv * 33 + 3;
      this.ctx.fillStyle = (i % 2 === 1) ? TH.barLight : TH.liveBar;
      const y = cy - bh / 2;
      const r = Math.min(7, barW / 2);
      this.ctx.beginPath();
      this.ctx.moveTo(x + r, y);
      this.ctx.arcTo(x + barW, y, x + barW, y + bh, r);
      this.ctx.arcTo(x + barW, y + bh, x, y + bh, r);
      this.ctx.arcTo(x, y + bh, x, y, r);
      this.ctx.arcTo(x, y, x + barW, y, r);
      this.ctx.closePath();
      this.ctx.fill();
      x += barW + gap;
    }
  }
}

/** 水平进度条 */
@Component
export struct TrackView {
  @Prop @Watch('redraw') percent: number = 0;
  private st: RenderingContextSettings = new RenderingContextSettings(true);
  private ctx: CanvasRenderingContext2D = new CanvasRenderingContext2D(this.st);

  build() {
    Canvas(this.ctx)
      .width('100%')
      .height(6)
      .onReady(() => this.draw())
  }

  redraw(): void {
    this.draw();
  }

  private draw(): void {
    const w = this.ctx.width;
    const h = this.ctx.height;
    this.ctx.clearRect(0, 0, w, h);
    const r = h / 2;
    this.ctx.fillStyle = TH.trackBg;
    this.ctx.beginPath();
    this.ctx.moveTo(r, 0);
    this.ctx.arcTo(w, 0, w, h, r);
    this.ctx.arcTo(w, h, 0, h, r);
    this.ctx.arcTo(0, h, 0, 0, r);
    this.ctx.arcTo(0, 0, w, 0, r);
    this.ctx.closePath();
    this.ctx.fill();

    const p = Math.max(0, Math.min(100, this.percent));
    const fw = w * p / 100;
    if (fw > 0.5) {
      this.ctx.fillStyle = TH.trackFill;
      this.ctx.beginPath();
      this.ctx.moveTo(r, 0);
      this.ctx.arcTo(fw, 0, fw, h, r);
      this.ctx.arcTo(fw, h, 0, h, r);
      this.ctx.arcTo(0, h, 0, 0, r);
      this.ctx.arcTo(0, 0, fw, 0, r);
      this.ctx.closePath();
      this.ctx.fill();
    }
  }
}
