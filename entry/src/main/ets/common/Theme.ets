/**
 * 声析 Auralis · 设计规范
 *
 * 全部取自安卓版 UI（AuralisUi.kt）反编译出的 42 个色 token，
 * 保证鸿蒙版与安卓版视觉一致。命名沿用原 token 名，便于对照维护。
 */

export class AuralisTheme {
  // ---- 基础色 ----
  /** 正文墨色 */
  readonly ink: string = '#19383D';
  /** 次要文字 */
  readonly secondary: string = '#5B7062';
  /** 页面底色（纸感） */
  readonly paper: string = '#F8FAF8';
  /** 分割线 */
  readonly line: string = '#E1E9E3';
  /** 品牌主色 */
  readonly brandGreen: string = '#477660';
  /** 品牌标深色 */
  readonly barDark: string = '#426850';
  /** 品牌标浅色 */
  readonly barLight: string = '#83AC8B';
  /** 品牌副标 */
  readonly brandCaption: string = '#86928B';
  /** eyebrow 小标 */
  readonly eyebrowColor: string = '#82928A';

  // ---- 控件 ----
  readonly iconButtonFg: string = '#5D7367';
  readonly iconButtonBg: string = '#EAF0EB';
  readonly primaryFg: string = '#F6FAF7';
  readonly secondaryButtonBg: string = '#E6EEE7';
  readonly secondaryButtonFg: string = '#3D6150';
  readonly textActionFg: string = '#748479';

  // ---- 评分 ----
  readonly scoreGreen: string = '#244F3B';
  readonly scoreUnit: string = '#7B8E7E';
  readonly scorePanelBg: string = '#E3EEE3';
  /** 中等分（60~69）面板底色 */
  readonly scorePanelWarn: string = '#F7F0DF';
  /** 低分面板底色 */
  readonly scorePanelBad: string = '#F8E7E1';
  readonly scoreTextWarn: string = '#83632A';
  readonly scoreTextBad: string = '#934627';
  readonly metricValue: string = '#4C6957';

  // ---- 卡片 / 朗读 ----
  readonly readingBg: string = '#EDF2EA';
  readonly readingInk: string = '#3E5947';
  readonly adviceIconBg: string = '#EEF3EC';
  readonly adviceIconFg: string = '#69896C';
  readonly adviceText: string = '#6E8073';
  readonly trackBg: string = '#E0E8DF';
  readonly trackFill: string = '#8CAA90';
  readonly metaDivider: string = '#D9E3DC';

  // ---- 环 / 声波 / 光晕 ----
  readonly ringTrack: string = '#E3EBE3';
  readonly ringFill: string = '#83A58E';
  readonly liveBar: string = '#73987C';
  readonly haloBorder: string = '#E0E9DF';
  readonly haloRingOuter: string = '#D5E4D9';
  readonly haloRingInner: string = '#C7DECC';
  readonly artDotColor: string = '#ACCAB3';

  // ---- 徽标 / 步骤 ----
  readonly badgeFg: string = '#5D7C67';
  readonly badgeDot: string = '#83A28C';
  readonly stepAccent: string = '#648571';
  readonly stepLabel: string = '#859586';

  // ---- 提示 / 错误 ----
  readonly errorFg: string = '#9B4F35';
  readonly errorBg: string = '#F4E8DF';
  readonly sheetText: string = '#718174';
  readonly sheetLabelColor: string = '#8D9C91';
  readonly howToBorder: string = '#DCE5DF';
  readonly howToNumber: string = '#7C9284';
  /** 弹层遮罩 rgba(25,53,43,0.27) */
  readonly scrim: string = '#4525352B';
}

/** 全局唯一实例 */
export const TH = new AuralisTheme();

/** 朗读文本（与安卓版逐字一致） */
export const READING_TEXT: string =
  '各位听众朋友，大家好。声音是我们最重要的表达工具，规律的呼吸与适度的用嗓，' +
  '能让嗓音始终保持良好的状态。请您保持平时的自然语速，平稳地读完这段话。';

/** 根据分数取面板底色与分数色 */
export function scoreTone(score: number): string[] {
  if (score < 40) {
    return [TH.scorePanelBad, TH.scoreTextBad];
  }
  if (score < 70) {
    return [TH.scorePanelWarn, TH.scoreTextWarn];
  }
  return [TH.scorePanelBg, TH.scoreGreen];
}
