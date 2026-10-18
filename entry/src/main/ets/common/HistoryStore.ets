/**
 * 本地嗓音档案：每次评测的结果存成一个 JSON 文件（应用沙箱内，不上云）。
 * 用于「日历记录」页按月回看。
 */
import { fileIo as fs } from '@kit.CoreFileKit';
import { util } from '@kit.ArkTS';

export interface Rec {
  id: number;
  ts: number;            // 毫秒时间戳
  score: number;         // 0-100 嗓音健康分
  prob: number;          // 不健康概率
  label: number;         // 0 健康 / 1 需关注
  speech: number;        // 语音活跃
  radar: number[];       // 五维雷达
  risk: string;
  summary: string;
  suggestions: string[];
  warnings: string[];
  families: number[];    // 8 大家族 1-10
}

export class HistoryStore {
  private path: string;

  constructor(filesDir: string) {
    this.path = filesDir + '/voice_history.json';
  }

  load(): Rec[] {
    try {
      const st = fs.statSync(this.path);
      if (st.size <= 0) {
        return [];
      }
      const buf = new ArrayBuffer(st.size);
      const f = fs.openSync(this.path, fs.OpenMode.READ_ONLY);
      fs.readSync(f.fd, buf);
      fs.closeSync(f);
      const decoder = util.TextDecoder.create('utf-8');
      const text = decoder.decodeToString(new Uint8Array(buf));
      const arr = JSON.parse(text) as Rec[];
      return arr;
    } catch (e) {
      return [];
    }
  }

  private write(list: Rec[]): void {
    try {
      const f = fs.openSync(this.path, fs.OpenMode.READ_WRITE | fs.OpenMode.CREATE | fs.OpenMode.TRUNC);
      fs.writeSync(f.fd, JSON.stringify(list));
      fs.closeSync(f);
    } catch (e) {
      // 存储失败不影响本次评测展示
    }
  }

  add(rec: Rec): Rec[] {
    const list = this.load();
    list.push(rec);
    this.write(list);
    return list;
  }

  remove(id: number): Rec[] {
    const list: Rec[] = [];
    const all = this.load();
    for (let i = 0; i < all.length; i++) {
      if (all[i].id !== id) {
        list.push(all[i]);
      }
    }
    this.write(list);
    return list;
  }

  clear(): Rec[] {
    this.write([]);
    return [];
  }
}

/** 某天的记录（本地时区） */
export function recordsOfDay(list: Rec[], year: number, month: number, day: number): Rec[] {
  const out: Rec[] = [];
  for (let i = 0; i < list.length; i++) {
    const d = new Date(list[i].ts);
    if (d.getFullYear() === year && d.getMonth() + 1 === month && d.getDate() === day) {
      out.push(list[i]);
    }
  }
  return out;
}

export function dayKey(ts: number): string {
  const d = new Date(ts);
  return d.getFullYear() + '-' + (d.getMonth() + 1) + '-' + d.getDate();
}

export function hhmm(ts: number): string {
  const d = new Date(ts);
  const h = d.getHours();
  const m = d.getMinutes();
  return (h < 10 ? '0' : '') + h + ':' + (m < 10 ? '0' : '') + m;
}

/** 当月天数与 1 号是周几（0=周日） */
export function monthInfo(year: number, month: number): number[] {
  const first = new Date(year, month - 1, 1);
  const days = new Date(year, month, 0).getDate();
  return [first.getDay(), days];
}
