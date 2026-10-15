import type { Series, SeriesByRun } from './api'

export function mergeSeries(prev: SeriesByRun, delta: SeriesByRun): SeriesByRun {
  const out: SeriesByRun = { ...prev }
  for (const [rid, keys] of Object.entries(delta)) {
    const cur = { ...(out[rid] || {}) }
    for (const [k, s] of Object.entries(keys)) {
      const p = cur[k]
      cur[k] = p ? { steps: p.steps.concat(s.steps), values: p.values.concat(s.values) } : s
    }
    out[rid] = cur
  }
  return out
}

export type Agg = { steps: number[]; mean: number[]; std: number[]; n: number }

function interp(x: number, xs: number[], ys: number[]): number {
  if (x <= xs[0]) return ys[0]
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1]
  let lo = 0, hi = xs.length - 1
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (xs[mid] <= x) lo = mid
    else hi = mid
  }
  const t = (x - xs[lo]) / (xs[hi] - xs[lo] || 1)
  return ys[lo] + t * (ys[hi] - ys[lo])
}

function sortedUnique(s: Series): Series {
  const idx = s.steps.map((_, i) => i).sort((a, b) => s.steps[a] - s.steps[b])
  const steps: number[] = [], values: number[] = [], counts: number[] = []
  for (const i of idx) {
    const st = s.steps[i]
    if (steps.length && steps[steps.length - 1] === st) {
      values[values.length - 1] += s.values[i]
      counts[counts.length - 1]++
    } else {
      steps.push(st); values.push(s.values[i]); counts.push(1)
    }
  }
  return { steps, values: values.map((v, i) => v / counts[i]) }
}

/** Mean ± std across runs. Exact step alignment when possible, otherwise interpolation on a shared grid. */
export function aggregate(list: Series[], maxPoints = 300): Agg | null {
  const ss = list.filter((s) => s.steps.length).map(sortedUnique)
  if (!ss.length) return null
  let grid: number[]
  let mat: number[][]
  const sameSteps = ss.every((s) => s.steps.length === ss[0].steps.length && s.steps.every((v, i) => v === ss[0].steps[i]))
  if (sameSteps && ss[0].steps.length <= maxPoints) {
    grid = ss[0].steps
    mat = ss.map((s) => s.values)
  } else {
    let lo = Math.max(...ss.map((s) => s.steps[0]))
    let hi = Math.min(...ss.map((s) => s.steps[s.steps.length - 1]))
    if (hi < lo) {
      lo = Math.min(...ss.map((s) => s.steps[0]))
      hi = Math.max(...ss.map((s) => s.steps[s.steps.length - 1]))
    }
    const n = Math.max(2, Math.min(maxPoints, Math.max(...ss.map((s) => s.steps.length))))
    grid = Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1))
    mat = ss.map((s) => grid.map((x) => interp(x, s.steps, s.values)))
  }
  const m = mat.length
  const mean = grid.map((_, j) => mat.reduce((a, r) => a + r[j], 0) / m)
  const std = grid.map((_, j) => (m > 1 ? Math.sqrt(mat.reduce((a, r) => a + (r[j] - mean[j]) ** 2, 0) / (m - 1)) : 0))
  return { steps: grid, mean, std, n: m }
}

/** Debiased exponential moving average (TensorBoard-style). */
export function smooth(values: number[], weight: number): number[] {
  if (weight <= 0) return values
  let last = 0, num = 0
  return values.map((v) => {
    last = last * weight + (1 - weight) * v
    num++
    return last / (1 - Math.pow(weight, num))
  })
}

const SPLIT_ORDER = ['train', 'val', 'test', 'eval', 'rollout', 'episode', 'final']
const PRIMARY = ['loss', 'acc', 'f1', 'precision', 'recall', 'rmse', 'mse', 'mae', 'r2', 'bleu', 'chrf', 'rouge1', 'rouge2', 'rougeL',
  'reward', 'mean_reward', 'ep_rew_mean', 'length', 'ep_len_mean', 'score']

export type MetricGroup = { name: string; keys: string[]; primary: boolean }

/** Group 'split/name' keys by name so train/val/test of the same metric share one chart. */
export function groupMetricKeys(keys: string[]): MetricGroup[] {
  const groups = new Map<string, string[]>()
  for (const k of keys) {
    const name = k.includes('/') ? k.slice(k.indexOf('/') + 1) : k
    if (!groups.has(name)) groups.set(name, [])
    groups.get(name)!.push(k)
  }
  const splitRank = (k: string) => {
    const i = SPLIT_ORDER.indexOf(k.split('/')[0])
    return i < 0 ? 99 : i
  }
  const out = [...groups.entries()].map(([name, ks]) => ({
    name,
    keys: ks.sort((a, b) => splitRank(a) - splitRank(b) || a.localeCompare(b)),
    primary: PRIMARY.includes(name),
  }))
  return out.sort((a, b) => {
    const ia = PRIMARY.indexOf(a.name), ib = PRIMARY.indexOf(b.name)
    return (ia < 0 ? 999 : ia) - (ib < 0 ? 999 : ib) || a.name.localeCompare(b.name)
  })
}

export function allKeys(series: SeriesByRun): string[] {
  const s = new Set<string>()
  for (const r of Object.values(series)) for (const k of Object.keys(r)) s.add(k)
  return [...s]
}

export const SERIES_VARS = ['--s1', '--s2', '--s3', '--s4', '--s5', '--s6', '--s7', '--s8']
