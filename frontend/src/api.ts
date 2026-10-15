// Typed client for the FastAPI backend.

export type Param = {
  name: string
  type: 'float' | 'int' | 'bool' | 'str' | 'choice' | 'int_list' | 'json'
  default: unknown
  help?: string
  min?: number
  max?: number
  choices?: string[]
  suggestions?: string[]
  log?: boolean
  group: string
  labels?: Record<string, string>
  option_groups?: Record<string, string>
  show_if?: { param: string; in?: unknown[]; not_in?: unknown[] }
}

export type PluginMeta = {
  kind: 'model' | 'dataset' | 'env'
  id: string
  name: string
  description: string
  source: string
  params: Param[]
  task?: string
  modality?: string
  framework?: string
  tasks?: string[]
  modalities?: string[]
  uses_gpu?: boolean
  supports_init?: boolean
  supports_eval?: boolean
  supports_resume?: boolean
  eval_without_init?: boolean
  any_task?: boolean
  session_task?: string
}

export type Plugins = { models: PluginMeta[]; datasets: PluginMeta[]; envs: PluginMeta[]; errors: { file: string; error: string }[] }

export type Headline = { key: string; higher_is_better: boolean; mean: number; std: number; n: number }

export type SessionBrief = {
  id: string
  name: string
  created_at: number
  status: string
  problem_kind: 'dataset' | 'env'
  problem_id: string
  model_id: string
  task: string
  modality: string
  framework: string
  device: string
  parent_id: string | null
  tags: string[]
  n_seeds: number
  n_completed: number
  progress: number
  progress_text: string
  started_at: number | null
  ended_at: number | null
  headline: Headline | null
  mode: 'train' | 'evaluate'
  group_id: string | null
  group_kind: string | null
  group_params: Record<string, unknown>
  init_session_id: string | null
}

export type Run = {
  id: string
  seed: number
  status: string
  progress: number
  progress_text: string
  error: string | null
  started_at: number | null
  ended_at: number | null
  summary?: Record<string, number>
  init_run_id?: string | null
  storage?: RunStorage
}

export type RunStorage = { has_model: boolean; model_bytes: number; has_final?: boolean; final_bytes?: number; has_checkpoint: boolean; checkpoint_bytes: number; media_bytes: number; total_bytes: number }

export type Stat = { mean: number; std: number; min: number; max: number; n: number; per_seed: Record<string, number> }
export type Change = { param: string; from: unknown; to: unknown }

export type SessionDetail = SessionBrief & {
  problem_params: Record<string, unknown>
  model_params: Record<string, unknown>
  seeds: number[]
  notes: string
  env_info: { versions?: Record<string, string>; plugin_hashes?: Record<string, string> }
  runs: Run[]
  summary_stats: Record<string, Stat>
  parent: { id: string; name: string } | null
  changes_from_parent: Change[]
  children: { id: string; name: string; created_at: number; mode: string }[]
  evaluations: { id: string; name: string; problem_id: string; created_at: number }[]
  plugins: Record<string, PluginMeta | null>
  options: Record<string, unknown>
  init_from: { session_id: string; strategy: string; run_id?: string; variant?: string } | null
  component_users?: { id: string; name: string; model_id: string }[]
  source: { id: string; name: string } | null
}

export type Media = { id: number; run_id: string; seed: number; step: number; name: string; kind: string; path: string | null; data: any; t: number }

export type SavedModel = {
  session_id: string
  session_name: string
  created_at: number
  model_id: string
  model_name: string
  problem_id: string
  task: string
  modality: string
  framework: string
  model_params: Record<string, unknown>
  problem_params: Record<string, unknown>
  headline: Headline | null
  runs: { run_id: string; seed: number; bytes: number; summary: Record<string, number>; has_final?: boolean }[]
  bytes: number
  has_final?: boolean
}

export function fmtBytes(b: number | undefined | null): string {
  if (!b) return '0 B'
  const u = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.min(u.length - 1, Math.floor(Math.log(b) / Math.log(1024)))
  return `${(b / 1024 ** i).toFixed(i ? 1 : 0)} ${u[i]}`
}

export type Series = { steps: number[]; values: number[] }
export type SeriesByRun = Record<string, Record<string, Series>>

export type LineageNode = {
  id: string
  name: string
  parent_id: string | null
  created_at: number
  status: string
  model_id: string
  problem_id: string
  headline: Headline | null
  changes: Change[]
}

export type Artifact = { run_id: string; seed: number; name: string; kind: string; data: any }

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

let onUnauthorized: () => void = () => {}
export function setUnauthorizedHandler(fn: () => void) {
  onUnauthorized = fn
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method, credentials: 'same-origin', headers: {} }
  if (body instanceof FormData) init.body = body
  else if (body !== undefined) {
    init.body = JSON.stringify(body)
    ;(init.headers as Record<string, string>)['Content-Type'] = 'application/json'
  }
  const res = await fetch(`/api${path}`, init)
  if (res.status === 401 && !path.startsWith('/auth')) onUnauthorized()
  if (!res.ok) {
    let msg = res.statusText
    try {
      const j = await res.json()
      msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail)
    } catch {
      /* not json */
    }
    throw new ApiError(res.status, msg)
  }
  const ct = res.headers.get('content-type') || ''
  return (ct.includes('application/json') ? res.json() : res.text()) as Promise<T>
}

export const api = {
  get: <T>(p: string) => request<T>('GET', p),
  post: <T>(p: string, b?: unknown) => request<T>('POST', p, b),
  put: <T>(p: string, b?: unknown) => request<T>('PUT', p, b),
  patch: <T>(p: string, b?: unknown) => request<T>('PATCH', p, b),
  del: <T>(p: string) => request<T>('DELETE', p),
}

export function wsUrl(path: string) {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  return `${proto}://${location.host}/api${path}`
}

// ---------------------------------------------------------------- formatting helpers
export function fmtNum(v: unknown, digits = 4): string {
  if (typeof v !== 'number' || !isFinite(v)) return v === null || v === undefined ? '—' : String(v)
  if (Number.isInteger(v) && Math.abs(v) < 1e7) return v.toLocaleString()
  const a = Math.abs(v)
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(2)
  return Number(v.toPrecision(digits)).toString()
}

export function fmtMetric(key: string, v: number | undefined): string {
  if (v === undefined || v === null) return '—'
  const name = key.split('/').pop() || key
  if (['acc', 'precision', 'recall', 'f1'].includes(name) && v <= 1) return `${(v * 100).toFixed(2)}%`
  if (key.endsWith('_s')) return fmtDuration(v)
  return fmtNum(v)
}

export function fmtDuration(s: number | null | undefined): string {
  if (s === null || s === undefined || !isFinite(s)) return '—'
  if (s < 60) return `${s.toFixed(s < 10 ? 1 : 0)}s`
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`
  return `${Math.floor(s / 3600)}h ${Math.round((s % 3600) / 60)}m`
}

export function fmtTime(t: number | null | undefined): string {
  if (!t) return '—'
  const d = new Date(t * 1000)
  const today = new Date()
  const sameDay = d.toDateString() === today.toDateString()
  return sameDay
    ? d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : d.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

export function fmtValue(v: unknown): string {
  if (Array.isArray(v)) return v.join(', ')
  if (typeof v === 'object' && v !== null) return JSON.stringify(v)
  if (typeof v === 'number') return fmtNum(v, 6)
  if (v === '' || v === null || v === undefined) return '—'
  return String(v)
}
