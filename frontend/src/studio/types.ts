import { api } from '../api'

export type Col = { name: string; type: string }
export type Warning = { step?: string; index?: number; kind: string; message: string }
export type Step = { id: string; type: string; params: Record<string, any>; enabled?: boolean }
export type Dataset = {
  id: string; name: string; slug: string; kind: 'source' | 'derived' | 'query'; parent_id: string | null; parent_name?: string | null
  origin: Record<string, any>; recipe: Step[]; query: string | null; target: string | null; task: string | null; description: string
  version: number; status: 'new' | 'building' | 'ready' | 'error'; error: string | null; rows: number | null; n_columns: number
  schema: Col[]; bytes: number | null; warnings: Warning[]; fitted: Record<string, any>; trace: { index: number; rows: number; columns: number }[]
  build_seconds: number | null; stale: boolean; building: boolean; created_at: number; updated_at: number; built_at: number | null
}
export type StepSpec = { type: string; label: string; category: string; help: string; fitted: boolean; python: boolean
  params: { name: string; type: string; default: any; label: string; help: string; choices?: string[] }[] }

export type ColProfile = {
  type: string; kind: string; missing: number; unique: number; min?: any; max?: any; mean?: number; std?: number; median?: number
  q1?: number; q3?: number; p01?: number; p99?: number; skew?: number; kurtosis?: number; zeros?: number; negative?: number; outliers?: number
  hist?: { edges: number[]; counts: number[] } | null; top?: { value: string; count: number; share: number }[]; numeric_like?: number; avg_length?: number
}
export type Alert = { severity: 'critical' | 'warning' | 'info'; column: string | null; kind: string; message: string }
export type Profile = {
  rows: number; columns: Record<string, ColProfile>; duplicates: number; split: Record<string, number> | null; sample_rows: number; sampled: boolean
  missing_cells: number; target: string | null; bytes: number | null; alerts: Alert[]
  correlations: { columns: string[]; pearson: (number | null)[][]; spearman: (number | null)[][]; strong_pairs: { a: string; b: string; r: number }[] } | null
  missingness: { columns: Record<string, number>; rows_with_missing: number; co_missing?: { columns: string[]; matrix: (number | null)[][] } }
  target_analysis?: { task: string; balance?: Record<string, number>; associations: { column: string; kind: string; mutual_info: number | null; corr: number | null
    rate_by_value?: Record<string, number>; positive_class?: string; mean_by_class?: Record<string, number | null> }[] }
  time: { column: string; unit: string; periods: string[]; counts: number[] } | null
}
export type ChartSpec = { type: string; x?: string; y?: string; color?: string; agg?: string; bins?: number; normalize?: boolean; filter?: string; top?: number; bucket?: string; value?: string }
export type ChartResult = { type: string; x?: string; y?: string; yTitle?: string; agg?: string; normalized?: boolean; note?: string; bucket?: string; series: any[] }

export const studio = {
  list: () => api.get<Dataset[]>('/studio/datasets'),
  get: (id: string) => api.get<Dataset>(`/studio/datasets/${id}`),
  create: (body: Record<string, unknown>) => api.post<Dataset>('/studio/datasets', body),
  patch: (id: string, body: Record<string, unknown>) => api.patch<Dataset>(`/studio/datasets/${id}`, body),
  build: (id: string) => api.post(`/studio/datasets/${id}/build`),
  remove: (id: string) => api.del(`/studio/datasets/${id}`),
  steps: () => api.get<StepSpec[]>('/studio/steps'),
  preview: (body: Record<string, unknown>) => api.post<any>('/studio/preview', body),
  profile: (id: string, refresh = false) => api.get<Profile>(`/studio/datasets/${id}/profile${refresh ? '?refresh=true' : ''}`),
  rows: (id: string, q: string) => api.get<{ total: number; columns: string[]; data: any[][] }>(`/studio/datasets/${id}/rows?${q}`),
  column: (id: string, name: string, by?: string) => api.get<any>(`/studio/datasets/${id}/column?name=${encodeURIComponent(name)}${by ? `&by=${encodeURIComponent(by)}` : ''}`),
  chart: (datasets: string[], spec: ChartSpec) => api.post<ChartResult>('/studio/chart', { datasets, spec }),
  compare: (datasets: string[], column: string, filter?: string) => api.post<any>('/studio/compare', { datasets, column, filter }),
  explore: (dataset: string, spec: Record<string, unknown>) => api.post<any>('/studio/explore', { dataset, spec }),
  apply: (body: Record<string, unknown>) => api.post<Dataset>('/studio/explore/apply', body),
  sql: (sql: string, limit = 1000) => api.post<{ columns: string[]; types?: string[]; rows: any[][]; row_count: number; truncated: boolean; seconds: number }>('/studio/sql', { sql, limit }),
  schema: () => api.get<{ name: string; label: string; kind: string; rows: number | null; id?: string; columns: Col[] }[]>('/studio/schema'),
  queries: () => api.get<{ id: string; name: string; sql: string; updated_at: number }[]>('/studio/queries'),
  saveQuery: (name: string, sql: string) => api.post('/studio/queries', { name, sql }),
  deleteQuery: (id: string) => api.del(`/studio/queries/${id}`),
  origins: () => api.get<any>('/studio/origins'),
  lineage: (id: string) => api.get<any>(`/studio/datasets/${id}/lineage`),
}

export const isNumericType = (t?: string) => !!t && /^(TINYINT|SMALLINT|INTEGER|BIGINT|HUGEINT|UTINYINT|USMALLINT|UINTEGER|UBIGINT|FLOAT|DOUBLE|REAL|DECIMAL)/.test(t)
export const fmtInt = (n?: number | null) => (n == null ? '—' : n.toLocaleString())
export const pct = (v?: number | null, d = 1) => (v == null ? '—' : `${(v * 100).toFixed(d)}%`)
export const short = (v: any): string => {
  if (v == null) return '∅'
  if (typeof v === 'number') {
    const a = Math.abs(v)
    if (Number.isInteger(v) && a < 1e9) return v.toLocaleString()
    if (a >= 1e9 || (a < 1e-3 && v !== 0)) return v.toExponential(2)
    if (a >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 1 })
    return String(Number(v.toPrecision(4)))
  }
  return String(v)
}
