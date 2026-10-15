// Types + helpers for the MLOps pages.

export type Version = {
  id: string; model_name: string; version: number; stage: string; source_session_id: string | null; source_run_id: string | null
  model_id: string; framework: string; task: string; modality: string; path: string; metrics: Record<string, number>
  description: string; created_by: string; created_at: number
}
export type RegisteredModel = {
  name: string; description: string; task: string; modality: string; model_id: string; created_at: number
  versions: Version[]; deployments: { id: string; name: string; champion_version_id: string | null; challenger_version_id: string | null }[]
}
export type FeatureProfile = {
  type: 'numeric' | 'categorical'; edges?: number[]; categories?: string[]; ref_hist: number[]
  mean?: number; std?: number; min?: number; max?: number; missing?: number
}
export type VersionDetail = Version & {
  card: Record<string, any>
  profile: { kind: string; task: string; classes: string[] | null; baseline: Record<string, number>; n_reference: number
    schema?: Record<string, string>; example?: Record<string, unknown>; features: Record<string, FeatureProfile>; prediction: any }
  size_bytes: number
  deployed_in: { id: string; name: string; role: string }[]
}

export type Reason = { severity: string; kind: string; key: string; message: string; value?: number }
export type Deployment = {
  id: string; name: string; model_name: string; kind: string; status: string; health: string; health_reasons: Reason[]
  champion_version_id: string | null; challenger_version_id: string | null; challenger_mode: string; canary_fraction: number
  champion: { id: string; version: number; stage: string } | null; challenger: { id: string; version: number; stage: string } | null
  requests_24h: number; error_rate_24h: number; labeled_24h: number; open_alerts: number; critical_alerts: number
  traffic: { scenario: string; rps: number; ramp_minutes: number; label_delay: number; elapsed_min: number } | null
  latest: { ts: number | null; n?: number; max_psi?: number; pred_psi?: number; latency_p95?: number; perf?: Headline | null }
  created_at: number; last_monitor_at: number | null
}
export type Headline = { key: string; value: number; baseline: number | null; drop?: number; se?: number; significant?: boolean }
export type DeploymentDetail = Deployment & {
  api_key: string; endpoint: string
  config: Record<string, any>; automation: Automation
  versions: (Pick<Version, 'id' | 'version' | 'stage' | 'metrics' | 'created_at' | 'created_by' | 'description' | 'source_session_id'>)[]
  schema?: { kind: string; task: string; classes: string[] | null; input_schema?: Record<string, string>; example?: Record<string, unknown>; baseline?: Record<string, number> }
}
export type Automation = {
  enabled: boolean; triggers: string[]; retrain_every_hours: number; evaluate_every_hours: number; cooldown_hours: number
  data: { source: string; feedback_days: number; min_labeled: number; problem_kind?: string | null; problem_id?: string | null; problem_params?: Record<string, unknown> }
  min_improvement: number; auto_promote: boolean; shadow_first: boolean; warm_start: boolean
}
export type Snapshot = {
  ts: number; n: number; health: string; error_rate: number; throughput_per_min: number; latency: { p50?: number; p95?: number; p99?: number }
  max_psi: number | null; mean_psi: number | null; n_drifted: number; feature_psi: Record<string, number | null>; feature_mean: Record<string, number | null>
  prediction: { psi?: number; cur_hist?: number[]; classes?: string[]; mean?: number }
  score_mean: number | null
  performance: { n_labeled: number; metrics: Record<string, number> | null; headline: Headline | null }
  shadow?: { n: number; agreement: number; performance: any } | null
  quality_rate: number
}
export type FeatureDrift = {
  type: string; psi: number | null; status: string; missing_rate: number; n: number; cur_hist?: number[]; mean?: number; mean_shift_sd?: number
  out_of_range_rate?: number; ks?: number; ks_p?: number; wasserstein_sd?: number; js?: number; unseen_rate?: number; unseen_examples?: string[]
  missing_delta?: number
}
export type Analysis = {
  n: number; window: [number, number]; errors: number; error_rate: number; throughput_per_min: number; latency: Record<string, number>
  features: Record<string, FeatureDrift>
  drift: { max_psi: number | null; mean_psi: number | null; n_drifted: number; n_warn: number; top: string[] }
  quality: { worst: string | null; worst_rate: number; issues: Record<string, number> }
  prediction?: { type: string; classes?: string[]; edges?: number[]; cur_hist: number[]; ref_hist: number[]; psi: number | null; mean?: number; ref_mean?: number }
  score?: { cur_hist: number[]; ref_hist: number[]; edges: number[]; mean: number; ref_mean: number; psi: number }
  performance: { n_labeled: number; metrics?: Record<string, number>; headline?: Headline; confusion?: { labels: string[]; matrix: number[][] }
    calibration?: { bin: number; pred: number; obs: number; n: number }[]; label_dist?: Record<string, number> }
  health: string; reasons: Reason[]
  reference: Record<string, FeatureProfile>
  reference_prediction: any
  shadow?: { n: number; agreement: number; performance: any; latency_p50: number | null }
  by_version: Record<string, number>
}
export type Alert = { id: number; deployment_id: string; key: string; kind: string; severity: string; message: string; status: string
  created_at: number; updated_at: number; resolved_at: number | null; value: number | null; deployment_name?: string }
export type DepEvent = { id: number; ts: number; kind: string; message: string; data: Record<string, any> }
export type AutomationRun = { id: string; kind: string; trigger: string; status: string; created_at: number; updated_at: number
  challenger_session_id: string | null; champion_session_id: string | null; challenger_version_id: string | null
  decision: string | null; message: string | null; data: Record<string, any> }

export const HEALTH_LABEL: Record<string, string> = { healthy: 'Healthy', warning: 'Warning', critical: 'Critical', no_data: 'No data' }
export const HEALTH_CLASS: Record<string, string> = { healthy: 'completed', warning: 'queued', critical: 'failed', no_data: '' }

/** Readable bin labels for a reference histogram. */
export function binLabels(f: { type: string; edges?: number[]; categories?: string[] }): string[] {
  if (f.type === 'categorical' || f.categories) return [...(f.categories ?? []), 'other']
  const e = f.edges ?? []
  const fmt = (v: number) => (Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(Math.abs(v) >= 10000 ? 0 : 1)}k` : Number(v.toPrecision(3)).toString())
  if (!e.length) return ['all']
  return [`< ${fmt(e[0])}`, ...e.slice(1).map((v, i) => `${fmt(e[i])}–${fmt(v)}`), `≥ ${fmt(e[e.length - 1])}`]
}

export function psiStatus(v: number | null | undefined, warn = 0.1, alert = 0.25): string {
  if (v === null || v === undefined) return 'unknown'
  return v >= alert ? 'alert' : v >= warn ? 'warn' : 'ok'
}
