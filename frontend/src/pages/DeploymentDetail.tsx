import { useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api, fmtMetric, fmtNum, fmtTime, fmtValue } from '../api'
import { baseLayout, Heatmap, Plot, useTheme, withAlpha, type Theme } from '../components/Chart'
import { ConfirmButton, ErrorBox, HealthBadge, Modal, StageBadge, usePoll } from '../components/ui'
import { RecordWaterfall } from '../components/Explain'
import {
  binLabels, psiStatus, type Alert, type Analysis, type AutomationRun, type DepEvent, type DeploymentDetail as Detail, type Snapshot,
} from '../mlops'

const TABS = ['Overview', 'Drift', 'Predictions', 'Performance', 'Versions', 'Automation', 'Settings', 'Integration', 'Events'] as const
const WINDOWS = [{ label: '15 min', m: 15 }, { label: '1 hour', m: 60 }, { label: '6 hours', m: 360 }, { label: '24 hours', m: 1440 }, { label: '7 days', m: 10080 }]

export default function DeploymentDetail() {
  const { id = '' } = useParams()
  const [d, err, reload] = usePoll(() => api.get<Detail>(`/deployments/${id}`), 5000, [id])
  const [snaps] = usePoll(() => api.get<Snapshot[]>(`/deployments/${id}/snapshots?hours=24`), 15000, [id])
  const [tab, setTab] = useState<(typeof TABS)[number]>('Overview')
  const [actErr, setActErr] = useState<unknown>()
  const nav = useNavigate()
  const act = async (fn: () => Promise<unknown>) => { setActErr(undefined); try { await fn(); reload() } catch (e) { setActErr(e) } }

  if (err) return <ErrorBox error={err} />
  if (!d) return <div className="muted">Loading…</div>
  return (
    <div>
      <div className="page-head">
        <div>
          <div className="row" style={{ gap: 10 }}>
            <h1>{d.name}</h1>
            <HealthBadge health={d.status === 'paused' ? 'no_data' : d.health} />
            {d.status === 'paused' && <span className="tag">paused</span>}
          </div>
          <div className="sub">
            <Link to="/registry">{d.model_name}</Link> · champion <b>v{d.champion?.version}</b>
            {d.challenger && <> · challenger <b>v{d.challenger.version}</b> in {d.challenger_mode}{d.challenger_mode === 'canary' && ` (${Math.round(d.canary_fraction * 100)}% of traffic)`}</>}
            {' · '}{d.kind === 'internal' ? 'served by this dashboard' : 'hosted elsewhere, pushes logs'}
            {d.last_monitor_at && <> · checked {fmtTime(d.last_monitor_at)}</>}
          </div>
        </div>
        <div className="row">
          <button onClick={() => act(() => api.post(`/deployments/${id}/monitor`))}>Check now</button>
          <button onClick={() => act(() => api.patch(`/deployments/${id}`, { status: d.status === 'active' ? 'paused' : 'active' }))}>{d.status === 'active' ? 'Pause' : 'Resume'}</button>
          <ConfirmButton className="danger" onConfirm={() => act(async () => { await api.del(`/deployments/${id}`); nav('/deployments') })}>Delete</ConfirmButton>
        </div>
      </div>
      <ErrorBox error={actErr} />
      {d.model_name === 'loan-default' || d.traffic ? <TrafficPanel d={d} onChange={reload} /> : null}
      <div className="tabs" role="tablist" style={{ marginTop: 14 }}>
        {TABS.map((t) => <button key={t} className={tab === t ? 'on' : ''} onClick={() => setTab(t)}>{t}{t === 'Overview' && d.open_alerts > 0 ? ` (${d.open_alerts})` : ''}</button>)}
      </div>
      {tab === 'Overview' && <Overview d={d} snaps={snaps ?? []} onChange={reload} />}
      {tab === 'Drift' && <DriftTab d={d} snaps={snaps ?? []} />}
      {tab === 'Predictions' && <PredictionsTab d={d} snaps={snaps ?? []} />}
      {tab === 'Performance' && <PerformanceTab d={d} snaps={snaps ?? []} />}
      {tab === 'Versions' && <VersionsTab d={d} onChange={reload} />}
      {tab === 'Automation' && <AutomationTab d={d} onChange={reload} />}
      {tab === 'Settings' && <SettingsTab d={d} onChange={reload} />}
      {tab === 'Integration' && <IntegrationTab d={d} />}
      {tab === 'Events' && <EventsTab id={id} />}
    </div>
  )
}

// ------------------------------------------------------------------ charts
type TLine = { name: string; x: number[]; y: (number | null)[]; color: string; dash?: string; fill?: boolean }

function TimeChart({ lines, yTitle, thresholds = [], height = 240, yRange, stacked }: {
  lines: TLine[]; yTitle: string; thresholds?: { y: number; label: string; color: string }[]; height?: number; yRange?: [number, number]; stacked?: boolean
}) {
  const theme = useTheme()
  if (!lines.some((l) => l.x.length)) return <div className="empty small">No snapshots yet</div>
  const data: Plotly.Data[] = lines.map((l) => ({
    type: 'scatter', mode: l.x.length > 1 ? 'lines' : 'markers', name: l.name, x: l.x.map((t) => new Date(t * 1000)), y: l.y,
    line: { color: l.color, width: 2, dash: l.dash as Plotly.Dash }, marker: { color: l.color, size: 7 }, connectgaps: false,
    ...(stacked ? { stackgroup: 'one', fillcolor: withAlpha(l.color, 0.55), line: { color: l.color, width: 0.5 } } : {}),
    hovertemplate: `${l.name}: %{y:.4g}<extra></extra>`,
  }))
  const layout = { ...baseLayout(theme, '', yTitle), height, margin: { l: 56, r: 12, t: 30, b: 34 } }
  layout.shapes = thresholds.map((t) => ({ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: t.y, y1: t.y, line: { color: t.color, width: 1, dash: 'dot' } }))
  layout.annotations = thresholds.map((t) => ({ xref: 'paper', x: 1, y: t.y, text: t.label, showarrow: false, xanchor: 'right', yanchor: 'bottom', font: { size: 10, color: theme.muted } }))
  if (yRange) layout.yaxis = { ...layout.yaxis, range: yRange }
  return <div style={{ height }}><Plot data={data} layout={layout} useResizeHandler style={{ width: '100%', height: '100%' }} config={{ displaylogo: false, responsive: true }} /></div>
}

function CompareBars({ labels, ref, cur, height = 240, theme, title }: { labels: string[]; ref: number[]; cur: number[]; height?: number; theme: Theme; title?: string }) {
  const layout = { ...baseLayout(theme, '', 'share'), height, barmode: 'group' as const, hovermode: 'x unified' as const, margin: { l: 50, r: 10, t: title ? 36 : 28, b: 60 } }
  if (title) layout.title = { text: title, font: { size: 12, color: theme.text2 } }
  layout.yaxis = { ...layout.yaxis, tickformat: '.0%' }
  layout.xaxis = { ...layout.xaxis, type: 'category', tickangle: labels.length > 6 ? -30 : 0 }
  return (
    <div style={{ height }}>
      <Plot useResizeHandler style={{ width: '100%', height: '100%' }} config={{ displaylogo: false, responsive: true }} layout={layout}
        data={[{ type: 'bar', name: 'training (reference)', x: labels, y: ref, marker: { color: withAlpha(theme.muted, 0.55) }, hovertemplate: '%{y:.1%}' },
          { type: 'bar', name: 'production (window)', x: labels, y: cur, marker: { color: theme.series[0] }, hovertemplate: '%{y:.1%}' }]} />
    </div>
  )
}

const x = (s: Snapshot[]) => s.map((v) => v.ts)

// ------------------------------------------------------------------ overview
function Overview({ d, snaps, onChange }: { d: Detail; snaps: Snapshot[]; onChange: () => void }) {
  const theme = useTheme()
  const [alerts, , reloadAlerts] = usePoll(() => api.get<Alert[]>(`/deployments/${d.id}/alerts`), 5000, [d.id])
  const s = snaps.filter((v) => v.n > 0)
  const perfKey = [...s].reverse().find((v) => v.performance?.headline)?.performance.headline?.key
  const cfg = d.config
  const last = d.latest
  const alertAct = (a: Alert, action: string) => api.post(`/deployments/${d.id}/alerts/${a.id}/${action}`).then(() => { reloadAlerts(); onChange() })
  const classes = s.find((v) => v.prediction?.classes)?.prediction.classes ?? []

  return (
    <div className="col" style={{ gap: 16 }}>
      {d.health_reasons.length > 0 && d.health !== 'no_data' && (
        <div className={d.health === 'critical' ? 'error-box' : 'notice'}>
          <b>Why {d.health}:</b>
          <ul style={{ margin: '4px 0 0 18px', padding: 0 }}>{d.health_reasons.map((r) => <li key={r.key}>{r.severity === 'critical' ? '✕ ' : '! '}{r.message}</li>)}</ul>
        </div>
      )}
      <div className="tiles">
        <Tile k="Requests (24h)" v={d.requests_24h.toLocaleString()} s={`${(d.error_rate_24h * 100).toFixed(2)}% errors`} />
        <Tile k={`Live ${last.perf?.key ?? 'performance'}`} v={last.perf ? fmtMetric(last.perf.key, last.perf.value) : '—'}
          s={last.perf ? `baseline ${fmtMetric(last.perf.key, last.perf.baseline ?? undefined)}${last.perf.se ? ` · ±${(2 * last.perf.se).toFixed(3)}` : ''}` : `${d.labeled_24h} labels so far`} />
        <Tile k="Max feature drift (PSI)" v={last.max_psi != null ? last.max_psi.toFixed(3) : '—'} s={psiStatus(last.max_psi, cfg.psi_warn, cfg.psi_alert)} />
        <Tile k="Prediction drift (PSI)" v={last.pred_psi != null ? last.pred_psi.toFixed(3) : '—'} s={psiStatus(last.pred_psi, cfg.psi_warn, cfg.psi_alert)} />
        <Tile k="p95 latency" v={last.latency_p95 != null ? `${last.latency_p95.toFixed(1)} ms` : '—'} s={`window ${cfg.window_minutes} min`} />
        <Tile k="Labeled (24h)" v={d.labeled_24h.toLocaleString()} s="ground truth received" />
      </div>
      {(alerts ?? []).length > 0 && (
        <div className="card">
          <div className="card-head"><h3>Open alerts</h3></div>
          <table><tbody>{alerts!.map((a) => (
            <tr key={a.id}>
              <td style={{ width: 120 }}><span className={`badge ${a.severity === 'critical' ? 'failed' : 'queued'}`}><span className="dot" />{a.severity}</span></td>
              <td>{a.message}{a.status === 'acknowledged' && <span className="muted small"> · acknowledged</span>}</td>
              <td className="small muted">since {fmtTime(a.created_at)}</td>
              <td style={{ textAlign: 'right' }}>
                {a.status === 'open' && <button className="sm ghost" onClick={() => alertAct(a, 'acknowledge')}>Ack</button>}
                <button className="sm ghost" onClick={() => alertAct(a, 'resolve')}>Resolve</button>
              </td>
            </tr>))}</tbody></table>
        </div>
      )}
      <div className="grid-2">
        <div className="card chart-card"><h3>Live {perfKey ?? 'performance'} vs. training baseline</h3>
          <TimeChart yTitle={perfKey ?? ''} lines={[
            { name: `live ${perfKey ?? ''}`, x: x(s), y: s.map((v) => v.performance?.headline?.value ?? null), color: theme.series[0] },
            { name: 'baseline', x: x(s), y: s.map((v) => v.performance?.headline?.baseline ?? null), color: theme.muted, dash: 'dash' },
          ]} /></div>
        <div className="card chart-card"><h3>Input drift (PSI)</h3>
          <TimeChart yTitle="PSI" thresholds={[{ y: cfg.psi_warn, label: 'moderate', color: '#eda100' }, { y: cfg.psi_alert, label: 'significant', color: '#e34948' }]}
            lines={[{ name: 'max feature PSI', x: x(s), y: s.map((v) => v.max_psi), color: theme.series[1] },
              { name: 'mean feature PSI', x: x(s), y: s.map((v) => v.mean_psi), color: theme.series[0] },
              { name: 'prediction PSI', x: x(s), y: s.map((v) => v.prediction?.psi ?? null), color: theme.series[2] }]} /></div>
        <div className="card chart-card"><h3>Prediction mix over time</h3>
          {classes.length ? <TimeChart yTitle="share" stacked yRange={[0, 1]} lines={classes.map((c, i) => ({ name: c, x: x(s), y: s.map((v) => v.prediction?.cur_hist?.[i] ?? null), color: theme.series[i % 8] }))} />
            : <TimeChart yTitle="mean prediction" lines={[{ name: 'mean', x: x(s), y: s.map((v) => v.prediction?.mean ?? null), color: theme.series[0] }]} />}</div>
        <div className="card chart-card"><h3>Traffic</h3>
          <TimeChart yTitle="requests / min" lines={[{ name: 'requests / min', x: x(s), y: s.map((v) => v.throughput_per_min), color: theme.series[0] },
            { name: 'error rate × 100', x: x(s), y: s.map((v) => (v.error_rate ?? 0) * 100), color: theme.series[7] }]} /></div>
        <div className="card chart-card"><h3>Latency per prediction</h3>
          <TimeChart yTitle="ms" lines={[{ name: 'p95', x: x(s), y: s.map((v) => v.latency?.p95 ?? null), color: theme.series[1] },
            { name: 'p50', x: x(s), y: s.map((v) => v.latency?.p50 ?? null), color: theme.series[0] }]} /></div>
      </div>
    </div>
  )
}

function Tile({ k, v, s }: { k: string; v: string; s?: string }) {
  const tone = s === 'alert' ? 'var(--critical-text)' : s === 'warn' ? '#b07800' : undefined
  return <div className="card tile"><div className="k">{k}</div><div className="v num">{v}</div><div className="s" style={tone ? { color: tone } : undefined}>{s === 'ok' ? 'stable' : s === 'warn' ? 'moderate shift' : s === 'alert' ? 'significant shift' : s}</div></div>
}

// ------------------------------------------------------------------ drift
function useAnalysis(id: string, minutes: number) {
  return usePoll(() => api.get<Analysis>(`/deployments/${id}/analysis?minutes=${minutes}`), 20000, [id, minutes])
}

function WindowPicker({ value, onChange }: { value: number; onChange: (m: number) => void }) {
  return <div className="segmented">{WINDOWS.map((w) => <button key={w.m} className={value === w.m ? 'on' : ''} onClick={() => onChange(w.m)}>{w.label}</button>)}</div>
}

function DriftTab({ d, snaps }: { d: Detail; snaps: Snapshot[] }) {
  const theme = useTheme()
  const [minutes, setMinutes] = useState(60)
  const [a, err] = useAnalysis(d.id, minutes)
  const [sel, setSel] = useState<string>()
  const rows = useMemo(() => Object.entries(a?.features ?? {}).sort((x, y) => (y[1].psi ?? -1) - (x[1].psi ?? -1)), [a])
  const cur = sel ?? rows[0]?.[0]
  const f = cur ? a?.features[cur] : undefined
  const ref = cur ? a?.reference[cur] : undefined
  const s = snaps.filter((v) => v.n > 0)
  return (
    <div className="col" style={{ gap: 14 }}>
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <div className="small text-2">Each feature's production distribution in the window vs. its training distribution. PSI &lt; {d.config.psi_warn} stable · ≥ {d.config.psi_alert} significant.</div>
        <WindowPicker value={minutes} onChange={setMinutes} />
      </div>
      <ErrorBox error={err} />
      {a && <div className="small text-2">{a.n.toLocaleString()} predictions in window · {a.drift.n_drifted} drifted, {a.drift.n_warn} moderately shifted features</div>}
      <div className="grid-2" style={{ gridTemplateColumns: 'minmax(0, 1.2fr) minmax(0, 1fr)' }}>
        <div className="card table-wrap">
          <table>
            <thead><tr><th>Feature</th><th className="num">PSI</th><th className="num">KS</th><th className="num">Shift</th><th className="num">Missing</th><th className="num">Out of range / new</th></tr></thead>
            <tbody>{rows.map(([k, v]) => (
              <tr key={k} className="clickable" onClick={() => setSel(k)} style={k === cur ? { background: 'var(--accent-soft)' } : undefined}>
                <td className="mono"><span className={`badge ${v.status === 'alert' ? 'failed' : v.status === 'warn' ? 'queued' : 'completed'}`} style={{ padding: '0 6px', marginRight: 6 }}><span className="dot" /></span>{k}</td>
                <td className="num" style={{ fontWeight: v.status !== 'ok' ? 700 : 400 }}>{v.psi != null ? v.psi.toFixed(3) : '—'}</td>
                <td className="num">{v.ks != null ? v.ks.toFixed(3) : v.js != null ? `JS ${v.js.toFixed(3)}` : '—'}</td>
                <td className="num">{v.mean_shift_sd != null ? `${v.mean_shift_sd >= 0 ? '+' : ''}${v.mean_shift_sd.toFixed(2)} sd` : '—'}</td>
                <td className="num">{(v.missing_rate * 100).toFixed(1)}%</td>
                <td className="num">{v.out_of_range_rate != null ? `${(v.out_of_range_rate * 100).toFixed(1)}%` : v.unseen_rate != null ? `${(v.unseen_rate * 100).toFixed(1)}%` : '—'}</td>
              </tr>))}</tbody>
          </table>
        </div>
        <div className="col" style={{ gap: 12 }}>
          {f && ref && f.cur_hist && (
            <div className="card chart-card">
              <h3 className="mono">{cur}</h3>
              <CompareBars theme={theme} labels={binLabels(ref)} ref={ref.ref_hist} cur={f.cur_hist} />
              {f.unseen_examples && f.unseen_examples.length > 0 && <div className="small text-2">New categories: {f.unseen_examples.join(', ')}</div>}
              {ref.type === 'numeric' && <div className="small text-2" style={{ padding: '0 4px 8px' }}>training mean {fmtNum(ref.mean)} · production mean {fmtNum(f.mean)} · training range {fmtNum(ref.min)} – {fmtNum(ref.max)}</div>}
            </div>
          )}
          {cur && (
            <div className="card chart-card"><h3>PSI over time · <span className="mono">{cur}</span></h3>
              <TimeChart height={200} yTitle="PSI" thresholds={[{ y: d.config.psi_alert, label: 'significant', color: '#e34948' }]}
                lines={[{ name: cur, x: x(s), y: s.map((v) => v.feature_psi?.[cur] ?? null), color: theme.series[1] }]} /></div>
          )}
        </div>
      </div>
      {a && Object.keys(a.quality.issues).length > 0 && (
        <div className="notice small"><b>Data quality:</b> {Object.entries(a.quality.issues).map(([k, v]) => `${k} (${(v * 100).toFixed(1)}% bad values)`).join(', ')}</div>
      )}
    </div>
  )
}

// ------------------------------------------------------------------ predictions
type LogRow = { id: number; request_id: string; ts: number; payload: any; features: any; prediction: any; score: number | null
  latency_ms: number | null; error: string | null; label: any; label_ts: number | null; source: string; version: number }

function PredictionsTab({ d, snaps }: { d: Detail; snaps: Snapshot[] }) {
  const theme = useTheme()
  const [minutes, setMinutes] = useState(60)
  const [a] = useAnalysis(d.id, minutes)
  const [log] = usePoll(() => api.get<LogRow[]>(`/deployments/${d.id}/predictions?limit=60`), 5000, [d.id])
  const [open, setOpen] = useState<LogRow>()
  const p = a?.prediction
  const s = snaps.filter((v) => v.n > 0)
  return (
    <div className="col" style={{ gap: 14 }}>
      <div className="row" style={{ justifyContent: 'flex-end' }}><WindowPicker value={minutes} onChange={setMinutes} /></div>
      <div className="grid-2">
        {p && (
          <div className="card chart-card"><h3>Prediction distribution {p.psi != null && <span className="muted small">· PSI {p.psi.toFixed(3)}</span>}</h3>
            <CompareBars theme={theme} labels={p.classes ?? binLabels({ type: 'numeric', edges: p.edges })} ref={p.ref_hist} cur={p.cur_hist} /></div>
        )}
        {a?.score && (
          <div className="card chart-card"><h3>Score distribution {d.schema?.classes?.length === 2 ? `· P(${d.schema.classes[1]})` : ''} <span className="muted small">· mean {a.score.mean.toFixed(3)} vs {a.score.ref_mean?.toFixed(3)} in training</span></h3>
            <CompareBars theme={theme} labels={binLabels({ type: 'numeric', edges: a.score.edges })} ref={a.score.ref_hist} cur={a.score.cur_hist} /></div>
        )}
        <div className="card chart-card"><h3>Mean score over time</h3>
          <TimeChart yTitle="score" lines={[{ name: 'mean score', x: x(s), y: s.map((v) => v.score_mean), color: theme.series[0] }]} /></div>
      </div>
      <div className="card">
        <div className="card-head"><h3>Recent predictions</h3><span className="small text-2">click a row for the full input</span></div>
        <div className="table-wrap" style={{ maxHeight: 460 }}>
          <table>
            <thead><tr><th>Time</th><th>Version</th><th>Input</th><th>Prediction</th><th className="num">Score</th><th>Ground truth</th><th className="num">Latency</th></tr></thead>
            <tbody>{(log ?? []).map((r) => (
              <tr key={r.id} className="clickable" onClick={() => setOpen(r)}>
                <td className="small">{new Date(r.ts * 1000).toLocaleTimeString()}</td>
                <td className="mono small">v{r.version}{r.source === 'pushed' && <span className="muted"> ext</span>}</td>
                <td className="small text-2" style={{ maxWidth: 380, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{summarize(r.payload ?? r.features)}</td>
                <td>{r.error ? <span className="err-text small">error</span> : fmtValue(r.prediction)}</td>
                <td className="num">{r.score != null ? r.score.toFixed(3) : '—'}</td>
                <td>{r.label == null ? <span className="muted small">pending</span> : <span style={{ color: String(r.label) === String(r.prediction) ? 'var(--good-text)' : 'var(--critical-text)' }}>{String(r.label) === String(r.prediction) ? '✓ ' : '✗ '}{fmtValue(r.label)}</span>}</td>
                <td className="num small">{r.latency_ms != null ? `${r.latency_ms.toFixed(1)} ms` : '—'}</td>
              </tr>))}</tbody>
          </table>
        </div>
      </div>
      {open && (
        <Modal title={`Request ${open.request_id}`} onClose={() => setOpen(undefined)}>
          <div className="col" style={{ gap: 12 }}>
            {d.schema?.kind === 'tabular' && !open.error && (
              <div className="card chart-card"><h3>Why this prediction?</h3>
                <RecordWaterfall path={`/deployments/${d.id}/predictions/${open.id}/explain`} /></div>
            )}
            <details><summary>Raw log record</summary><pre className="log mono small">{JSON.stringify(open, null, 2)}</pre></details>
          </div>
        </Modal>
      )}
    </div>
  )
}

function summarize(v: any): string {
  if (v == null) return '—'
  if (typeof v === 'string') return v
  if (Array.isArray(v)) return `[${v.slice(0, 6).map((n) => (typeof n === 'number' ? n.toFixed(2) : n)).join(', ')}${v.length > 6 ? ', …' : ''}]`
  return Object.entries(v).slice(0, 6).map(([k, x]) => `${k}=${typeof x === 'number' ? Number(x.toPrecision(4)) : x}`).join(', ')
}

// ------------------------------------------------------------------ performance
function PerformanceTab({ d, snaps }: { d: Detail; snaps: Snapshot[] }) {
  const theme = useTheme()
  const [minutes, setMinutes] = useState(60)
  const [a] = useAnalysis(d.id, minutes)
  const s = snaps.filter((v) => v.performance?.metrics)
  const keys = [...new Set(s.flatMap((v) => Object.keys(v.performance.metrics ?? {})))]
  const perf = a?.performance
  return (
    <div className="col" style={{ gap: 14 }}>
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <div className="small text-2">Computed from predictions whose ground-truth labels have arrived ({perf?.n_labeled ?? 0} in this window). Labels usually lag predictions.</div>
        <WindowPicker value={minutes} onChange={setMinutes} />
      </div>
      <div className="card chart-card"><h3>Live metrics over time (rolling {d.config.window_minutes}-minute windows)</h3>
        <TimeChart yTitle="metric" lines={keys.map((k, i) => ({ name: k, x: x(s), y: s.map((v) => v.performance.metrics?.[k] ?? null), color: theme.series[i % 8] }))} /></div>
      <div className="grid-2">
        {perf?.metrics && (
          <div className="card card-pad">
            <h3 style={{ marginBottom: 8 }}>This window</h3>
            <table><thead><tr><th>Metric</th><th className="num">Live</th><th className="num">Training baseline</th></tr></thead>
              <tbody>{Object.entries(perf.metrics).map(([k, v]) => <tr key={k}><td className="mono">{k}</td><td className="num">{fmtNum(v, 4)}</td>
                <td className="num text-2">{fmtNum((d.schema?.baseline as any)?.[`test/${k}`] ?? (perf.headline?.key === k ? perf.headline?.baseline : undefined), 4)}</td></tr>)}</tbody></table>
            {perf.headline?.se != null && <div className="small text-2" style={{ marginTop: 8 }}>{perf.headline.key} 95% interval ±{(2 * perf.headline.se).toFixed(3)} (bootstrap) — drops smaller than this are noise.</div>}
          </div>
        )}
        {perf?.confusion && <div className="card chart-card"><h3>Confusion matrix (labeled predictions)</h3><Heatmap labels={perf.confusion.labels} matrix={perf.confusion.matrix} /></div>}
        {perf?.calibration && perf.calibration.length > 0 && (
          <div className="card chart-card"><h3>Calibration: predicted P({d.schema?.classes?.[1]}) vs. observed rate</h3>
            <div style={{ height: 280 }}><Plot useResizeHandler style={{ width: '100%', height: '100%' }} config={{ displaylogo: false, responsive: true }}
              layout={{ ...baseLayout(theme, 'predicted probability', 'observed rate'), height: 280, hovermode: 'closest', showlegend: false,
                xaxis: { ...baseLayout(theme, 'predicted probability').xaxis, range: [0, 1] }, yaxis: { ...baseLayout(theme, '', 'observed rate').yaxis, range: [0, 1] } }}
              data={[{ x: [0, 1], y: [0, 1], type: 'scatter', mode: 'lines', line: { color: theme.muted, dash: 'dot', width: 1 }, hoverinfo: 'skip' },
                { x: perf.calibration.map((c) => c.pred), y: perf.calibration.map((c) => c.obs), text: perf.calibration.map((c) => `n=${c.n}`) as any, type: 'scatter', mode: 'lines+markers',
                  line: { color: theme.series[0], width: 2 }, marker: { size: perf.calibration.map((c) => Math.min(16, 5 + Math.sqrt(c.n))), color: theme.series[0] },
                  hovertemplate: 'predicted %{x:.2f} → observed %{y:.2f} (%{text})<extra></extra>' }]} /></div></div>
        )}
        {a?.shadow && (
          <div className="card card-pad">
            <h3 style={{ marginBottom: 8 }}>Champion vs. shadow challenger (v{d.challenger?.version})</h3>
            <div className="kv small">
              <div>Requests scored</div><div className="num">{a.shadow.n}</div>
              <div>Agreement with champion</div><div className="num">{(a.shadow.agreement * 100).toFixed(1)}%</div>
              {Object.entries(a.shadow.performance?.metrics ?? {}).map(([k, v]) => <div key={k} style={{ display: 'contents' }}><div>Shadow {k}</div>
                <div className="num">{fmtNum(v as number, 4)} <span className="muted">vs champion {fmtNum(perf?.metrics?.[k], 4)}</span></div></div>)}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ versions
function VersionsTab({ d, onChange }: { d: Detail; onChange: () => void }) {
  const [err, setErr] = useState<unknown>()
  const [frac, setFrac] = useState(String(d.canary_fraction))
  const act = (fn: () => Promise<unknown>) => fn().then(onChange).catch(setErr)
  return (
    <div className="col" style={{ gap: 14 }}>
      <ErrorBox error={err} />
      <div className="small text-2">
        <b>Promote</b> hot-swaps the production model (no downtime) and is recorded for rollback.
        A <b>shadow</b> challenger scores every request silently so you can compare it on live traffic; a <b>canary</b> answers a fraction of real requests.
      </div>
      <div className="row">
        <ConfirmButton onConfirm={() => act(() => api.post(`/deployments/${d.id}/rollback`))}>↶ Roll back to previous version</ConfirmButton>
        {d.challenger && (
          <>
            <div className="segmented">
              {['shadow', 'canary'].map((m) => <button key={m} className={d.challenger_mode === m ? 'on' : ''} onClick={() => act(() => api.patch(`/deployments/${d.id}`, { challenger_mode: m }))}>{m}</button>)}
            </div>
            {d.challenger_mode === 'canary' && <span className="row small">traffic share <input style={{ width: 70 }} value={frac} onChange={(e) => setFrac(e.target.value)} /><button className="sm" onClick={() => act(() => api.patch(`/deployments/${d.id}`, { canary_fraction: Number(frac) }))}>Set</button></span>}
            <button onClick={() => act(() => api.patch(`/deployments/${d.id}`, { challenger_version_id: null }))}>Remove challenger</button>
          </>
        )}
      </div>
      <div className="card table-wrap">
        <table>
          <thead><tr><th>Version</th><th>Stage</th><th>Role</th><th className="num">Offline test metric</th><th>Created</th><th></th></tr></thead>
          <tbody>{d.versions.map((v) => {
            const role = v.id === d.champion_version_id ? 'champion' : v.id === d.challenger_version_id ? `challenger (${d.challenger_mode})` : ''
            const mk = Object.keys(v.metrics).find((k) => ['test/acc', 'test/rmse', 'test/bleu', 'final/mean_reward'].includes(k))
            return (
              <tr key={v.id}>
                <td className="mono" style={{ fontWeight: 600 }}>v{v.version}{v.created_by === 'automation' && <span className="tag" style={{ marginLeft: 6 }}>auto</span>}</td>
                <td><StageBadge stage={v.stage} /></td>
                <td>{role}</td>
                <td className="num">{mk ? <>{fmtMetric(mk, v.metrics[mk])} <span className="muted small">{mk}</span></> : '—'}</td>
                <td className="small">{fmtTime(v.created_at)}{v.source_session_id && <> · <Link to={`/sessions/${v.source_session_id}`}>session</Link></>}</td>
                <td style={{ textAlign: 'right' }}>
                  {v.id !== d.champion_version_id && (
                    <div className="row" style={{ justifyContent: 'flex-end', gap: 4 }}>
                      {v.id !== d.challenger_version_id && <button className="sm" onClick={() => act(() => api.patch(`/deployments/${d.id}`, { challenger_version_id: v.id, challenger_mode: 'shadow' }))}>Shadow test</button>}
                      <ConfirmButton className="sm primary" onConfirm={() => act(() => api.post(`/deployments/${d.id}/promote`, { version_id: v.id }))}>Promote</ConfirmButton>
                    </div>
                  )}
                </td>
              </tr>
            )
          })}</tbody>
        </table>
      </div>
      <div className="small text-2">New versions come from <Link to="/registry">registering</Link> a trained session, or from automated retraining.</div>
    </div>
  )
}

// ------------------------------------------------------------------ automation
function AutomationTab({ d, onChange }: { d: Detail; onChange: () => void }) {
  const [a, setA] = useState(d.automation)
  const [err, setErr] = useState<unknown>()
  const [saved, setSaved] = useState(false)
  const [runs, , reloadRuns] = usePoll(() => api.get<AutomationRun[]>(`/deployments/${d.id}/automation/runs`), 5000, [d.id])
  const set = (patch: Partial<typeof a>) => setA((x) => ({ ...x, ...patch }))
  const setData = (patch: Partial<typeof a.data>) => setA((x) => ({ ...x, data: { ...x.data, ...patch } }))
  const save = () => api.patch(`/deployments/${d.id}`, { automation: a }).then(() => { setSaved(true); setTimeout(() => setSaved(false), 1500); onChange() }).catch(setErr)
  const run = (kind: string) => api.post(`/deployments/${d.id}/automation/run`, { kind }).then(reloadRuns).catch(setErr)
  const decide = (r: AutomationRun, action: string) => api.post(`/deployments/${d.id}/automation/runs/${r.id}/${action}`).then(() => { reloadRuns(); onChange() }).catch(setErr)
  const num = (v: string) => (v === '' ? 0 : Number(v))
  const trig = (t: string, on: boolean) => set({ triggers: on ? [...a.triggers, t] : a.triggers.filter((x) => x !== t) })

  return (
    <div className="col" style={{ gap: 14 }}>
      <ErrorBox error={err} />
      <div className="card card-pad col" style={{ gap: 12 }}>
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <h2>Continuous training</h2>
          <label className="check"><input type="checkbox" checked={a.enabled} onChange={(e) => set({ enabled: e.target.checked })} /> <b>Enabled</b></label>
        </div>
        <div className="small text-2">When triggered, a challenger is retrained on fresh data and the champion is re-evaluated on the <i>same</i> data split.
          The challenger is registered only if it beats the champion by at least the minimum improvement.</div>
        <div className="grid-3">
          <div className="col" style={{ gap: 4 }}>
            <span className="lbl small" style={{ fontWeight: 550 }}>Retrain when</span>
            {[['drift', 'input or prediction drift is detected'], ['performance', 'live performance drops (critical)'], ['schedule', 'on a fixed schedule']].map(([t, l]) => (
              <label key={t} className="check small"><input type="checkbox" checked={a.triggers.includes(t)} onChange={(e) => trig(t, e.target.checked)} />{l}</label>))}
          </div>
          <label className="field"><span className="lbl">Retrain every (hours)</span><input value={a.retrain_every_hours} onChange={(e) => set({ retrain_every_hours: num(e.target.value) })} /><span className="help">schedule trigger · 0 = off</span></label>
          <label className="field"><span className="lbl">Cooldown (hours)</span><input value={a.cooldown_hours} onChange={(e) => set({ cooldown_hours: num(e.target.value) })} /><span className="help">min time between drift/performance retrains</span></label>
          <label className="field"><span className="lbl">Re-test champion every (hours)</span><input value={a.evaluate_every_hours} onChange={(e) => set({ evaluate_every_hours: num(e.target.value) })} /><span className="help">scheduled evaluation on fresh data · 0 = off</span></label>
          <label className="field"><span className="lbl">Training data</span>
            <select value={a.data.source} onChange={(e) => setData({ source: e.target.value })}>
              <option value="feedback">labeled production traffic (feedback)</option>
              <option value="original">the original training dataset (re-loaded, e.g. an updated CSV)</option>
              <option value="live">a live data source (recent window)</option>
              <option value="problem">a specific dataset…</option>
            </select></label>
          {a.data.source === 'live' && <LiveSourcePick data={a.data as any} set={(p) => setData(p as any)} />}
          {a.data.source === 'feedback' && <>
            <label className="field"><span className="lbl">Use labels from the last (days)</span><input value={a.data.feedback_days} onChange={(e) => setData({ feedback_days: num(e.target.value) })} /></label>
            <label className="field"><span className="lbl">Minimum labeled examples</span><input value={a.data.min_labeled} onChange={(e) => setData({ min_labeled: num(e.target.value) })} /></label>
          </>}
          {a.data.source === 'problem' && <>
            <label className="field"><span className="lbl">Dataset id</span><input value={a.data.problem_id ?? ''} onChange={(e) => setData({ problem_id: e.target.value, problem_kind: 'dataset' })} placeholder="e.g. upload.my_updated_csv" /></label>
            <label className="field"><span className="lbl">Dataset params (JSON)</span><input className="mono" value={JSON.stringify(a.data.problem_params ?? {})} onChange={(e) => { try { setData({ problem_params: JSON.parse(e.target.value) }) } catch { /* keep typing */ } }} /></label>
          </>}
          <label className="field"><span className="lbl">Minimum improvement to accept</span><input value={a.min_improvement} onChange={(e) => set({ min_improvement: num(e.target.value) })} /><span className="help">absolute gain on the headline test metric</span></label>
          <div className="col" style={{ gap: 4 }}>
            <label className="check small"><input type="checkbox" checked={a.auto_promote} onChange={(e) => set({ auto_promote: e.target.checked })} /><b>Auto-promote</b>&nbsp;passing challengers</label>
            <label className="check small"><input type="checkbox" checked={a.shadow_first} onChange={(e) => set({ shadow_first: e.target.checked })} />Run passing challengers in shadow first</label>
            <label className="check small"><input type="checkbox" checked={a.warm_start} onChange={(e) => set({ warm_start: e.target.checked })} />Warm-start from the champion</label>
          </div>
        </div>
        <div className="row"><button className="primary" onClick={save}>Save policy</button>{saved && <span className="muted small">Saved</span>}
          <span className="grow" /><button onClick={() => run('retrain')}>Retrain challenger now</button><button onClick={() => run('evaluate')}>Re-test champion now</button></div>
      </div>
      <div className="card">
        <div className="card-head"><h3>Automation runs</h3></div>
        {!runs?.length ? <div className="empty small">No runs yet.</div> : (
          <div className="table-wrap"><table>
            <thead><tr><th>Started</th><th>Kind</th><th>Trigger</th><th>Status</th><th className="num">Champion</th><th className="num">Challenger</th><th>Sessions</th><th></th></tr></thead>
            <tbody>{runs.map((r) => (
              <tr key={r.id}>
                <td className="small">{fmtTime(r.created_at)}</td><td>{r.kind}</td><td>{r.trigger}</td>
                <td><span className={`badge ${r.status === 'promoted' || r.status === 'completed' ? 'completed' : r.status === 'running' ? 'running' : r.status === 'awaiting_approval' ? 'queued' : r.status === 'failed' ? 'failed' : 'stopped'}`}><span className="dot" />{r.status.replace('_', ' ')}</span>
                  {r.decision && <div className="small text-2" style={{ maxWidth: 340 }}>{r.decision}</div>}</td>
                <td className="num">{r.data.champion_score != null ? fmtNum(r.data.champion_score, 4) : '—'}</td>
                <td className="num">{r.data.challenger_score != null ? <>{fmtNum(r.data.challenger_score, 4)}{r.data.gain != null && <div className={`small ${r.data.gain > 0 ? '' : 'err-text'}`}>{r.data.gain > 0 ? '+' : ''}{r.data.gain.toFixed(4)}</div>}</> : '—'}</td>
                <td className="small">{r.challenger_session_id && <Link to={`/sessions/${r.challenger_session_id}`}>challenger</Link>} {r.champion_session_id && <Link to={`/sessions/${r.champion_session_id}`}>champion eval</Link>}</td>
                <td>{r.status === 'awaiting_approval' && <div className="row" style={{ gap: 4 }}><button className="sm primary" onClick={() => decide(r, 'approve')}>Approve</button><button className="sm" onClick={() => decide(r, 'reject')}>Reject</button></div>}</td>
              </tr>))}</tbody>
          </table></div>
        )}
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ settings
const SETTINGS: [string, string, string][] = [
  ['window_minutes', 'Monitoring window (minutes)', 'each check looks at this much recent traffic'],
  ['interval_minutes', 'Check every (minutes)', ''],
  ['min_samples', 'Minimum predictions per window', 'fewer = "no data"'],
  ['psi_warn', 'PSI warning', 'industry default 0.1'], ['psi_alert', 'PSI alert', 'industry default 0.25'],
  ['perf_warn_drop', 'Performance drop: warning', 'absolute, vs. baseline'], ['perf_alert_drop', 'Performance drop: critical', ''],
  ['perf_min_labeled', 'Labels needed for performance alerts', ''],
  ['error_rate_alert', 'Error rate: critical', 'fraction of failed requests'], ['latency_p95_warn_ms', 'p95 latency warning (ms)', ''],
  ['quality_warn', 'Data quality warning', 'extra missing / out-of-range / unseen rate'], ['retention_days', 'Keep logs for (days)', ''],
]

function SettingsTab({ d, onChange }: { d: Detail; onChange: () => void }) {
  const [c, setC] = useState<Record<string, any>>(d.config)
  const [err, setErr] = useState<unknown>()
  const [saved, setSaved] = useState(false)
  const save = () => api.patch(`/deployments/${d.id}`, { config: Object.fromEntries(Object.entries(c).map(([k, v]) => [k, k === 'device' || k === 'performance_metric' ? v : Number(v)])) })
    .then(() => { setSaved(true); setTimeout(() => setSaved(false), 1500); onChange() }).catch(setErr)
  return (
    <div className="card card-pad col" style={{ gap: 12 }}>
      <h2>Monitoring & serving</h2>
      <ErrorBox error={err} />
      <div className="grid-3">
        {SETTINGS.map(([k, l, h]) => <label key={k} className="field"><span className="lbl">{l}</span><input value={c[k] ?? ''} onChange={(e) => setC({ ...c, [k]: e.target.value })} />{h && <span className="help">{h}</span>}</label>)}
        <label className="field"><span className="lbl">Headline performance metric</span>
          <select value={c.performance_metric ?? 'auto'} onChange={(e) => setC({ ...c, performance_metric: e.target.value })}>
            {['auto', 'auc', 'acc', 'f1', 'precision', 'recall', 'rmse', 'mae', 'r2', 'bleu', 'episode_return'].map((m) => <option key={m}>{m}</option>)}
          </select><span className="help">auto = AUC for binary classifiers, else accuracy / RMSE / BLEU</span></label>
        <label className="field"><span className="lbl">Serving device</span>
          <select value={c.device ?? 'cpu'} onChange={(e) => setC({ ...c, device: e.target.value })}><option>cpu</option><option>cuda</option></select></label>
      </div>
      <div className="row"><button className="primary" onClick={save}>Save</button>{saved && <span className="muted small">Saved</span>}</div>
    </div>
  )
}

// ------------------------------------------------------------------ integration
function IntegrationTab({ d }: { d: Detail }) {
  const [show, setShow] = useState(false)
  const key = show ? d.api_key : '<API_KEY>'
  const example = JSON.stringify(d.schema?.example ?? {}, null, 0)
  const base = d.endpoint.replace(/\/api\/serve\/.*$/, '')
  return (
    <div className="col" style={{ gap: 14 }}>
      <div className="card card-pad col">
        <h2>Endpoint & key</h2>
        <div className="kv small"><div>Prediction endpoint</div><div className="mono">{d.endpoint}/predict</div>
          <div>Feedback (ground truth)</div><div className="mono">{d.endpoint}/feedback</div>
          <div>Log push (external models)</div><div className="mono">{d.endpoint}/log</div>
          <div>API key</div><div className="row mono">{show ? d.api_key : '•'.repeat(20)} <button className="sm" onClick={() => setShow(!show)}>{show ? 'Hide' : 'Show'}</button></div></div>
        {d.schema?.input_schema && <div className="small text-2">Input fields: {Object.entries(d.schema.input_schema).map(([k, t]) => `${k} (${t})`).join(', ')}</div>}
      </div>
      <div className="card card-pad col">
        <h3>1 · Call the model served here</h3>
        <pre className="log mono small">{`curl -X POST ${d.endpoint}/predict \\
  -H "X-API-Key: ${key}" -H "Content-Type: application/json" \\
  -d '{"instances": [${example}]}'

# later, when the true outcome is known (joins on request_id)
curl -X POST ${d.endpoint}/feedback -H "X-API-Key: ${key}" -H "Content-Type: application/json" \\
  -d '{"items": [{"request_id": "<id from the response>", "label": "<true label>"}]}'`}</pre>
        <pre className="log mono small">{`# Python — copy backend/mltrain/sdk/mltrain_client.py into your app (stdlib only)
from mltrain_client import Client
c = Client("${base}", "${d.id}", api_key="${key}")
out = c.predict([${example}])
c.feedback(out["request_ids"][0], label="...")`}</pre>
      </div>
      <div className="card card-pad col">
        <h3>2 · Model hosted somewhere else</h3>
        <div className="small text-2">Download the serving package (FastAPI app + Dockerfile + SDK). It predicts locally and pushes logs back here,
          so this dashboard never needs to reach into your deployment.</div>
        <div className="row">{d.versions.slice(0, 4).map((v) => <a key={v.id} className="btn sm" href={`/api/registry/versions/${v.id}/export?deployment=${d.id}`}>Package v{v.version}</a>)}</div>
        <pre className="log mono small">{`# or instrument your own service:
c = Client("${base}", "${d.id}", api_key="${key}")
rid = c.log(input=features, prediction=pred, score=prob, latency_ms=ms, model_version=${d.champion?.version ?? 1})
c.feedback(rid, label=truth)   # whenever ground truth arrives`}</pre>
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ events
function EventsTab({ id }: { id: string }) {
  const [ev] = usePoll(() => api.get<DepEvent[]>(`/deployments/${id}/events`), 5000, [id])
  return (
    <div className="card">
      <div className="card-head"><h3>Audit log</h3><span className="small text-2">promotions, rollbacks, automation decisions, health changes, config edits</span></div>
      <table><tbody>{(ev ?? []).map((e) => (
        <tr key={e.id}><td className="small" style={{ width: 160 }}>{fmtTime(e.ts)}</td><td style={{ width: 110 }}><span className="tag">{e.kind}</span></td><td>{e.message}</td></tr>
      ))}</tbody></table>
    </div>
  )
}

// ------------------------------------------------------------------ demo traffic
const SCENARIOS: [string, string][] = [
  ['none', 'Normal traffic (no drift)'], ['downturn', 'Economic downturn — gradual: incomes & credit scores fall, defaults rise'],
  ['new_segment', 'New customer segment — sudden: young, thin-file applicants'], ['broken_feature', 'Broken feature — upstream bug sends income in thousands'],
  ['concept', 'Concept drift — same applicants, more defaults (only labels reveal it)'],
]

function TrafficPanel({ d, onChange }: { d: Detail; onChange: () => void }) {
  const [scenario, setScenario] = useState('downturn')
  const [rps, setRps] = useState('6')
  const [ramp, setRamp] = useState('10')
  const [delay, setDelay] = useState('20')
  const [err, setErr] = useState<unknown>()
  const t = d.traffic
  const start = () => api.post(`/deployments/${d.id}/traffic`, { scenario, rps: Number(rps), ramp_minutes: Number(ramp), label_delay: Number(delay) }).then(onChange).catch(setErr)
  return (
    <div className="card card-pad col" style={{ gap: 8 }}>
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <h3>Demo traffic simulator</h3>
        {t && <span className="badge running"><span className="dot" />running “{t.scenario}” · {t.rps} req/s · {t.elapsed_min.toFixed(1)} min</span>}
      </div>
      <ErrorBox error={err} />
      <div className="row">
        <select style={{ maxWidth: 520 }} value={scenario} onChange={(e) => setScenario(e.target.value)}>{SCENARIOS.map(([k, l]) => <option key={k} value={k}>{l}</option>)}</select>
        <label className="row small">req/s <input style={{ width: 60 }} value={rps} onChange={(e) => setRps(e.target.value)} /></label>
        <label className="row small">ramp (min) <input style={{ width: 60 }} value={ramp} onChange={(e) => setRamp(e.target.value)} /></label>
        <label className="row small">label delay (s) <input style={{ width: 60 }} value={delay} onChange={(e) => setDelay(e.target.value)} /></label>
        <button className="primary" onClick={start}>{t ? 'Restart with scenario' : 'Start traffic'}</button>
        {t && <button onClick={() => api.del(`/deployments/${d.id}/traffic`).then(onChange).catch(setErr)}>Stop</button>}
      </div>
      <div className="small muted">Sends synthetic applications through the real prediction endpoint and reports the true outcome after the label delay.</div>
    </div>
  )
}

function LiveSourcePick({ data, set }: { data: Record<string, any>; set: (p: Record<string, any>) => void }) {
  const [sources] = usePoll(() => api.get<{ id: string; name: string; config: Record<string, any> }[]>('/data/sources'), 0)
  return (
    <>
      <label className="field"><span className="lbl">Live source</span>
        <select value={data.source_id ?? ''} onChange={(e) => set({ source_id: e.target.value })}>
          <option value="">choose…</option>{(sources ?? []).filter((s) => s.config.target).map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
        </select><span className="help">must have the same input columns as this deployment</span></label>
      <label className="field"><span className="lbl">Train on the last (hours)</span><input value={data.lookback_hours ?? 720} onChange={(e) => set({ lookback_hours: Number(e.target.value) })} /></label>
      <label className="field"><span className="lbl">Recency weighting</span>
        <select value={data.weighting ?? 'exponential'} onChange={(e) => set({ weighting: e.target.value })}>{['none', 'exponential', 'linear', 'step'].map((w) => <option key={w}>{w}</option>)}</select></label>
      <label className="field"><span className="lbl">Half-life (hours)</span><input value={data.half_life_hours ?? 168} onChange={(e) => set({ half_life_hours: Number(e.target.value) })} /></label>
    </>
  )
}
