import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api, fmtNum, fmtTime } from '../api'
import { baseLayout, Plot, useTheme } from '../components/Chart'
import { ConfirmButton, ErrorBox, HealthBadge, usePoll } from '../components/ui'
import { ago, Schedules } from './Data'

type Col = { missing: number | null; ref_missing: number | null; psi: number | null; mean: number | null; ref_mean: number | null; out_of_range: number | null
  unseen: number | null; unseen_examples: string[] | null; kind: string; is_target: boolean; drift_status: string | null; psi_noise: number | null }
type Latest = { ts: number; rows: number; first_ts: number; last_ts: number; lag_s: number; stale_after_s: number; window_rows: number; recent_rows: number; reference_rows: number
  quality_rows: number; volume: { last_hour: number; typical_hour: number | null; ratio: number | null }; schema_changes: { change: string; column: string; from?: string; to?: string }[]
  columns: Record<string, Col>; label_coverage?: number; duplicate_ids?: number; target_recent?: Record<string, number>; target_reference?: Record<string, number>
  reasons: { severity: string; kind: string; message: string }[]; health: string }
type Detail = { id: string; name: string; health: string; config: Record<string, any>; latest: Latest | null; last_error: string | null; dataset_id: string; is_demo: boolean
  sessions: { id: string; name: string; created_at: number }[] }
type Snap = { ts: number; health: string; lag_s: number | null; last_hour: number | null; typical_hour: number | null; label_coverage: number | null
  missing: Record<string, number | null>; psi: Record<string, number | null> }
type Volume = { hours: number[]; counts: number[]; target_daily?: { days: number[]; classes?: string[]; shares?: number[][]; mean?: number[] } }

const dur = (s?: number | null) => (s == null ? '—' : s < 90 ? `${Math.round(s)} s` : s < 5400 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`)

export default function SourceDetail() {
  const { id = '' } = useParams()
  const [d, err, reload] = usePoll(() => api.get<Detail>(`/data/sources/${id}`), 10000, [id])
  const [snaps] = usePoll(() => api.get<Snap[]>(`/data/sources/${id}/snapshots?hours=72`), 30000, [id])
  const [vol] = usePoll(() => api.get<Volume>(`/data/sources/${id}/volume?hours=168`), 60000, [id])
  const [rows] = usePoll(() => api.get<Record<string, unknown>[]>(`/data/sources/${id}/preview?n=25`), 15000, [id])
  const [actErr, setActErr] = useState<unknown>()
  const [busy, setBusy] = useState(false)
  const nav = useNavigate()
  const theme = useTheme()
  if (err) return <ErrorBox error={err} />
  if (!d) return <div className="muted">Loading…</div>
  const m = d.latest
  const cfg = d.config
  const act = (p: Promise<unknown>) => p.then(reload).catch(setActErr)
  const check = () => { setBusy(true); api.post(`/data/sources/${id}/profile`).then(reload).catch(setActErr).finally(() => setBusy(false)) }
  const cols = Object.entries(m?.columns ?? {}).sort((a, b) => (b[1].psi ?? -1) - (a[1].psi ?? -1))
  const snapX = (snaps ?? []).map((s) => new Date(s.ts * 1000))
  const topMissing = cols.filter(([, c]) => (c.missing ?? 0) > 0 || (c.ref_missing ?? 0) > 0).slice(0, 5).map(([k]) => k)
  const chart = (data: Plotly.Data[], layout: Partial<Plotly.Layout>, h = 240) => (
    <div style={{ height: h }}><Plot useResizeHandler style={{ width: '100%', height: '100%' }} config={{ displaylogo: false, responsive: true }} data={data} layout={{ ...layout, height: h }} /></div>
  )

  return (
    <div>
      <div className="page-head">
        <div>
          <div className="row" style={{ gap: 10 }}><h1>{d.name}</h1><HealthBadge health={d.health === 'unknown' ? 'no_data' : d.health} /><span className="tag">live</span></div>
          <div className="sub mono small">{cfg.connector.kind} · {cfg.connector.path ?? String(cfg.connector.url).replace(/:[^:@/]+@/, ':•••@')} · {cfg.table ?? 'query'} · time {cfg.ts_col}{cfg.target && <> · target {cfg.target} ({cfg.task})</>}</div>
          <div className="sub small">checked {m ? fmtTime(m.ts) : 'never'} · every {cfg.profile_every_minutes} min{cfg.description && <> · {cfg.description}</>}</div>
        </div>
        <div className="row">
          {cfg.target && <button className="primary" onClick={() => nav(`/new?problem=${d.dataset_id}`)}>Train on this source</button>}
          <button disabled={busy} onClick={check}>{busy ? 'Checking…' : 'Check now'}</button>
          <ConfirmButton className="danger" onConfirm={() => act(api.del(`/data/sources/${id}`)).then(() => nav('/data'))}>Remove</ConfirmButton>
        </div>
      </div>
      <ErrorBox error={actErr ?? d.last_error} />
      {d.is_demo && <DemoStream />}
      {m && m.reasons.length > 0 && (
        <div className={d.health === 'critical' ? 'error-box' : 'notice'} style={{ marginTop: 12 }}>
          <b>Why {d.health}:</b><ul style={{ margin: '4px 0 0 18px', padding: 0 }}>{m.reasons.map((r, i) => <li key={i}>{r.severity === 'critical' ? '✕ ' : '! '}{r.message}</li>)}</ul>
        </div>
      )}
      {m && <>
        <div className="tiles section">
          <div className="card tile"><div className="k">Rows</div><div className="v num">{m.rows?.toLocaleString()}</div><div className="s">since {m.first_ts ? new Date(m.first_ts * 1000).toLocaleDateString() : '—'}</div></div>
          <div className="card tile"><div className="k">Newest row</div><div className="v num">{dur(m.lag_s)}</div><div className="s">ago · stale after {dur(m.stale_after_s)}</div></div>
          <div className="card tile"><div className="k">Rows last hour</div><div className="v num">{m.volume.last_hour}</div><div className="s">typical {m.volume.typical_hour?.toFixed(0) ?? '—'}{m.volume.ratio != null && ` · ${(m.volume.ratio * 100).toFixed(0)}%`}</div></div>
          {m.label_coverage != null && <div className="card tile"><div className="k">Labeled (recent)</div><div className="v num">{(m.label_coverage * 100).toFixed(1)}%</div><div className="s">rows whose target is known</div></div>}
          <div className="card tile"><div className="k">Windows</div><div className="v num" style={{ fontSize: '1.1rem' }}>{m.recent_rows.toLocaleString()} / {m.reference_rows.toLocaleString()}</div><div className="s">last {cfg.recent_hours} h vs previous {cfg.reference_days} d</div></div>
          <div className="card tile"><div className="k">Schema</div><div className="v num">{m.schema_changes.length ? `${m.schema_changes.length} change${m.schema_changes.length > 1 ? 's' : ''}` : 'stable'}</div><div className="s">{Object.keys(m.columns).length} monitored columns</div></div>
        </div>
        <div className="grid-2 section">
          <div className="card chart-card"><h3>Rows per hour (7 days)</h3>
            {vol && chart([{ type: 'bar', x: vol.hours.map((h) => new Date(h * 1000)), y: vol.counts, marker: { color: theme.series[0] }, hovertemplate: '%{x|%a %H:00}: %{y} rows<extra></extra>' }],
              { ...baseLayout(theme, '', 'rows'), showlegend: false, bargap: 0.1 })}</div>
          <div className="card chart-card"><h3>Freshness: age of the newest row at each check</h3>
            {snaps && chart([{ type: 'scatter', mode: 'lines+markers', x: snapX, y: snaps.map((s) => (s.lag_s ?? 0) / 60), line: { color: theme.series[1], width: 2 }, marker: { size: 4 }, hovertemplate: '%{y:.1f} min<extra></extra>' }],
              { ...baseLayout(theme, '', 'minutes'), showlegend: false, shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: m.stale_after_s / 60, y1: m.stale_after_s / 60, line: { color: '#e34948', dash: 'dot', width: 1 } }] })}</div>
          {vol?.target_daily?.classes && (
            <div className="card chart-card"><h3>Target mix per day ({cfg.target})</h3>
              {chart(vol.target_daily.classes.map((c, i) => ({ type: 'bar', name: c, x: vol.target_daily!.days.map((t) => new Date(t * 1000)), y: vol.target_daily!.shares!.map((r) => r[i]),
                marker: { color: theme.series[i % 8] }, hovertemplate: `${c}: %{y:.1%}<extra></extra>` })), { ...baseLayout(theme, '', 'share'), barmode: 'stack', yaxis: { ...baseLayout(theme, '', 'share').yaxis, tickformat: '.0%' } })}</div>
          )}
          {vol?.target_daily?.mean && (
            <div className="card chart-card"><h3>Target mean per day ({cfg.target})</h3>
              {chart([{ type: 'scatter', mode: 'lines+markers', x: vol.target_daily.days.map((t) => new Date(t * 1000)), y: vol.target_daily.mean, line: { color: theme.series[0] } }], { ...baseLayout(theme, '', 'mean'), showlegend: false })}</div>
          )}
          <div className="card chart-card"><h3>Drift: last {cfg.recent_hours} h vs previous {cfg.reference_days} d (PSI)</h3>
            {chart([{ type: 'bar', orientation: 'h', y: cols.map(([k]) => k), x: cols.map(([, c]) => c.psi ?? 0),
              marker: { color: cols.map(([, c]) => ((c.psi ?? 0) >= Math.max(cfg.drift_alert ?? 0.25, c.psi_noise ?? 0) ? '#e34948' : (c.psi ?? 0) >= 0.1 ? '#eda100' : theme.series[0])) },
              hovertemplate: '%{y}: %{x:.3f}<extra></extra>' }],
              { ...baseLayout(theme, 'PSI'), showlegend: false, hovermode: 'closest', margin: { l: 10, r: 20, t: 10, b: 40 }, yaxis: { ...baseLayout(theme, '').yaxis, automargin: true, type: 'category', autorange: 'reversed' },
                shapes: [0.1, 0.25].map((v) => ({ type: 'line', x0: v, x1: v, yref: 'paper', y0: 0, y1: 1, line: { color: v === 0.25 ? '#e34948' : '#eda100', dash: 'dot', width: 1 } }))}, Math.max(220, 40 + cols.length * 24))}</div>
          {topMissing.length > 0 && snaps && (
            <div className="card chart-card"><h3>Missing values over time (last {cfg.quality_minutes ?? 30} min at each check)</h3>
              {chart(topMissing.map((k, i) => ({ type: 'scatter', mode: 'lines', name: k, x: snapX, y: snaps.map((s) => s.missing?.[k] ?? null), line: { color: theme.series[i % 8], width: 2 }, hovertemplate: `${k}: %{y:.1%}<extra></extra>` })),
                { ...baseLayout(theme, '', 'missing'), yaxis: { ...baseLayout(theme, '', 'missing').yaxis, tickformat: '.0%' } })}</div>
          )}
        </div>
        <div className="card section">
          <div className="card-head"><h3>Columns</h3><span className="small text-2">quality on the last {cfg.quality_minutes ?? 30} min ({m.quality_rows} rows) · drift on the last {cfg.recent_hours} h</span></div>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Column</th><th>Type</th><th className="num">Missing now</th><th className="num">Missing before</th><th className="num">PSI</th><th className="num">Mean now / before</th><th>Out of range / new values</th></tr></thead>
              <tbody>{cols.map(([k, c]) => (
                <tr key={k}>
                  <td className="mono">{k}{c.is_target && <span className="tag" style={{ marginLeft: 6 }}>target</span>}</td><td className="small">{c.kind}</td>
                  <td className="num" style={!c.is_target && (c.missing ?? 0) - (c.ref_missing ?? 0) >= 0.1 ? { color: 'var(--critical-text)', fontWeight: 700 } : undefined}
                    title={c.is_target ? 'Recent rows whose outcome is not known yet (label delay)' : undefined}>{c.missing != null ? `${(c.missing * 100).toFixed(1)}%` : '—'}{c.is_target && <span className="muted small"> pending</span>}</td>
                  <td className="num text-2">{c.ref_missing != null ? `${(c.ref_missing * 100).toFixed(1)}%` : '—'}</td>
                  <td className="num" style={c.drift_status === 'alert' ? { color: 'var(--critical-text)', fontWeight: 700 } : undefined}>{c.psi != null ? c.psi.toFixed(3) : '—'}</td>
                  <td className="num small">{c.mean != null ? `${fmtNum(c.mean)} / ${fmtNum(c.ref_mean)}` : '—'}</td>
                  <td className="small">{c.out_of_range ? `${(c.out_of_range * 100).toFixed(1)}% outside history` : ''}{c.unseen ? `${(c.unseen * 100).toFixed(1)}% new: ${(c.unseen_examples ?? []).join(', ')}` : ''}</td>
                </tr>))}</tbody>
            </table>
          </div>
        </div>
        {m.schema_changes.length > 0 && (
          <div className="card card-pad section col">
            <h3>Schema changes since the source was connected</h3>
            {m.schema_changes.map((c, i) => <div key={i} className="small mono">{c.change}: {c.column}{c.change === 'type' && ` (${c.from} → ${c.to})`}</div>)}
            <button style={{ alignSelf: 'flex-start' }} onClick={() => act(api.patch(`/data/sources/${id}`, { reset_schema_baseline: true }))}>Accept the current schema</button>
          </div>
        )}
      </>}
      <div className="section"><h2 style={{ marginBottom: 8 }}>Training</h2>
        <Schedules problemId={d.dataset_id} />
        {d.sessions.length > 0 && <div className="small text-2" style={{ marginTop: 8 }}>Recent sessions on this source: {d.sessions.slice(0, 8).map((s) => <Link key={s.id} to={`/sessions/${s.id}`} style={{ marginRight: 10 }}>{s.name}</Link>)}</div>}
      </div>
      {rows && rows.length > 0 && (
        <div className="card section">
          <div className="card-head"><h3>Newest rows</h3><span className="small text-2">live preview (read-only)</span></div>
          <div className="table-wrap" style={{ maxHeight: 360 }}>
            <table><thead><tr>{Object.keys(rows[0]).map((k) => <th key={k} className="mono">{k}</th>)}</tr></thead>
              <tbody>{rows.map((r, i) => <tr key={i}>{Object.entries(r).map(([k, v]) => <td key={k} className="small mono">{v == null ? <span className="muted">null</span> : String(v)}</td>)}</tr>)}</tbody></table>
          </div>
        </div>
      )}
      <SourceSettings d={d} onSaved={reload} />
    </div>
  )
}

const SETTINGS: [string, string, string][] = [
  ['profile_every_minutes', 'Check health every (min)', ''], ['recent_hours', 'Recent window (h)', 'compared for drift'],
  ['reference_days', 'Reference window (days)', ''], ['quality_minutes', 'Quality window (min)', 'missing / new values / range'],
  ['stale_minutes', 'Stale after (min)', 'empty = automatic (5 typical gaps)'], ['drift_alert', 'Drift alert (PSI)', 'default 0.25'],
  ['missing_increase_alert', 'Missing-rate jump alert', 'absolute, default 0.1'], ['volume_drop_ratio', 'Volume drop alert', 'last hour / typical, default 0.3'],
  ['volume_spike_ratio', 'Volume spike warning', 'default 3×'],
]

function SourceSettings({ d, onSaved }: { d: Detail; onSaved: () => void }) {
  const [c, setC] = useState<Record<string, any>>(d.config)
  const [err, setErr] = useState<unknown>()
  const [saved, setSaved] = useState(false)
  const save = () => {
    const out: Record<string, any> = {}
    for (const [k] of SETTINGS) out[k] = c[k] === '' || c[k] == null ? null : Number(c[k])
    out.exclude = typeof c.exclude === 'string' ? c.exclude.split(',').map((x: string) => x.trim()).filter(Boolean) : c.exclude
    api.patch(`/data/sources/${d.id}`, { config: out }).then(() => { setSaved(true); setTimeout(() => setSaved(false), 1500); onSaved() }).catch(setErr)
  }
  return (
    <details className="card card-pad section">
      <summary>Monitoring settings</summary>
      <ErrorBox error={err} />
      <div className="grid-3" style={{ marginTop: 10 }}>
        {SETTINGS.map(([k, l, h]) => <label key={k} className="field"><span className="lbl">{l}</span><input value={c[k] ?? ''} onChange={(e) => setC({ ...c, [k]: e.target.value })} />{h && <span className="help">{h}</span>}</label>)}
        <label className="field"><span className="lbl">Ignored columns</span><input value={Array.isArray(c.exclude) ? c.exclude.join(', ') : c.exclude ?? ''} onChange={(e) => setC({ ...c, exclude: e.target.value })} /></label>
      </div>
      <div className="row" style={{ marginTop: 10 }}><button className="primary" onClick={save}>Save</button>{saved && <span className="small muted">Saved</span>}</div>
    </details>
  )
}

function DemoStream() {
  const [st, , reload] = usePoll(() => api.get<{ pid?: number; rate?: number; incident?: string; started?: number }>('/data/demo/stream'), 5000)
  const [incident, setIncident] = useState('none')
  const [err, setErr] = useState<unknown>()
  const running = !!st?.pid
  return (
    <div className="card card-pad col" style={{ gap: 8, marginTop: 12 }}>
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <h3>Demo database stream</h3>
        {running ? <span className="badge running"><span className="dot" />streaming {((st!.rate ?? 0) * 3600).toFixed(0)} rows/h · incident: {st!.incident} · since {ago(st!.started)}</span> : <span className="badge stopped"><span className="dot" />stopped</span>}
      </div>
      <ErrorBox error={err} />
      <div className="row">
        <select style={{ maxWidth: 380 }} value={incident} onChange={(e) => setIncident(e.target.value)}>
          <option value="none">Normal data</option>
          <option value="missing_values">Incident: 60% of incomes arrive missing</option>
          <option value="new_category">Incident: new loan purpose "crypto" appears</option>
        </select>
        <button className="primary" onClick={() => api.post('/data/demo/stream', { incident }).then(reload).catch(setErr)}>{running ? 'Restart stream' : 'Start stream'}</button>
        {running && <button onClick={() => api.del('/data/demo/stream').then(reload).catch(setErr)}>Stop stream (source goes stale)</button>}
      </div>
      <div className="small muted">The demo history covers 30 days in which the economy worsens over the last 10 — try training on different time windows and recency weightings.</div>
    </div>
  )
}
