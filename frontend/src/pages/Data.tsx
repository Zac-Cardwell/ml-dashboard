import { useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { api, fmtTime } from '../api'
import { QualityReport } from '../components/DataReport'
import { RemovedDatasets, RemoveDatasetModal } from '../components/DatasetRemoval'
import { ConfirmButton, ErrorBox, HealthBadge, Modal, usePoll } from '../components/ui'

export type Source = {
  id: string; name: string; health: string; health_reasons: { severity: string; kind: string; message: string }[]
  config: Record<string, any>; last_profiled_at: number | null; last_error: string | null; created_at: number
  latest: { rows?: number; last_ts?: number; lag_s?: number; volume?: { last_hour: number; typical_hour: number | null; ratio: number | null }; label_coverage?: number; recent_rows?: number }
}
type StaticDs = { id: string; name: string; description: string; task: string; modality: string; source: string; report: { grade: string; n_issues: number; created_at: number } | null }
export type Schedule = { id: string; name: string; enabled: number; template: Record<string, any>; cadence: Record<string, any>; cadence_text: string
  register_as: string | null; next_run_at: number | null; last_run_at: number | null; last_session_id: string | null
  runs: { session_id: string; created_at: number; status: string; registered_version: string | null; name: string }[] }

const TABS = ['Live sources', 'Static datasets', 'Schedules'] as const

export function ago(t?: number | null): string {
  if (!t) return '—'
  const s = Math.max(0, Date.now() / 1000 - t)
  return s < 90 ? `${Math.round(s)} s ago` : s < 5400 ? `${Math.round(s / 60)} min ago` : s < 172800 ? `${(s / 3600).toFixed(1)} h ago` : `${(s / 86400).toFixed(1)} days ago`
}

export default function Data() {
  const [sp, setSp] = useSearchParams()
  const tab = (TABS.find((t) => t.toLowerCase().startsWith(sp.get('tab') ?? 'live')) ?? 'Live sources') as (typeof TABS)[number]
  return (
    <div>
      <div className="page-head">
        <div>
          <h1>Data</h1>
          <div className="sub">Live sources keep growing and are monitored continuously; static datasets get a one-off quality report.</div>
        </div>
      </div>
      <div className="tabs">{TABS.map((t) => <button key={t} className={tab === t ? 'on' : ''} onClick={() => setSp({ tab: t.split(' ')[0].toLowerCase() })}>{t}</button>)}</div>
      {tab === 'Live sources' && <LiveSources />}
      {tab === 'Static datasets' && <StaticDatasets />}
      {tab === 'Schedules' && <Schedules />}
    </div>
  )
}

// ------------------------------------------------------------------ live sources
function LiveSources() {
  const [sources, err, reload] = usePoll(() => api.get<Source[]>('/data/sources'), 10000)
  const [adding, setAdding] = useState(false)
  const [demoBusy, setDemoBusy] = useState(false)
  const [demoErr, setDemoErr] = useState<unknown>()
  const nav = useNavigate()
  const hasDemo = sources?.some((s) => s.id === 'loan-applications')
  return (
    <div className="col" style={{ gap: 14 }}>
      <ErrorBox error={err ?? demoErr} />
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <div className="small text-2">A live source is a table, query or set of files — on this computer or in AWS / Google Cloud / Azure — that keeps receiving rows, with a timestamp column.</div>
        <div className="row">
          {!hasDemo && <button disabled={demoBusy} onClick={() => { setDemoBusy(true); api.post<{ id: string }>('/data/demo/setup').then((r) => nav(`/data/sources/${r.id}`)).catch(setDemoErr).finally(() => setDemoBusy(false)) }}>
            {demoBusy ? 'Creating demo database…' : 'Add demo live database'}</button>}
          <button className="primary" onClick={() => setAdding(true)}>+ Connect a source</button>
        </div>
      </div>
      {sources && !sources.length && <div className="card empty">No live sources yet. Connect a SQLite / PostgreSQL / MySQL database or a folder of CSV / Parquet files — or try the demo.</div>}
      <div className="grid-3" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(330px, 1fr))' }}>
        {(sources ?? []).map((s) => (
          <Link key={s.id} to={`/data/sources/${s.id}`} className="card card-pad col" style={{ gap: 6, color: 'inherit', textDecoration: 'none' }}>
            <div className="row" style={{ justifyContent: 'space-between' }}><h2>{s.name}</h2><HealthBadge health={s.health === 'unknown' ? 'no_data' : s.health} /></div>
            <div className="small text-2 mono">{s.config.connector?.kind} · {s.config.table ?? 'query'} · target {s.config.target ?? '—'}</div>
            <div className="kv small" style={{ marginTop: 4 }}>
              <div>Rows</div><div className="num">{s.latest.rows?.toLocaleString() ?? '—'}</div>
              <div>Newest row</div><div className="num">{ago(s.latest.last_ts)}</div>
              <div>Last hour</div><div className="num">{s.latest.volume ? `${s.latest.volume.last_hour} rows (typical ${s.latest.volume.typical_hour?.toFixed(0) ?? '—'})` : '—'}</div>
              <div>Labels (recent)</div><div className="num">{s.latest.label_coverage != null ? `${(s.latest.label_coverage * 100).toFixed(0)}%` : '—'}</div>
            </div>
            {s.health_reasons.length > 0 && <div className="small" style={{ color: s.health === 'critical' ? 'var(--critical-text)' : 'var(--text-2)' }}>{s.health_reasons[0].message}{s.health_reasons.length > 1 && ` (+${s.health_reasons.length - 1})`}</div>}
          </Link>
        ))}
      </div>
      {adding && <AddSource onClose={() => setAdding(false)} onCreated={(id) => { setAdding(false); reload(); nav(`/data/sources/${id}`) }} />}
    </div>
  )
}

const KIND_HELP: Record<string, string> = {
  sqlite: 'Path to a .db / .sqlite file. Opened read-only.',
  sql: 'SQLAlchemy URL, e.g. postgresql+psycopg://user:${DB_PASSWORD}@localhost:5432/mydb or mysql+pymysql://user:${DB_PASSWORD}@localhost/mydb — ${VAR} is read from data/cloud.env or the server environment, so the password isn\'t stored.',
  csv: 'A CSV file, a folder of CSVs, or a glob like /data/logs/*.csv — or cloud storage: s3://bucket/logs/, gs://bucket/*.csv, az://container/folder (credentials: Settings → Cloud data). New files are picked up automatically.',
  parquet: 'A Parquet file, a folder or a glob — local or s3:// · gs:// · az:// (credentials: Settings → Cloud data).',
}
const CLOUD_AUTH: Record<string, { key: string; label: string; placeholder: string }[]> = {
  aws: [{ key: 'profile', label: 'AWS profile', placeholder: 'default' }, { key: 'region', label: 'Region', placeholder: 'us-east-1' }],
  gcp: [{ key: 'credentials_file', label: 'Key file', placeholder: 'empty = default credentials' }, { key: 'project', label: 'Project', placeholder: '' }],
  azure: [{ key: 'account_name', label: 'Storage account', placeholder: '' }, { key: 'connection_string_env', label: 'Connection string env var', placeholder: '' }],
}
const providerOf = (p: string) => (/^s3a?:\/\//.test(p) ? 'aws' : /^gc?s:\/\//.test(p) ? 'gcp' : /^(az|abfss?|adl):\/\//.test(p) ? 'azure' : null)

function AddSource({ onClose, onCreated }: { onClose: () => void; onCreated: (id: string) => void }) {
  const [kind, setKind] = useState('sqlite')
  const [loc, setLoc] = useState('')
  const [auth, setAuth] = useState<Record<string, string>>({})
  const [wh] = usePoll(() => api.get<{ warehouses: Record<string, { label: string; template: string; help: string; driver: boolean }> }>('/cloud/status').then((r) => r.warehouses), 0)
  const [whHelp, setWhHelp] = useState('')
  const prov = providerOf(loc)
  const [tables, setTables] = useState<string[]>()
  const [table, setTable] = useState('')
  const [query, setQuery] = useState('')
  const [insp, setInsp] = useState<{ columns: { name: string; dtype: string }[]; timestamp_candidates: string[]; sample: Record<string, unknown>[] }>()
  const [m, setM] = useState({ name: '', ts_col: '', target: '', task: 'classification', id_col: '', exclude: '', timezone: 'UTC', profile_every_minutes: '15', recent_hours: '24', reference_days: '7' })
  const [err, setErr] = useState<unknown>()
  const [busy, setBusy] = useState(false)
  const conn = () => {
    const a = Object.fromEntries(Object.entries(auth).filter(([, v]) => v))
    return kind === 'sql' ? { kind, url: loc } : { kind, path: loc, ...(prov && Object.keys(a).length ? { auth: a } : {}) }
  }
  const connect = () => { setErr(undefined); setInsp(undefined); api.post<{ tables: string[] }>('/data/connect/tables', { connector: conn() }).then((r) => { setTables(r.tables); setTable(r.tables[0] ?? '') }).catch(setErr) }
  const inspect = () => {
    setErr(undefined)
    api.post<typeof insp>('/data/connect/inspect', { connector: conn(), table: query ? null : table, query: query || null }).then((r) => {
      setInsp(r)
      setM((x) => ({ ...x, name: x.name || table || 'live source', ts_col: r!.timestamp_candidates[0] ?? '', target: '', id_col: r!.columns.find((c) => c.name.toLowerCase() === 'id')?.name ?? '' }))
    }).catch(setErr)
  }
  const save = () => {
    setBusy(true); setErr(undefined)
    const config = { connector: conn(), table: query ? null : table, query: query || null, ts_col: m.ts_col, target: m.target || null, task: m.target ? m.task : null,
      id_col: m.id_col || null, exclude: m.exclude.split(',').map((x) => x.trim()).filter(Boolean), timezone: m.timezone,
      profile_every_minutes: Number(m.profile_every_minutes), recent_hours: Number(m.recent_hours), reference_days: Number(m.reference_days) }
    api.post<{ id: string }>('/data/sources', { name: m.name, config }).then((r) => onCreated(r.id)).catch(setErr).finally(() => setBusy(false))
  }
  const cols = insp?.columns.map((c) => c.name) ?? []
  const sel = (k: keyof typeof m, label: string, opts: string[], help?: string, blank?: string) => (
    <label className="field"><span className="lbl">{label}</span>
      <select value={m[k]} onChange={(e) => setM({ ...m, [k]: e.target.value })}>{blank !== undefined && <option value="">{blank}</option>}{opts.map((o) => <option key={o}>{o}</option>)}</select>
      {help && <span className="help">{help}</span>}</label>
  )
  return (
    <Modal title="Connect a live data source" onClose={onClose}>
      <div className="col" style={{ gap: 12 }}>
        <ErrorBox error={err} />
        <div className="row" style={{ flexWrap: 'nowrap' }}>
          <select style={{ width: 250 }} value={kind} onChange={(e) => { setKind(e.target.value); setTables(undefined); setInsp(undefined) }}>
            <option value="sqlite">SQLite</option><option value="sql">SQL database / cloud warehouse</option><option value="csv">CSV files (local or cloud)</option>
            <option value="parquet">Parquet files (local or cloud)</option>
          </select>
          <input placeholder={kind === 'sql' ? 'postgresql+psycopg://user:pw@localhost/db' : '/path/to/data'} value={loc} onChange={(e) => setLoc(e.target.value)} />
          <button onClick={connect} disabled={!loc}>Connect</button>
        </div>
        <div className="small text-2">{KIND_HELP[kind]}</div>
        {kind === 'sql' && wh && (
          <div className="row small" style={{ gap: 6 }}>Cloud warehouse templates:
            {Object.entries(wh).map(([k, w]) => <button key={k} className="sm ghost" onClick={() => { setLoc(w.template); setWhHelp(`${w.help}${w.driver ? '' : ' (driver not installed)'}`) }}>{w.label}</button>)}
          </div>
        )}
        {kind === 'sql' && whHelp && <div className="small text-2">{whHelp} Replace the UPPERCASE parts.</div>}
        {kind !== 'sql' && prov && (
          <div className="row" style={{ gap: 8 }}>
            {CLOUD_AUTH[prov].map((f) => (
              <label key={f.key} className="field" style={{ width: 210 }}><span className="lbl">{f.label} <span className="muted">(optional)</span></span>
                <input value={auth[f.key] ?? ''} placeholder={f.placeholder} onChange={(e) => setAuth({ ...auth, [f.key]: e.target.value })} /></label>))}
            <span className="small muted" style={{ alignSelf: 'center' }}>Overrides Settings → Cloud data for this source only</span>
          </div>
        )}
        {tables && (
          <div className="row" style={{ flexWrap: 'nowrap' }}>
            {(kind === 'sqlite' || kind === 'sql') ? <>
              <select value={table} onChange={(e) => setTable(e.target.value)} style={{ maxWidth: 260 }}>{tables.map((t) => <option key={t}>{t}</option>)}</select>
              <input className="mono" placeholder="…or a SQL query (optional)" value={query} onChange={(e) => setQuery(e.target.value)} />
            </> : <span className="small text-2">{tables.length} file(s) found</span>}
            <button className="primary" onClick={inspect}>Inspect</button>
          </div>
        )}
        {insp && <>
          <div className="table-wrap" style={{ maxHeight: 200 }}>
            <table><thead><tr>{insp.columns.map((c) => <th key={c.name} className="mono">{c.name}<div className="muted" style={{ fontWeight: 400 }}>{c.dtype}</div></th>)}</tr></thead>
              <tbody>{insp.sample.slice(0, 5).map((r, i) => <tr key={i}>{insp.columns.map((c) => <td key={c.name} className="small mono">{r[c.name] == null ? <span className="muted">null</span> : String(r[c.name])}</td>)}</tr>)}</tbody></table>
          </div>
          <div className="grid-3">
            <label className="field"><span className="lbl">Name</span><input value={m.name} onChange={(e) => setM({ ...m, name: e.target.value })} /></label>
            {sel('ts_col', 'Timestamp column', cols, insp.timestamp_candidates.length ? `Detected: ${insp.timestamp_candidates.join(', ')}` : 'Required: when each row happened')}
            {sel('timezone', 'Timestamps without a zone are in', ['UTC', 'local'], 'local = this computer’s time zone')}
            {sel('target', 'Target column (what to predict)', cols, 'Optional — without it the source is monitored but not trainable', '(none)')}
            {m.target && sel('task', 'Task', ['classification', 'regression'])}
            {sel('id_col', 'Row id column', cols, 'Optional — used to detect duplicate rows', '(none)')}
            <label className="field"><span className="lbl">Ignore columns</span><input placeholder="comma separated" value={m.exclude} onChange={(e) => setM({ ...m, exclude: e.target.value })} /></label>
            <label className="field"><span className="lbl">Check health every (min)</span><input value={m.profile_every_minutes} onChange={(e) => setM({ ...m, profile_every_minutes: e.target.value })} /></label>
            <label className="field"><span className="lbl">Recent window (hours)</span><input value={m.recent_hours} onChange={(e) => setM({ ...m, recent_hours: e.target.value })} /><span className="help">compared against the reference window for drift</span></label>
            <label className="field"><span className="lbl">Reference window (days)</span><input value={m.reference_days} onChange={(e) => setM({ ...m, reference_days: e.target.value })} /></label>
          </div>
          <button className="primary" style={{ alignSelf: 'flex-start' }} disabled={busy || !m.ts_col || !m.name} onClick={save}>{busy ? 'Connecting & profiling…' : 'Add source'}</button>
        </>}
      </div>
    </Modal>
  )
}

// ------------------------------------------------------------------ static datasets
function StaticDatasets() {
  const [ds, err, reload] = usePoll(() => api.get<StaticDs[]>('/data/static'), 0)
  const [open, setOpen] = useState<StaticDs>()
  const [removing, setRemoving] = useState<StaticDs>()
  const [ver, setVer] = useState(0)
  const [q, setQ] = useState('')
  const shown = (ds ?? []).filter((d) => !q || `${d.name} ${d.id}`.toLowerCase().includes(q.toLowerCase()))
  const gradeColor: Record<string, string> = { A: 'completed', B: 'queued', C: 'stopped', D: 'failed' }
  return (
    <div className="col" style={{ gap: 12 }}>
      <ErrorBox error={err} />
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <div className="small text-2">Downloaded or uploaded datasets that rarely change. A quality report checks missing values, duplicates, leakage, imbalance and whether the train/test split is representative.</div>
        <input style={{ maxWidth: 260 }} placeholder="Filter…" value={q} onChange={(e) => setQ(e.target.value)} />
      </div>
      <div className="card table-wrap">
        <table>
          <thead><tr><th>Dataset</th><th>Kind</th><th>Quality</th><th>Checked</th><th></th></tr></thead>
          <tbody>{shown.map((d) => (
            <tr key={d.id} className="clickable" onClick={() => setOpen(d)}>
              <td>{d.name}<div className="muted small mono">{d.id}</div></td>
              <td className="small">{d.modality} · {d.task}{d.source !== 'builtin' && <span className="tag" style={{ marginLeft: 6 }}>{d.source === 'upload' ? 'uploaded' : d.source.includes('/optional/') ? 'optional' : 'custom'}</span>}</td>
              <td>{d.report ? <span className={`badge ${gradeColor[d.report.grade]}`}><span className="dot" />grade {d.report.grade} · {d.report.n_issues} issue{d.report.n_issues === 1 ? '' : 's'}</span> : <span className="muted small">not checked</span>}</td>
              <td className="small">{d.report ? fmtTime(d.report.created_at) : ''}</td>
              <td style={{ textAlign: 'right', whiteSpace: 'nowrap' }}><button className="sm">{d.report ? 'View report' : 'Run report'}</button>
                <button className="sm ghost danger" onClick={(e) => { e.stopPropagation(); setRemoving(d) }}>Remove</button></td>
            </tr>))}</tbody>
        </table>
      </div>
      {open && <Modal title={`Data quality · ${open.name}`} onClose={() => { setOpen(undefined); reload() }}><QualityReport datasetId={open.id} /></Modal>}
      {removing && <RemoveDatasetModal id={removing.id} onClose={() => setRemoving(undefined)}
        onDone={() => { setRemoving(undefined); reload(); setVer((v) => v + 1) }} />}
      <RemovedDatasets version={ver} onRestored={reload} />
    </div>
  )
}

// ------------------------------------------------------------------ schedules
export function Schedules({ problemId }: { problemId?: string }) {
  const [rows, err, reload] = usePoll(() => api.get<Schedule[]>('/schedules'), 10000)
  const [actErr, setActErr] = useState<unknown>()
  const act = (p: Promise<unknown>) => p.then(reload).catch(setActErr)
  const shown = (rows ?? []).filter((s) => !problemId || s.template.problem_id === problemId)
  return (
    <div className="col" style={{ gap: 12 }}>
      <ErrorBox error={err ?? actErr} />
      {!problemId && <div className="small text-2">Recurring training runs. Create one from <Link to="/new">New training</Link> → Runs & options → “Repeat on a schedule”.
        For live sources each run trains on a fresh, frozen snapshot of the chosen time window.</div>}
      {!shown.length ? <div className="card empty">No schedules yet.</div> : (
        <div className="card table-wrap">
          <table>
            <thead><tr><th>Schedule</th><th>When</th><th>Next run</th><th>Recent runs</th><th>Registers as</th><th></th></tr></thead>
            <tbody>{shown.map((s) => (
              <tr key={s.id}>
                <td><b>{s.name}</b><div className="small muted mono">{s.template.model_id} on {s.template.problem_id}</div></td>
                <td className="small">{s.cadence_text}{!s.enabled && <span className="tag" style={{ marginLeft: 6 }}>paused</span>}</td>
                <td className="small">{s.enabled && s.next_run_at ? new Date(s.next_run_at * 1000).toLocaleString([], { weekday: 'short', hour: '2-digit', minute: '2-digit', month: 'short', day: 'numeric' }) : '—'}</td>
                <td className="small">{s.runs.slice(0, 4).map((r) => (
                  <div key={r.session_id}><Link to={`/sessions/${r.session_id}`}>{fmtTime(r.created_at)}</Link> <span className={`badge ${r.status}`} style={{ padding: '0 6px' }}><span className="dot" />{r.status}</span></div>))}</td>
                <td className="small mono">{s.register_as ?? '—'}</td>
                <td style={{ textAlign: 'right', whiteSpace: 'nowrap' }}>
                  <button className="sm" onClick={() => act(api.post(`/schedules/${s.id}/run`))}>Run now</button>
                  <button className="sm ghost" onClick={() => act(api.patch(`/schedules/${s.id}`, { enabled: !s.enabled }))}>{s.enabled ? 'Pause' : 'Resume'}</button>
                  <ConfirmButton className="sm ghost danger" onConfirm={() => act(api.del(`/schedules/${s.id}`))}>Delete</ConfirmButton>
                </td>
              </tr>))}</tbody>
          </table>
        </div>
      )}
    </div>
  )
}
