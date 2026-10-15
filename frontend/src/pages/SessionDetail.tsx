import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import {
  api, fmtBytes, fmtDuration, fmtMetric, fmtNum, fmtTime, fmtValue, wsUrl,
  type Artifact, type LineageNode, type Run, type SeriesByRun, type SessionDetail as Detail,
} from '../api'
import { LiveView, Snapshots } from '../components/Snapshots'
import { ExportModal } from '../components/ExportModal'
import { GeneratePanel } from '../components/Generate'
import { Beeswarm, GradCamGallery, ImportanceChart, RecordWaterfall, type GradCam, type ShapGlobal } from '../components/Explain'
import { Heatmap, LineChart, linesForKeys, useTheme, baseLayout, Plot } from '../components/Chart'
import { ConfirmButton, ErrorBox, HeadlineValue, Modal, Progress, StatusBadge, usePoll } from '../components/ui'
import type { RegisteredModel } from '../mlops'
import { allKeys, groupMetricKeys, mergeSeries, type MetricGroup } from '../metrics'
import { usePluginNames } from './Sessions'

const X_TITLE: Record<string, string> = { torch: 'epoch', sb3: 'timestep', hf: 'step', sklearn: 'iteration', xgboost: 'round', lightgbm: 'round' }
const TABS = ['Overview', 'Charts', 'Snapshots', 'Generate', 'Explain', 'Runs & logs', 'Models', 'Config', 'Lineage', 'Artifacts'] as const

/** Live metric stream over WebSocket, with polling fallback. */
function useLiveSession(id: string) {
  const [series, setSeries] = useState<SeriesByRun>({})
  const [runs, setRuns] = useState<Run[]>([])
  const [status, setStatus] = useState<string>()
  const [mediaVersion, setMediaVersion] = useState(0)
  const lastId = useRef(0)

  useEffect(() => {
    setSeries({}); setRuns([]); lastId.current = 0
    let ws: WebSocket | null = null
    let poll: ReturnType<typeof setInterval> | undefined
    let closed = false
    const apply = (m: { series: SeriesByRun; last_id: number; runs?: Run[]; status?: string; media_last_id?: number }) => {
      if (m.media_last_id) setMediaVersion(m.media_last_id)
      if (m.last_id > lastId.current) {
        setSeries((s) => mergeSeries(s, m.series))
        lastId.current = m.last_id
      }
      if (m.runs) setRuns(m.runs)
      if (m.status) setStatus(m.status)
    }
    const startPolling = () => {
      if (poll || closed) return
      const tick = () => api.get<{ series: SeriesByRun; last_id: number }>(`/sessions/${id}/metrics?aggregate=false&since_id=${lastId.current}`).then(apply).catch(() => {})
      tick()
      poll = setInterval(tick, 3000)
    }
    try {
      ws = new WebSocket(wsUrl(`/ws/sessions/${id}?since_id=0`))
      ws.onmessage = (e) => {
        const m = JSON.parse(e.data)
        if (m.type === 'update') apply(m)
      }
      ws.onerror = startPolling
      ws.onclose = (e) => { if (!closed && e.code !== 1000) startPolling() }
    } catch {
      startPolling()
    }
    return () => { closed = true; ws?.close(); if (poll) clearInterval(poll) }
  }, [id])
  return { series, runs, status, mediaVersion }
}

export default function SessionDetail() {
  const { id = '' } = useParams()
  const nav = useNavigate()
  const live = useLiveSession(id)
  const [detail, err, reload] = usePoll(() => api.get<Detail>(`/sessions/${id}`), 0, [id])
  const [tab, setTab] = useState<(typeof TABS)[number]>('Overview')
  const [actionErr, setActionErr] = useState<unknown>()
  const [registering, setRegistering] = useState(false)
  const [exporting, setExporting] = useState(false)
  const name = usePluginNames()

  // refresh summary data whenever a run changes status
  const runSig = live.runs.map((r) => r.status).join()
  useEffect(() => { if (runSig) reload() }, [runSig]) // eslint-disable-line react-hooks/exhaustive-deps

  const runs: Run[] = useMemo(() => {
    if (!detail) return live.runs
    const byId = Object.fromEntries(live.runs.map((r) => [r.id, r]))
    return detail.runs.map((r) => ({ ...r, ...(byId[r.id] ?? {}) }))
  }, [detail, live.runs])
  const seedOf = useMemo(() => Object.fromEntries(runs.map((r) => [r.id, r.seed])), [runs])

  if (err) return <ErrorBox error={err} />
  if (!detail) return <div className="muted">Loading…</div>
  const status = live.status ?? detail.status
  const active = ['running', 'queued', 'stopping'].includes(status)
  const hasModels = detail.runs.some((r) => r.storage?.has_model)
  const canResume = !active && !!detail.plugins.model?.supports_resume &&
    runs.some((r) => ['failed', 'stopped'].includes(r.status) && detail.runs.find((d) => d.id === r.id)?.storage?.has_checkpoint)
  const act = async (fn: () => Promise<unknown>) => {
    setActionErr(undefined)
    try { await fn(); reload() } catch (e) { setActionErr(e) }
  }

  return (
    <div>
      <div className="page-head">
        <div style={{ minWidth: 0 }}>
          <div className="row" style={{ gap: 10 }}>
            <EditableTitle value={detail.name} onSave={(n) => act(() => api.patch(`/sessions/${id}`, { name: n }))} />
            <StatusBadge status={status} />
            {detail.mode === 'evaluate' && <span className="tag">evaluation only</span>}
            {detail.group_id && <Link className="tag" to={`/compare?group=${detail.group_id}`}>{detail.group_kind} group →</Link>}
          </div>
          <div className="sub">
            {name(detail.model_id)} on {name(detail.problem_id)} · {detail.task} · {detail.device} · created {fmtTime(detail.created_at)}
            {detail.parent && <> · {detail.source?.id === detail.parent.id ? 'fine-tuned' : 'retrained'} from <Link to={`/sessions/${detail.parent.id}`}>{detail.parent.name}</Link></>}
            {detail.mode === 'evaluate' && detail.source && <> · evaluates the model from <Link to={`/sessions/${detail.source.id}`}>{detail.source.name}</Link></>}
            {Object.keys(detail.group_params ?? {}).length > 0 && <> · {Object.entries(detail.group_params).map(([k, v]) => `${k.split('.').pop()}=${fmtValue(v)}`).join(', ')}</>}
          </div>
        </div>
        <div className="row">
          <button className="primary" onClick={() => nav(`/new?from=${id}`)}>↻ Train again</button>
          {hasModels && detail.plugins.model?.supports_init && detail.mode === 'train' && <button onClick={() => nav(`/new?init=${id}`)}>Fine-tune</button>}
          {hasModels && detail.plugins.model?.supports_eval && <button onClick={() => nav(`/new?eval=${id}`)}>Evaluate</button>}
          {hasModels && detail.mode === 'train' && <button onClick={() => setRegistering(true)}>Register</button>}
          {!active && <button onClick={() => setExporting(true)} title="Export as a standalone GitHub repository">Export</button>}
          {active && <button className="danger" onClick={() => act(() => api.post(`/sessions/${id}/stop`))}>■ Stop</button>}
          {canResume && <button onClick={() => act(() => api.post(`/sessions/${id}/resume`))}>▶ Resume</button>}
          {!active && runs.some((r) => ['failed', 'stopped'].includes(r.status)) && (
            <button onClick={() => act(() => api.post(`/sessions/${id}/retry`))}>Restart failed seeds</button>
          )}
          <a className="btn" href={`/api/sessions/${id}/export`} target="_blank" rel="noreferrer">JSON</a>
          <ConfirmButton className="danger" disabled={active} onConfirm={() => act(async () => { await api.del(`/sessions/${id}`); nav('/') })}>Delete</ConfirmButton>
        </div>
      </div>
      <ErrorBox error={actionErr} />

      {active && (
        <div className="card card-pad" style={{ marginBottom: 16 }}>
          <div className="col" style={{ gap: 8 }}>
            {runs.map((r) => (
              <div key={r.id} className="row" style={{ flexWrap: 'nowrap' }}>
                <span className="small num" style={{ width: 60 }}>seed {r.seed}</span>
                <div className="grow"><Progress value={r.progress ?? 0} /></div>
                <span className="small muted" style={{ width: 210, textAlign: 'right' }}>{r.status === 'running' ? r.progress_text : r.status}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="tabs" role="tablist">
        {TABS.filter((t) => t === 'Generate' ? detail.task === 'generation' : t === 'Explain' ? detail.task !== 'generation' : true).map((t) => <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? 'on' : ''} onClick={() => setTab(t)}>{t}</button>)}
      </div>

      {registering && <RegisterModal detail={detail} onClose={() => setRegistering(false)} />}
      {exporting && <ExportModal sessionIds={[detail.id]} onClose={() => setExporting(false)} />}
      {tab === 'Overview' && <Overview detail={detail} series={live.series} seedOf={seedOf} runs={runs} />}
      {tab === 'Charts' && <ChartsGrid series={live.series} seedOf={seedOf} framework={detail.framework} primaryOnly={false} />}
      {tab === 'Snapshots' && <Snapshots sessionId={id} version={live.mediaVersion} stepLabel={detail.mode === 'evaluate' ? 'step' : X_TITLE[detail.framework] ?? 'step'} />}
      {tab === 'Generate' && <GeneratePanel detail={detail} />}
      {tab === 'Explain' && <ExplainTab detail={detail} />}
      {tab === 'Runs & logs' && <RunsTab runs={runs} />}
      {tab === 'Models' && <ModelsTab detail={detail} onChange={reload} />}
      {tab === 'Config' && <ConfigTab detail={detail} onSaved={reload} />}
      {tab === 'Lineage' && <LineageTab id={id} detail={detail} />}
      {tab === 'Artifacts' && <ArtifactsTab id={id} runSig={runSig} />}
    </div>
  )
}

function EditableTitle({ value, onSave }: { value: string; onSave: (v: string) => void }) {
  const [editing, setEditing] = useState(false)
  const [v, setV] = useState(value)
  useEffect(() => setV(value), [value])
  if (!editing) return <h1 style={{ cursor: 'text' }} title="Click to rename" onClick={() => setEditing(true)}>{value}</h1>
  return (
    <input autoFocus value={v} style={{ fontSize: '1.3rem', fontWeight: 600, maxWidth: 520 }} onChange={(e) => setV(e.target.value)}
      onBlur={() => { setEditing(false); if (v.trim() && v !== value) onSave(v.trim()) }}
      onKeyDown={(e) => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur(); if (e.key === 'Escape') { setV(value); setEditing(false) } }} />
  )
}

// ------------------------------------------------------------------ overview
function Overview({ detail, series, seedOf, runs }: { detail: Detail; series: SeriesByRun; seedOf: Record<string, number>; runs: Run[] }) {
  const st = detail.summary_stats
  const tiles: { k: string; label: string }[] = []
  const pick = (k: string, label: string) => st[k] && tiles.push({ k, label })
  pick('epochs_trained', 'Epochs trained'); pick('best_epoch', 'Best epoch'); pick('timesteps_trained', 'Timesteps')
  pick('steps_trained', 'Optimizer steps'); pick('n_params', 'Parameters')
  pick('train_time_s', 'Train time / seed'); pick('fit_time_s', 'Fit time / seed')
  const failed = runs.filter((r) => r.status === 'failed')

  return (
    <div className="col" style={{ gap: 18 }}>
      {detail.problem_kind === 'env' && <LiveView runs={runs} />}
      {failed.length > 0 && (
        <div className="error-box">{failed.length} seed{failed.length > 1 ? 's' : ''} failed. Last error (seed {failed[0].seed}):{'\n'}{(failed[0].error ?? '').split('\n').slice(-6).join('\n')}</div>
      )}
      <div className="tiles">
        <div className="card tile">
          <div className="k">{detail.headline?.key ?? 'Result'}</div>
          <div className="v num">{detail.headline ? fmtMetric(detail.headline.key, detail.headline.mean) : '—'}</div>
          <div className="s">{detail.headline && detail.headline.n > 1 ? `± ${fmtMetric(detail.headline.key, detail.headline.std)} over ${detail.headline.n} seeds` : detail.headline ? '1 seed' : 'waiting for completed runs'}</div>
        </div>
        {tiles.map((t) => (
          <div className="card tile" key={t.k}>
            <div className="k">{t.label}</div>
            <div className="v num">{fmtMetric(t.k, st[t.k].mean)}</div>
            <div className="s">{st[t.k].n > 1 ? `range ${fmtMetric(t.k, st[t.k].min)} – ${fmtMetric(t.k, st[t.k].max)}` : ' '}</div>
          </div>
        ))}
        <div className="card tile">
          <div className="k">Seeds</div>
          <div className="v num">{detail.n_completed}/{detail.n_seeds}</div>
          <div className="s">completed</div>
        </div>
      </div>
      <ChartsGrid series={series} seedOf={seedOf} framework={detail.framework} primaryOnly />
      <Components detail={detail} />
      <FinalMetricsTable detail={detail} />
      <BestVsFinal detail={detail} />
    </div>
  )
}

/** Which interchangeable parts this model was built from (and which models reuse its parts). */
function Components({ detail }: { detail: Detail }) {
  const slots = (detail.plugins.model?.params ?? []).filter((p) => p.type === 'choice' && p.option_groups)
  const users = detail.component_users ?? []
  if (!slots.length && !users.length) return null
  return (
    <div className="card">
      <div className="card-head"><h2>Components</h2><span className="muted small">interchangeable parts of this model</span></div>
      {slots.length > 0 && (
        <table><tbody>{slots.map((p) => {
          const v = String(detail.model_params[p.name] ?? p.default)
          const frozen = v !== 'builtin' && !!detail.model_params[`${p.name}_freeze`]
          const sid = v.startsWith('session:') ? v.split(':')[1] : null
          return (
            <tr key={p.name}>
              <td style={{ width: 160 }}><b>{p.name}</b></td>
              <td>{sid ? <Link to={`/sessions/${sid}`}>{p.labels?.[v] ?? v}</Link> : (p.labels?.[v] ?? v)}
                {p.labels?.[v] === undefined && v.startsWith('session:') && <span className="muted small"> (source session deleted)</span>}</td>
              <td className="small">{p.option_groups?.[v] && <span className="tag">{p.option_groups[v]}</span>} {frozen && <span className="tag">frozen</span>}</td>
            </tr>
          )
        })}</tbody></table>
      )}
      {users.length > 0 && (
        <div className="card-pad small" style={{ paddingTop: slots.length ? 8 : undefined }}>
          Parts of this model are reused by: {users.map((u, i) => <span key={u.id}>{i > 0 && ', '}<Link to={`/sessions/${u.id}`}>{u.name}</Link></span>)}
        </div>
      )}
    </div>
  )
}

function FinalMetricsTable({ detail }: { detail: Detail }) {
  const st = detail.summary_stats
  const keys = Object.keys(st).filter((k) => !k.startsWith('final_model/')).sort((a, b) => {
    const r = (k: string) => (k.startsWith('test/') ? 0 : k.startsWith('final/') ? 0 : k.startsWith('val/') ? 1 : k.includes('/') ? 2 : 3)
    return r(a) - r(b) || a.localeCompare(b)
  })
  if (!keys.length) return null
  const seeds = detail.runs.filter((r) => r.status === 'completed').map((r) => r.seed)
  return (
    <div className="card">
      <div className="card-head"><h2>Final metrics</h2><span className="muted small">mean ± std across completed seeds</span></div>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Metric</th><th className="num">Mean</th><th className="num">Std</th><th className="num">Min</th><th className="num">Max</th>
            {seeds.length > 1 && seeds.map((s) => <th key={s} className="num">seed {s}</th>)}</tr></thead>
          <tbody>
            {keys.map((k) => (
              <tr key={k}>
                <td className="mono">{k}</td>
                <td className="num" style={{ fontWeight: 600 }}>{fmtMetric(k, st[k].mean)}</td>
                <td className="num">{st[k].n > 1 ? fmtMetric(k, st[k].std) : '—'}</td>
                <td className="num">{fmtMetric(k, st[k].min)}</td>
                <td className="num">{fmtMetric(k, st[k].max)}</td>
                {seeds.length > 1 && seeds.map((s) => <td key={s} className="num text-2">{fmtMetric(k, st[k].per_seed[String(s)])}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ charts
export function ChartControls({ showSeeds, setShowSeeds, smoothing, setSmoothing, multi }: {
  showSeeds: boolean; setShowSeeds: (b: boolean) => void; smoothing: number; setSmoothing: (n: number) => void; multi: boolean
}) {
  return (
    <div className="row small text-2" style={{ gap: 18 }}>
      {multi && <label className="check"><input type="checkbox" checked={showSeeds} onChange={(e) => setShowSeeds(e.target.checked)} /> Show individual seeds</label>}
      <label className="row" style={{ gap: 8 }}>Smoothing
        <input type="range" min={0} max={0.99} step={0.01} value={smoothing} onChange={(e) => setSmoothing(Number(e.target.value))} style={{ width: 120 }} />
        <span className="num" style={{ width: 30 }}>{smoothing.toFixed(2)}</span>
      </label>
    </div>
  )
}

function ChartsGrid({ series, seedOf, framework, primaryOnly }: { series: SeriesByRun; seedOf: Record<string, number>; framework: string; primaryOnly: boolean }) {
  const theme = useTheme()
  const [showSeeds, setShowSeeds] = useState(false)
  const [smoothing, setSmoothing] = useState(framework === 'sb3' ? 0.6 : 0)
  const [logY, setLogY] = useState<Record<string, boolean>>({})
  const groups = useMemo(() => groupMetricKeys(allKeys(series)), [series])
  // a metric logged once (sklearn final scores at step 0) makes a dull chart; show those in the table instead
  const plottable = groups.filter((g) => g.keys.some((k) => Object.values(series).some((s) => (s[k]?.steps.length ?? 0) > 1)))
  const shown = primaryOnly ? plottable.filter((g) => g.primary).slice(0, 6) : plottable
  const others = primaryOnly ? [] : shown.filter((g) => !g.primary)
  const main = primaryOnly ? shown : shown.filter((g) => g.primary)
  const multi = Object.keys(series).length > 1
  if (!groups.length) return <div className="card empty">Waiting for the first metrics…</div>
  if (!shown.length) return primaryOnly ? null : <div className="card empty">No curves to plot (this model reports only final metrics).</div>

  const renderGroup = (g: MetricGroup) => (
    <div className="card chart-card" key={g.name}>
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <h3>{g.name}</h3>
        <button className="ghost sm" onClick={() => setLogY((l) => ({ ...l, [g.name]: !l[g.name] }))}>{logY[g.name] ? 'linear' : 'log'} y</button>
      </div>
      <LineChart lines={linesForKeys(g.keys, series, seedOf, theme)} xTitle={X_TITLE[framework] ?? 'step'} showSeeds={showSeeds} smoothing={smoothing} logY={logY[g.name]} />
    </div>
  )
  return (
    <div className="col" style={{ gap: 12 }}>
      <ChartControls showSeeds={showSeeds} setShowSeeds={setShowSeeds} smoothing={smoothing} setSmoothing={setSmoothing} multi={multi} />
      <div className="grid-2">{main.map(renderGroup)}</div>
      {others.length > 0 && (
        <details open={!main.length}>
          <summary>{others.length} more metric{others.length > 1 ? 's' : ''}</summary>
          <div className="grid-2" style={{ marginTop: 8 }}>{others.map(renderGroup)}</div>
        </details>
      )}
    </div>
  )
}

// ------------------------------------------------------------------ runs / logs
function RunsTab({ runs }: { runs: Run[] }) {
  const [sel, setSel] = useState<string | undefined>(runs[0]?.id)
  const run = runs.find((r) => r.id === sel)
  const [log] = usePoll(() => (sel ? api.get<string>(`/runs/${sel}/log?tail=500`) : Promise.resolve('')), run?.status === 'running' ? 3000 : 0, [sel, run?.status])
  const [files] = usePoll(() => (sel ? api.get<{ name: string; size: number }[]>(`/runs/${sel}/files`) : Promise.resolve([])), 0, [sel, run?.status])
  const logRef = useRef<HTMLPreElement>(null)
  useEffect(() => { if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight }, [log])

  return (
    <div className="col" style={{ gap: 14 }}>
      <div className="card table-wrap">
        <table>
          <thead><tr><th>Seed</th><th>Status</th><th>Progress</th><th>Started</th><th className="num">Duration</th><th>Error</th></tr></thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.id} className="clickable" onClick={() => setSel(r.id)} style={r.id === sel ? { background: 'var(--accent-soft)' } : undefined}>
                <td className="num">{r.seed}</td>
                <td><StatusBadge status={r.status} /></td>
                <td style={{ minWidth: 140 }}><Progress value={r.progress ?? 0} /><span className="muted small">{r.progress_text}</span></td>
                <td className="small">{fmtTime(r.started_at)}</td>
                <td className="num small">{r.started_at ? fmtDuration((r.ended_at ?? Date.now() / 1000) - r.started_at) : '—'}</td>
                <td className="small err-text" style={{ maxWidth: 380 }}>{r.error ? r.error.trim().split('\n').pop() : ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {run && (
        <div className="grid-2" style={{ gridTemplateColumns: 'minmax(0,3fr) minmax(240px,1fr)' }}>
          <div className="card">
            <div className="card-head"><h3>Log · seed {run.seed}</h3>{run.status === 'running' && <span className="muted small">live</span>}</div>
            <div className="card-pad"><pre className="log" ref={logRef}>{log || 'No output yet.'}</pre></div>
          </div>
          <div className="card">
            <div className="card-head"><h3>Files</h3></div>
            <div className="card-pad col" style={{ gap: 6 }}>
              {(files ?? []).length === 0 && <span className="muted small">No files.</span>}
              {(files ?? []).map((f) => (
                <a key={f.name} href={`/api/runs/${run.id}/files/${f.name}`} className="small row" style={{ justifyContent: 'space-between' }}>
                  <span className="mono">{f.name}</span><span className="muted">{(f.size / 1024).toFixed(0)} KB</span>
                </a>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// ------------------------------------------------------------------ config
function ParamTable({ title, values, changed }: { title: string; values: Record<string, unknown>; changed: Record<string, unknown> }) {
  return (
    <div className="card">
      <div className="card-head"><h3>{title}</h3></div>
      <div className="table-wrap">
        <table>
          <tbody>
            {Object.entries(values).map(([k, v]) => (
              <tr key={k}>
                <td className="mono" style={{ width: '40%' }}>{k}</td>
                <td className="mono">
                  {k in changed ? <><span className="diff-from">{fmtValue(changed[k])}</span> <span className="diff-to">{fmtValue(v)}</span></> : fmtValue(v)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function ConfigTab({ detail, onSaved }: { detail: Detail; onSaved: () => void }) {
  const [notes, setNotes] = useState(detail.notes)
  const [tags, setTags] = useState(detail.tags.join(', '))
  const [saved, setSaved] = useState(false)
  const changed = (prefix: string) => Object.fromEntries(detail.changes_from_parent.filter((c) => c.param.startsWith(prefix)).map((c) => [c.param.slice(prefix.length), c.from]))
  const save = async () => {
    await api.patch(`/sessions/${detail.id}`, { notes, tags: tags.split(',').map((t) => t.trim()).filter(Boolean) })
    setSaved(true); onSaved(); setTimeout(() => setSaved(false), 1500)
  }
  return (
    <div className="col" style={{ gap: 14 }}>
      {detail.parent && (
        <div className="card card-pad">
          <h3 style={{ marginBottom: 8 }}>Changes from <Link to={`/sessions/${detail.parent.id}`}>{detail.parent.name}</Link></h3>
          {detail.changes_from_parent.length === 0 ? <span className="muted">Identical configuration.</span> : (
            <div className="kv mono small">
              {detail.changes_from_parent.map((c) => (
                <div key={c.param} style={{ display: 'contents' }}>
                  <div>{c.param}</div>
                  <div><span className="diff-from">{fmtValue(c.from)}</span> → <span className="diff-to">{fmtValue(c.to)}</span></div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
      <div className="grid-2">
        <ParamTable title={`Problem · ${detail.problem_id}`} values={detail.problem_params} changed={changed('problem.')} />
        <ParamTable title={`Model · ${detail.model_id}`} values={detail.model_params} changed={changed('model.')} />
        <ParamTable title={`Session options · ${detail.mode}`} values={detail.options} changed={{}} />
      </div>
      <div className="grid-2">
        <div className="card card-pad col">
          <h3>Notes & tags</h3>
          <textarea rows={4} value={notes} onChange={(e) => setNotes(e.target.value)} placeholder="Observations, conclusions…" />
          <input value={tags} onChange={(e) => setTags(e.target.value)} placeholder="tags, comma separated" />
          <div className="row"><button onClick={save}>Save</button>{saved && <span className="muted small">Saved</span>}</div>
        </div>
        <div className="card card-pad col">
          <h3>Reproducibility</h3>
          <div className="kv small">
            <div>Seeds</div><div className="mono">{detail.seeds.join(', ')}</div>
            <div>Device</div><div className="mono">{detail.device}</div>
            {Object.entries(detail.env_info.versions ?? {}).map(([k, v]) => <div key={k} style={{ display: 'contents' }}><div>{k}</div><div className="mono">{v}</div></div>)}
            {Object.entries(detail.env_info.plugin_hashes ?? {}).map(([k, v]) => <div key={k} style={{ display: 'contents' }}><div className="mono small">{k.split('/').pop()}</div><div className="mono">sha256:{v}</div></div>)}
          </div>
        </div>
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ lineage
function LineageTab({ id, detail }: { id: string; detail?: Detail }) {
  const theme = useTheme()
  const [lin, err] = usePoll(() => api.get<{ root: string; current: string; nodes: LineageNode[] }>(`/sessions/${id}/lineage`), 0, [id])
  const depth = useCallback((n: LineageNode, nodes: LineageNode[]): number => {
    let d = 0, cur: LineageNode | undefined = n
    while (cur?.parent_id && d < 50) { cur = nodes.find((x) => x.id === cur!.parent_id); d++ }
    return d
  }, [])
  if (err) return <ErrorBox error={err} />
  if (!lin) return <div className="muted">Loading…</div>
  // depth-first order so children sit under their parent
  const ordered: LineageNode[] = []
  const visit = (pid: string | null) => lin.nodes.filter((n) => n.parent_id === pid || (pid === null && n.id === lin.root)).forEach((n) => {
    if (ordered.includes(n)) return
    ordered.push(n); visit(n.id)
  })
  visit(null)
  const withMetric = lin.nodes.filter((n) => n.headline)
  const key = withMetric[0]?.headline?.key

  return (
    <div className="col" style={{ gap: 14 }}>
      {detail && detail.evaluations.length > 0 && (
        <div className="card card-pad">
          <h3 style={{ marginBottom: 8 }}>This session's saved models are used by</h3>
          <div className="col" style={{ gap: 4 }}>
            {detail.evaluations.map((e) => <div key={e.id} className="row small"><Link to={`/sessions/${e.id}`}>{e.name}</Link><span className="muted">{fmtTime(e.created_at)}</span></div>)}
          </div>
        </div>
      )}
      {lin.nodes.length === 1 && <div className="notice">This session has no ancestors or descendants yet. Use <b>Train again</b> to branch from it; every retrain is linked here.</div>}
      {withMetric.length > 1 && key && (
        <div className="card chart-card">
          <h3>{key} across the lineage (chronological)</h3>
          <div style={{ height: 280 }}><Plot useResizeHandler style={{ width: '100%', height: '100%' }} config={{ displaylogo: false, responsive: true }}
            data={[{
              x: withMetric.map((n) => n.name), y: withMetric.map((n) => n.headline!.mean),
              error_y: { type: 'data', array: withMetric.map((n) => n.headline!.std), visible: true, color: theme.series[0] },
              type: 'scatter', mode: 'lines+markers', line: { color: theme.series[0], width: 2 },
              marker: { size: withMetric.map((n) => (n.id === id ? 13 : 9)), color: theme.series[0], line: { color: theme.surface, width: 2 } },
              hovertemplate: '%{x}<br>%{y:.4g}<extra></extra>',
            }]}
            layout={{ ...baseLayout(theme, '', key), hovermode: 'closest', showlegend: false, xaxis: { ...baseLayout(theme, '').xaxis, tickangle: -20, automargin: true } }} /></div>
        </div>
      )}
      <div className="card card-pad tree">
        {ordered.map((n) => (
          <div key={n.id} className={`tree-node ${n.id === id ? 'current' : ''}`} style={{ marginLeft: depth(n, lin.nodes) * 26 }}>
            <span className="muted" style={{ width: 14 }}>{n.parent_id ? '↳' : '●'}</span>
            <div className="grow col" style={{ gap: 3 }}>
              <div className="row">
                <Link to={`/sessions/${n.id}`} style={{ fontWeight: 600 }}>{n.name}</Link>
                <StatusBadge status={n.status} />
                <HeadlineValue h={n.headline} />
                <span className="muted small">{fmtTime(n.created_at)}</span>
              </div>
              {n.changes.length > 0 && (
                <div className="mono small text-2">
                  {n.changes.map((c) => <span key={c.param} style={{ marginRight: 14 }}>{c.param}: <span className="diff-from">{fmtValue(c.from)}</span> → <span className="diff-to">{fmtValue(c.to)}</span></span>)}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ artifacts
function ArtifactsTab({ id, runSig }: { id: string; runSig: string }) {
  const [arts, err] = usePoll(() => api.get<Artifact[]>(`/sessions/${id}/artifacts`), 0, [id, runSig])
  const [seedSel, setSeedSel] = useState<Record<string, string>>({})
  if (err) return <ErrorBox error={err} />
  if (!arts) return <div className="muted">Loading…</div>
  if (!arts.length) return <div className="card empty">No artifacts yet. Confusion matrices, sample predictions and tables appear here when runs finish.</div>
  const byName = new Map<string, Artifact[]>()
  for (const a of arts) {
    if (!byName.has(a.name)) byName.set(a.name, [])
    byName.get(a.name)!.push(a)
  }
  return (
    <div className="col" style={{ gap: 14 }}>
      {[...byName.entries()].map(([name, list]) => {
        const choice = seedSel[name] ?? (list[0].kind === 'confusion_matrix' && list.length > 1 ? 'sum' : String(list[0].seed))
        const one = list.find((a) => String(a.seed) === choice) ?? list[0]
        let data = one.data
        if (choice === 'sum' && one.kind === 'confusion_matrix') {
          data = { labels: list[0].data.labels, matrix: list[0].data.matrix.map((row: number[], i: number) => row.map((_: number, j: number) => list.reduce((acc, a) => acc + (a.data.matrix[i]?.[j] ?? 0), 0))) }
        }
        return (
          <div className="card" key={name}>
            <div className="card-head">
              <h3>{name.replace(/_/g, ' ')}</h3>
              {list.length > 1 && (
                <select style={{ width: 'auto' }} value={choice} onChange={(e) => setSeedSel((s) => ({ ...s, [name]: e.target.value }))}>
                  {one.kind === 'confusion_matrix' && <option value="sum">all seeds (summed)</option>}
                  {list.map((a) => <option key={a.seed} value={String(a.seed)}>seed {a.seed}</option>)}
                </select>
              )}
            </div>
            <div className="card-pad">
              {one.kind === 'confusion_matrix' ? <Heatmap labels={data.labels} matrix={data.matrix} />
                : one.kind === 'table' ? (
                  <div className="table-wrap" style={{ maxHeight: 480 }}>
                    <table>
                      <thead><tr>{data.columns.map((c: string) => <th key={c}>{c}</th>)}</tr></thead>
                      <tbody>{data.rows.map((r: unknown[], i: number) => <tr key={i}>{r.map((c, j) => <td key={j} className={typeof c === 'number' ? 'num' : ''}>{fmtValue(c)}</td>)}</tr>)}</tbody>
                    </table>
                  </div>
                ) : <pre className="log">{typeof data === 'string' ? data : JSON.stringify(data, null, 2)}</pre>}
            </div>
          </div>
        )
      })}
    </div>
  )
}

// ------------------------------------------------------------------ saved models
function ModelsTab({ detail, onChange }: { detail: Detail; onChange: () => void }) {
  const [err, setErr] = useState<unknown>()
  const nav = useNavigate()
  const saving = detail.options.save_model !== false
  const del = async (path: string) => {
    setErr(undefined)
    try { await api.del(path); onChange() } catch (e) { setErr(e) }
  }
  const total = detail.runs.reduce((a, r) => a + (r.storage?.total_bytes ?? 0), 0)
  const models = detail.runs.reduce((a, r) => a + (r.storage?.model_bytes ?? 0) + (r.storage?.final_bytes ?? 0) + (r.storage?.checkpoint_bytes ?? 0), 0)
  const anyFinal = detail.runs.some((r) => r.storage?.has_final)
  const keep = (detail.options.keep as string) ?? 'both'
  const evalOk = detail.plugins.model?.supports_eval
  return (
    <div className="col" style={{ gap: 14 }}>
      <ErrorBox error={err} />
      {detail.mode === 'evaluate'
        ? <div className="notice">Evaluation sessions don't save models; they evaluate the model from {detail.source ? <Link to={`/sessions/${detail.source.id}`}>{detail.source.name}</Link> : 'a pretrained checkpoint'}.</div>
        : !saving ? <div className="notice">Model saving was turned off for this session.</div>
        : <div className="small text-2"><b>Best</b> = the weights with the best validation score during training (protects against late overfitting);
            <b> Final</b> = the weights after the last epoch/step. {keep === 'final' ? 'This session kept only the final weights (shown as “Model”).'
            : keep === 'best' ? 'This session kept only the best weights.' : 'Tree models (sklearn, XGBoost, LightGBM) save one model — boosting already keeps the best iteration.'}</div>}
      <div className="card">
        <div className="card-head">
          <h3>Saved models & checkpoints</h3>
          <div className="row small">
            <span className="muted">session total on disk {fmtBytes(total)}</span>
            <ConfirmButton className="sm danger" disabled={!models} onConfirm={() => del(`/sessions/${detail.id}/models`)}>Delete all models ({fmtBytes(models)})</ConfirmButton>
          </div>
        </div>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Seed</th><th>Status</th><th className="num">{anyFinal ? 'Best model' : 'Model'}</th>{anyFinal && <th className="num">Final model</th>}<th className="num">Checkpoint</th><th className="num">Snapshots</th><th></th></tr></thead>
            <tbody>
              {detail.runs.map((r) => {
                const s = r.storage
                return (
                  <tr key={r.id}>
                    <td className="num">{r.seed}</td>
                    <td><StatusBadge status={r.status} /></td>
                    <td className="num">{s?.has_model ? <VariantCell bytes={s.model_bytes} onEval={evalOk ? () => nav(`/new?eval=${detail.id}&run=${r.id}`) : undefined}
                      onDelete={s.has_final ? () => del(`/runs/${r.id}/model?variant=best`) : undefined} metaHref={`/api/runs/${r.id}/files/model/meta.json?inline=true`} /> : <span className="muted">—</span>}</td>
                    {anyFinal && <td className="num">{s?.has_final ? <VariantCell bytes={s.final_bytes ?? 0} onEval={evalOk ? () => nav(`/new?eval=${detail.id}&run=${r.id}&variant=final`) : undefined}
                      onDelete={() => del(`/runs/${r.id}/model?variant=final`)} metaHref={`/api/runs/${r.id}/files/model_final/meta.json?inline=true`} /> : <span className="muted">—</span>}</td>}
                    <td className="num">{s?.has_checkpoint ? fmtBytes(s.checkpoint_bytes) : <span className="muted">—</span>}</td>
                    <td className="num">{s?.media_bytes ? fmtBytes(s.media_bytes) : <span className="muted">—</span>}</td>
                    <td style={{ textAlign: 'right' }}>
                      <div className="row" style={{ justifyContent: 'flex-end', gap: 4 }}>
                        {(s?.has_model || s?.has_final || s?.has_checkpoint) && (
                          <ConfirmButton className="sm ghost danger" onConfirm={() => del(`/runs/${r.id}/model`)}>Delete all</ConfirmButton>
                        )}
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
      <div className="small text-2">
        Model files can be downloaded from <b>Runs & logs → Files</b>. A checkpoint is kept only for stopped/failed seeds so they can be resumed;
        deleting the model also removes it. Snapshots (videos etc.) are kept until the session is deleted.
      </div>
    </div>
  )
}

function VariantCell({ bytes, onEval, onDelete, metaHref }: { bytes: number; onEval?: () => void; onDelete?: () => void; metaHref: string }) {
  return (
    <div className="row" style={{ justifyContent: 'flex-end', gap: 2, flexWrap: 'nowrap' }}>
      <span>{fmtBytes(bytes)}</span>
      <a className="btn sm ghost" href={metaHref} target="_blank" rel="noreferrer" title="meta.json">meta</a>
      {onEval && <button className="sm ghost" onClick={onEval}>Evaluate</button>}
      {onDelete && <ConfirmButton className="sm ghost danger" onConfirm={onDelete}>✕</ConfirmButton>}
    </div>
  )
}

/** Best-vs-final comparison for sessions that kept both (final_model/... summary keys). */
function BestVsFinal({ detail }: { detail: Detail }) {
  const rows = detail.runs.filter((r) => r.summary && Object.keys(r.summary).some((k) => k.startsWith('final_model/')))
  if (!rows.length) return null
  const keys = Array.from(new Set(rows.flatMap((r) => Object.keys(r.summary!).filter((k) => k.startsWith('final_model/'))
    .map((k) => k.slice('final_model/'.length)).filter((k) => typeof r.summary![k] === 'number')))).filter((k) => /^(val|test|final)\//.test(k)).slice(0, 6)
  if (!keys.length) return null
  return (
    <div className="card">
      <div className="card-head"><h2>Best vs. final weights</h2><span className="muted small">final weights scored on the same splits — if final is clearly worse, the model overfit after its best epoch</span></div>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Seed</th>{keys.map((k) => <th key={k} className="num mono small">{k}<div className="muted">best → final</div></th>)}</tr></thead>
          <tbody>{rows.map((r) => (
            <tr key={r.id}><td className="num">{r.seed}</td>{keys.map((k) => {
              const b = r.summary![k] as number | undefined, f = r.summary![`final_model/${k}`] as number | undefined
              return <td key={k} className="num small">{fmtNum(b)} → {fmtNum(f)}</td>
            })}</tr>))}</tbody>
        </table>
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ register in the model registry
function RegisterModal({ detail, onClose }: { detail: Detail; onClose: () => void }) {
  const [models] = usePoll(() => api.get<RegisteredModel[]>('/registry'), 0)
  const [name, setName] = useState(detail.name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 40))
  const [run, setRun] = useState('')
  const [variant, setVariant] = useState('best')
  const anyFinal = detail.runs.some((r) => r.storage?.has_final)
  const [desc, setDesc] = useState(detail.notes ?? '')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<unknown>()
  const nav = useNavigate()
  const same = (models ?? []).filter((m) => m.model_id === detail.model_id)
  const submit = async () => {
    setBusy(true); setErr(undefined)
    try {
      await api.post('/registry', { model_name: name, session_id: detail.id, run_id: run || null, description: desc, variant })
      nav('/registry')
    } catch (e) { setErr(e); setBusy(false) }
  }
  return (
    <Modal title="Register in the model registry" onClose={onClose}>
      <div className="col" style={{ gap: 12 }}>
        <div className="small text-2">Creates an immutable copy of the saved model plus a profile of its training data (used for drift monitoring) and a model card.
          Registering under an existing name adds a new version.</div>
        <ErrorBox error={err} />
        <label className="field"><span className="lbl">Registered model name</span>
          <input list="reg-names" value={name} onChange={(e) => setName(e.target.value)} />
          <datalist id="reg-names">{same.map((m) => <option key={m.name} value={m.name} />)}</datalist>
          {same.some((m) => m.name === name) && <span className="help">Will be added as v{(same.find((m) => m.name === name)!.versions[0]?.version ?? 0) + 1}</span>}</label>
        <label className="field"><span className="lbl">Which seed's model</span>
          <select value={run} onChange={(e) => setRun(e.target.value)}>
            <option value="">best seed (by headline metric)</option>
            {detail.runs.filter((r) => variant === 'final' ? r.storage?.has_final : r.storage?.has_model).map((r) => <option key={r.id} value={r.id}>seed {r.seed}</option>)}
          </select></label>
        {anyFinal && <label className="field"><span className="lbl">Weights</span>
          <select value={variant} onChange={(e) => setVariant(e.target.value)}>
            <option value="best">best (by validation)</option><option value="final">final (last epoch)</option>
          </select></label>}
        <label className="field"><span className="lbl">Description</span><textarea value={desc} onChange={(e) => setDesc(e.target.value)} /></label>
        <button className="primary" style={{ alignSelf: 'flex-start' }} disabled={busy || !name.trim()} onClick={submit}>{busy ? 'Registering & profiling…' : 'Register'}</button>
      </div>
    </Modal>
  )
}

// ------------------------------------------------------------------ explainability
function ExplainTab({ detail }: { detail: Detail }) {
  const withModel = detail.runs.filter((r) => r.storage?.has_model)
  const [rid, setRid] = useState(withModel[0]?.id)
  const [variant, setVariant] = useState('best')
  const hasFinal = detail.runs.find((r) => r.id === rid)?.storage?.has_final
  const tabular = detail.modality === 'tabular' && detail.framework !== 'sb3'
  const image = detail.modality === 'image' && detail.framework === 'torch'
  if (!withModel.length) return <div className="card empty">Explanations need a saved model. Turn on “save model” and train again.</div>
  if (!tabular && !image) return <div className="card empty">Explanations are available for tabular models (SHAP) and PyTorch image models (Grad-CAM).</div>
  return (
    <div className="col" style={{ gap: 14 }}>
      {(withModel.length > 1 || hasFinal) && (
        <div className="row small">Explain the model of
          <select style={{ width: 'auto' }} value={rid} onChange={(e) => setRid(e.target.value)}>{withModel.map((r) => <option key={r.id} value={r.id}>seed {r.seed}</option>)}</select>
          {hasFinal && <select style={{ width: 'auto' }} value={variant} onChange={(e) => setVariant(e.target.value)}>
            <option value="best">best weights</option><option value="final">final weights</option></select>}
        </div>
      )}
      {tabular && rid && <ShapPanel path={`/explain/${hasFinal && variant === 'final' ? 'run_final' : 'run'}/${rid}`} />}
      {image && rid && <GradCamPanel path={`/explain/${hasFinal && variant === 'final' ? 'run_final' : 'run'}/${rid}`} />}
    </div>
  )
}

export function ShapPanel({ path }: { path: string }) {
  const [g, setG] = useState<ShapGlobal>()
  const [err, setErr] = useState<unknown>()
  const [busy, setBusy] = useState(false)
  const [sel, setSel] = useState<number>()
  useEffect(() => {
    let alive = true
    setBusy(true); setG(undefined); setErr(undefined); setSel(undefined)
    api.post<ShapGlobal>(`${path}/shap`).then((r) => alive && setG(r)).catch((e) => alive && setErr(e)).finally(() => alive && setBusy(false))
    return () => { alive = false }
  }, [path])
  if (err) return <ErrorBox error={err} />
  if (busy || !g) return <div className="card card-pad muted">Computing SHAP values on held-out data… (first time only; cached afterwards)</div>
  const ex = sel !== undefined ? g.examples[sel] : undefined
  return (
    <div className="col" style={{ gap: 14 }}>
      <div className="small text-2">{g.method}, explaining <b>{g.explains}</b> on {g.n} held-out examples. Average {g.explains}: {g.base_value.toFixed(3)}.</div>
      <div className="grid-2">
        <div className="card chart-card"><h3>Which features matter most</h3><ImportanceChart g={g} /></div>
        <div className="card chart-card"><h3>How feature values push predictions</h3><Beeswarm g={g} /></div>
      </div>
      <div className="grid-2">
        <div className="card">
          <div className="card-head"><h3>Explain a single prediction</h3><span className="small text-2">click a row</span></div>
          <div className="table-wrap" style={{ maxHeight: 420 }}>
            <table>
              <thead><tr><th>#</th><th>Prediction</th><th className="num">{g.explains}</th><th>Actual</th></tr></thead>
              <tbody>{g.examples.map((e, i) => (
                <tr key={i} className="clickable" onClick={() => setSel(i)} style={i === sel ? { background: 'var(--accent-soft)' } : undefined}>
                  <td className="num">{i + 1}</td><td>{fmtValue(e.prediction)}</td><td className="num">{e.output.toFixed(3)}</td>
                  <td style={{ color: String(e.label) === String(e.prediction) ? 'var(--good-text)' : 'var(--critical-text)' }}>{fmtValue(e.label)}</td>
                </tr>))}</tbody>
            </table>
          </div>
        </div>
        <div className="card chart-card">
          <h3>{ex ? `Example #${(sel ?? 0) + 1}` : 'Waterfall'}</h3>
          {ex ? <RecordWaterfall path={`${path}/record`} record={ex.record} /> : <div className="empty small">Pick an example to see how each feature moved its prediction.</div>}
        </div>
      </div>
    </div>
  )
}

function GradCamPanel({ path }: { path: string }) {
  const [g, setG] = useState<GradCam>()
  const [mine, setMine] = useState<GradCam>()
  const [err, setErr] = useState<unknown>()
  useEffect(() => { api.post<GradCam>(`${path}/gradcam`, {}).then(setG).catch(setErr) }, [path])
  const upload = async (files: FileList | null) => {
    if (!files?.length) return
    const imgs = await Promise.all([...files].slice(0, 12).map((f) => new Promise<string>((res) => {
      const r = new FileReader(); r.onload = () => res(String(r.result).split(',')[1]); r.readAsDataURL(f)
    })))
    api.post<GradCam>(`${path}/gradcam`, { images: imgs }).then(setMine).catch(setErr)
  }
  return (
    <div className="col" style={{ gap: 14 }}>
      <ErrorBox error={err} />
      <div className="card card-pad col">
        <div className="row" style={{ justifyContent: 'space-between' }}><h3>Your images</h3>
          <input type="file" accept="image/*" multiple style={{ width: 'auto' }} onChange={(e) => upload(e.target.files)} /></div>
        {mine ? <GradCamGallery g={mine} /> : <div className="small muted">Upload images to see what the model looks at (they're resized to the model's input size).</div>}
      </div>
      <div className="card card-pad col"><h3>Test-set examples</h3>{g ? <GradCamGallery g={g} /> : <div className="muted small">Computing Grad-CAM…</div>}</div>
    </div>
  )
}
