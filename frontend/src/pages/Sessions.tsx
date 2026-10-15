import { useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api, fmtDuration, fmtTime, type SessionBrief } from '../api'
import { usePlugins } from '../App'
import { ErrorBox, HeadlineValue, Progress, StatusBadge, usePoll } from '../components/ui'
import { QueuePanel } from '../components/QueuePanel'
import { ExportModal } from '../components/ExportModal'

const FILTERS = ['all', 'active', 'completed', 'failed', 'stopped'] as const

export function usePluginNames() {
  const { plugins } = usePlugins()
  return useMemo(() => {
    const m: Record<string, string> = {}
    for (const p of [...(plugins?.models ?? []), ...(plugins?.datasets ?? []), ...(plugins?.envs ?? [])]) m[p.id] = p.name
    return (id: string) => m[id] ?? id
  }, [plugins])
}

export default function Sessions() {
  const [sessions, err] = usePoll(() => api.get<SessionBrief[]>('/sessions'), 3000)
  const [filter, setFilter] = useState<(typeof FILTERS)[number]>('all')
  const [q, setQ] = useState('')
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [exporting, setExporting] = useState(false)
  const nav = useNavigate()
  const name = usePluginNames()

  const shown = useMemo(() => (sessions ?? []).filter((s) => {
    if (filter === 'active' && !['running', 'queued', 'stopping', 'scheduled'].includes(s.status)) return false
    if (filter === 'failed' && !['failed', 'partial'].includes(s.status)) return false
    if ((filter === 'completed' || filter === 'stopped') && s.status !== filter) return false
    if (q) {
      const hay = `${s.name} ${s.model_id} ${s.problem_id} ${name(s.model_id)} ${name(s.problem_id)} ${s.tags.join(' ')}`.toLowerCase()
      if (!hay.includes(q.toLowerCase())) return false
    }
    return true
  }), [sessions, filter, q, name])

  const active = (sessions ?? []).filter((s) => ['running', 'queued'].includes(s.status)).length
  const toggle = (id: string) => setSelected((s) => {
    const n = new Set(s)
    if (n.has(id)) n.delete(id)
    else n.add(id)
    return n
  })

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>Training sessions</h1>
          <div className="sub">{sessions ? `${sessions.length} sessions · ${active} active` : 'Loading…'}</div>
        </div>
        <div className="row">
          {selected.size >= 1 && (
            <>
              <button onClick={() => nav(`/compare?ids=${[...selected].join(',')}`)}>Compare {selected.size}</button>
              <button onClick={() => setExporting(true)}>Export {selected.size} to GitHub…</button>
            </>
          )}
          <Link className="btn ghost" to="/exports">Exports</Link>
          <Link className="btn primary" to="/new">+ New training</Link>
        </div>
      </div>
      <ErrorBox error={err} />
      {exporting && <ExportModal sessionIds={[...selected]} onClose={() => setExporting(false)} />}
      <QueuePanel />
      <div className="row" style={{ marginBottom: 12 }}>
        <div className="segmented">
          {FILTERS.map((f) => <button key={f} className={filter === f ? 'on' : ''} onClick={() => setFilter(f)}>{f[0].toUpperCase() + f.slice(1)}</button>)}
        </div>
        <input style={{ maxWidth: 280 }} placeholder="Search name, model, dataset, tag…" value={q} onChange={(e) => setQ(e.target.value)} />
      </div>
      <div className="card table-wrap">
        {sessions && !shown.length ? (
          <div className="empty">
            {sessions.length ? 'No sessions match the filter.' : <>No training sessions yet. <Link to="/new">Start one</Link>.</>}
          </div>
        ) : (
          <table>
            <thead>
              <tr>
                <th style={{ width: 30 }} aria-label="select" />
                <th>Session</th><th>Status</th><th className="hide-sm">Model</th><th className="hide-sm">Problem</th><th className="num hide-sm">Seeds</th>
                <th className="hide-sm" style={{ minWidth: 110 }}>Progress</th><th>Result</th><th className="hide-sm">Started</th><th className="num hide-sm">Duration</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((s) => (
                <tr key={s.id} className="clickable" onClick={() => nav(`/sessions/${s.id}`)}>
                  <td onClick={(e) => e.stopPropagation()}>
                    <input type="checkbox" checked={selected.has(s.id)} onChange={() => toggle(s.id)} aria-label={`select ${s.name}`} />
                  </td>
                  <td>
                    <div style={{ fontWeight: 550 }}>{s.name}</div>
                    <div className="chips" style={{ marginTop: 3 }}>
                      {s.mode === 'evaluate' && <span className="tag" title="Evaluation of a saved model, no training">eval</span>}
                      {s.parent_id && <span className="tag" title="Re-trained / fine-tuned from another session">↳ retrain</span>}
                      {s.group_id && (
                        <Link className="tag" to={`/compare?group=${s.group_id}`} onClick={(e) => e.stopPropagation()} title="Open the group comparison">
                          {s.group_kind}{Object.keys(s.group_params ?? {}).length ? ': ' + Object.entries(s.group_params).map(([k, v]) => `${k.split('.').pop()}=${v}`).join(', ') : ''}
                        </Link>
                      )}
                      {s.tags.map((t) => <span key={t} className="tag">{t}</span>)}
                    </div>
                    <div className="show-sm muted small">{name(s.model_id)} · {name(s.problem_id)}</div>
                  </td>
                  <td><StatusBadge status={s.status} /></td>
                  <td className="hide-sm">{name(s.model_id)}<div className="muted small">{s.framework} · {s.device}</div></td>
                  <td className="hide-sm">{name(s.problem_id)}<div className="muted small">{s.task}</div></td>
                  <td className="num hide-sm">{s.n_completed}/{s.n_seeds}</td>
                  <td className="hide-sm">
                    {['running', 'queued', 'stopping', 'scheduled'].includes(s.status) ? (
                      <div className="col" style={{ gap: 3 }}>
                        <Progress value={s.progress} />
                        <span className="muted small">{s.progress_text || s.status}</span>
                      </div>
                    ) : <span className="muted small">{Math.round(s.progress * 100)}%</span>}
                  </td>
                  <td><HeadlineValue h={s.headline} /></td>
                  <td className="small hide-sm" style={{ whiteSpace: 'nowrap' }}>{fmtTime(s.started_at ?? s.created_at)}</td>
                  <td className="num small hide-sm">{s.started_at ? fmtDuration((s.ended_at ?? Date.now() / 1000) - s.started_at) : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
