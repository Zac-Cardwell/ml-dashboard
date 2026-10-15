import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { fmtTime } from '../api'
import { ErrorBox } from '../components/ui'
import { type Dataset, studio } from './types'

const KIND: Record<string, string> = { source: 'import', derived: 'recipe', query: 'SQL' }

/** Upstream chain, downstream datasets, and the training sessions that used this dataset. */
export function LineageTab({ ds }: { ds: Dataset }) {
  const [l, setL] = useState<any>()
  const [err, setErr] = useState<unknown>()
  useEffect(() => { studio.lineage(ds.id).then(setL).catch(setErr) }, [ds.id, ds.version])
  if (err) return <ErrorBox error={err} />
  if (!l) return <div className="muted">Loading…</div>
  return (
    <div className="grid-2" style={{ alignItems: 'start' }}>
      <div className="card">
        <div className="card-head"><h3>Data flow</h3></div>
        <div className="card-pad lineage">
          {l.ancestors.map((a: any) => <Node key={a.id} id={a.id} name={a.name} kind={a.kind} sub={`v${a.version}`} />)}
          {l.joins.filter(Boolean).length > 0 && <div className="small muted lineage-join">+ joined with {l.joins.filter(Boolean).length} other dataset(s)</div>}
          <Node id={ds.id} name={ds.name} kind={ds.kind} sub={`v${ds.version} · this dataset`} current />
          {l.descendants.map((d: any) => <Node key={d.id} id={d.id} name={d.name} kind={d.kind} depth={d.depth} />)}
          {l.descendants.length === 0 && <div className="small muted" style={{ marginTop: 8 }}>No datasets are built from this one yet.</div>}
        </div>
      </div>
      <div className="card">
        <div className="card-head"><h3>Training sessions</h3><span className="small muted mono">studio.{ds.slug}</span></div>
        {l.sessions.length === 0 ? <div className="card-pad small muted">No models trained on this dataset yet.</div> : (
          <div className="table-wrap"><table className="dense"><thead><tr><th>Session</th><th>Model</th><th>Task</th><th>Created</th></tr></thead>
            <tbody>{l.sessions.map((s: any) => <tr key={s.id}><td><Link to={`/sessions/${s.id}`}>{s.name}</Link></td><td className="mono small">{s.model_id}</td>
              <td className="small">{s.task}</td><td className="small muted">{fmtTime(s.created_at)}</td></tr>)}</tbody></table></div>
        )}
        <div className="card-pad small muted" style={{ borderTop: '1px solid var(--border)' }}>Every run logs the Studio dataset version it trained on (its{" "}
          <span className="mono">dataset_info</span> artifact). New runs always use the latest built version.</div>
      </div>
    </div>
  )
}

function Node({ id, name, kind, sub, depth = 0, current }: { id: string; name: string; kind: string; sub?: string; depth?: number; current?: boolean }) {
  return (
    <div className={`lin-node${current ? ' current' : ''}`} style={{ marginLeft: depth * 22 }}>
      <span className="lin-kind">{KIND[kind] ?? kind}</span>
      {current ? <b>{name}</b> : <Link to={`/studio/d/${id}`}>{name}</Link>}
      {sub && <span className="small muted">{sub}</span>}
    </div>
  )
}
