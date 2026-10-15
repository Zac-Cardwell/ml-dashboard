import { useState } from 'react'
import { Link } from 'react-router-dom'
import { api, fmtTime } from '../api'
import { ErrorBox, usePoll } from './ui'

type Q = { id: string; name: string; priority: number; scheduled_at: number | null; device: string; running: number; queued: number; runs: number; created_at: number }
const PRI: [number, string][] = [[-1, 'Low'], [0, 'Normal'], [1, 'High'], [2, 'Urgent']]

function toLocalInput(t: number) {
  const d = new Date(t * 1000)
  return new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0, 16)
}

/** What's running and what's waiting, in the order the scheduler will start it. */
export function QueuePanel() {
  const [q, , reload] = usePoll(() => api.get<Q[]>('/queue'), 3000)
  const [err, setErr] = useState<unknown>()
  const [editing, setEditing] = useState<string>()
  const [when, setWhen] = useState('')
  if (!q || !q.length) return null
  const patch = (id: string, body: object) => api.patch(`/sessions/${id}/queue`, body).then(reload).catch(setErr)
  return (
    <div className="card" style={{ marginBottom: 14 }}>
      <div className="card-head"><h3>Queue</h3><span className="small text-2">running first, then by priority; scheduled sessions wait for their start time</span></div>
      <ErrorBox error={err} />
      <div className="table-wrap">
        <table><tbody>{q.map((s, i) => {
          const scheduled = s.scheduled_at && s.scheduled_at > Date.now() / 1000
          return (
            <tr key={s.id}>
              <td className="num small muted" style={{ width: 30 }}>{s.running ? '▶' : i + 1}</td>
              <td><Link to={`/sessions/${s.id}`}>{s.name}</Link><div className="small muted">{s.device} · {s.running ? `${s.running} running, ` : ''}{s.queued} of {s.runs} waiting</div></td>
              <td style={{ width: 130 }}>
                <select value={s.priority} onChange={(e) => patch(s.id, { priority: Number(e.target.value) })}>{PRI.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select>
              </td>
              <td className="small" style={{ width: 280 }}>
                {editing === s.id ? (
                  <div className="row" style={{ flexWrap: 'nowrap', gap: 4 }}>
                    <input type="datetime-local" value={when} onChange={(e) => setWhen(e.target.value)} />
                    <button className="sm" onClick={() => { patch(s.id, when ? { scheduled_at: new Date(when).getTime() / 1000 } : { clear_schedule: true }); setEditing(undefined) }}>Set</button>
                  </div>
                ) : scheduled ? <>starts {fmtTime(s.scheduled_at)}</> : s.running ? 'running' : 'next free slot'}
              </td>
              <td style={{ textAlign: 'right', whiteSpace: 'nowrap' }}>
                {!s.running && scheduled && <button className="sm" onClick={() => patch(s.id, { clear_schedule: true })}>Start now</button>}
                {!s.running && editing !== s.id && <button className="sm ghost" onClick={() => { setEditing(s.id); setWhen(s.scheduled_at ? toLocalInput(s.scheduled_at) : '') }}>Schedule…</button>}
              </td>
            </tr>
          )
        })}</tbody></table>
      </div>
    </div>
  )
}
