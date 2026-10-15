import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, fmtBytes } from '../api'
import { ErrorBox, Modal, usePoll } from './ui'

type Plan = { id: string; name: string; kind: 'upload' | 'file' | 'hide' | 'live'; action: string; sessions: number
  data_bytes: number; data_paths: string[]; reports: number; file?: string }
type Removed = { kind: 'hidden' | 'trash'; id: string; name: string; restore: string; deleted?: string }

/** Confirm-and-remove dialog for one dataset or RL environment (shows exactly what will happen first). */
export function RemoveDatasetModal({ id, onClose, onDone, kind = 'dataset' }: { id: string; onClose: () => void; onDone: () => void; kind?: 'dataset' | 'env' }) {
  const noun = kind === 'env' ? 'environment' : 'dataset'
  const [plan, setPlan] = useState<Plan>()
  const [delData, setDelData] = useState(true)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<unknown>()
  useEffect(() => { api.get<Plan>(`/plugins/${kind}/${id}/removal`).then(setPlan).catch(setErr) }, [id, kind])
  const go = async () => {
    setBusy(true); setErr(undefined)
    try { await api.del(`/plugins/${kind}/${id}?delete_data=${delData}`); onDone() } catch (e) { setErr(e); setBusy(false) }
  }
  return (
    <Modal title={`Remove ${noun}${plan ? ` · ${plan.name}` : ''}`} onClose={onClose}>
      <div className="col" style={{ gap: 12 }}>
        <ErrorBox error={err} />
        {plan && (plan.kind === 'live' ? (
          <div className="small">{plan.action} <Link to="/data">Open Data →</Link></div>
        ) : (
          <>
            <div className="small">{plan.action}</div>
            {plan.sessions > 0 && <div className="notice small">{plan.sessions} session{plan.sessions > 1 ? 's' : ''} used this {noun}. Their results, charts and saved
              models stay; re-running them (Train again / Resume) needs the {noun} back.</div>}
            {(plan.data_bytes > 0 || plan.reports > 0) ? (
              <label className="row small" style={{ gap: 8, flexWrap: 'nowrap', alignItems: 'flex-start' }}>
                <input type="checkbox" checked={delData} onChange={(e) => setDelData(e.target.checked)} style={{ width: 'auto', marginTop: 2 }} />
                <span>Also delete downloaded data{plan.data_bytes > 0 && <> ({fmtBytes(plan.data_bytes)}: <span className="mono">{plan.data_paths.join(', ')}</span>)</>}
                  {plan.reports > 0 && <> and {plan.reports} quality report{plan.reports > 1 ? 's' : ''}</>}. It's downloaded again if the {noun} is used later.</span>
              </label>
            ) : <div className="small muted">No downloaded files for this {noun}.</div>}
            <button className="danger" style={{ alignSelf: 'flex-start' }} disabled={busy} onClick={go}>{busy ? 'Removing…' : `Remove ${noun}`}</button>
          </>
        ))}
      </div>
    </Modal>
  )
}

/** Hidden built-ins and trashed plugin files (datasets or environments), with Restore. */
export function RemovedDatasets({ version, onRestored, kind = 'dataset' }: { version: number; onRestored: () => void; kind?: 'dataset' | 'env' }) {
  const [items, err, reload] = usePoll(() => api.get<Removed[]>(`/plugins-removed/${kind}`), 0, [version, kind])
  const [busy, setBusy] = useState<string>()
  if (!items?.length) return err ? <ErrorBox error={err} /> : null
  const restore = async (key: string) => {
    setBusy(key)
    try { await api.post(`/plugins-removed/${kind}/restore`, { key }); reload(); onRestored() } finally { setBusy(undefined) }
  }
  return (
    <div className="card">
      <div className="card-head"><h3>Removed {kind === 'env' ? 'environments' : 'datasets'}</h3><span className="small muted">restore to list them again</span></div>
      <table><tbody>{items.map((r) => (
        <tr key={r.kind + r.id}>
          <td>{r.name}{r.kind === 'trash' && <div className="mono small muted">user_plugins/{r.name} · removed {r.deleted}</div>}</td>
          <td className="small"><span className="tag">{r.kind === 'hidden' ? 'hidden' : 'in trash'}</span></td>
          <td style={{ textAlign: 'right' }}><button className="sm" disabled={busy === r.restore} onClick={() => restore(r.restore)}>Restore</button></td>
        </tr>))}</tbody></table>
    </div>
  )
}
