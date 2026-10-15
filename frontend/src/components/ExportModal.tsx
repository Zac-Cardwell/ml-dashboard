import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, fmtBytes, type SessionDetail } from '../api'
import { ErrorBox, Modal } from './ui'

export type GitIdentity = { name: string; email: string; git: boolean; gh: boolean }

/** Build a standalone, GitHub-ready repository from one or more sessions (nothing is published here). */
export function ExportModal({ sessionIds, onClose }: { sessionIds: string[]; onClose: () => void }) {
  const nav = useNavigate()
  const [sessions, setSessions] = useState<SessionDetail[]>()
  const [title, setTitle] = useState('')
  const [desc, setDesc] = useState('')
  const [over, setOver] = useState<Record<string, { title: string; description: string }>>({})
  const [models, setModels] = useState('none')
  const [videos, setVideos] = useState(true)
  const [license, setLicense] = useState('MIT')
  const [author, setAuthor] = useState('')
  const [email, setEmail] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<unknown>()

  useEffect(() => {
    Promise.all(sessionIds.map((id) => api.get<SessionDetail>(`/sessions/${id}`))).then((ss) => {
      setSessions(ss)
      setOver(Object.fromEntries(ss.map((s) => [s.id, { title: s.name, description: s.notes ?? '' }])))
      if (!title) setTitle(ss.length === 1 ? ss[0].name : 'ML experiments')
    }).catch(setErr)
    api.get<{ identity: GitIdentity }>('/exports').then((r) => { setAuthor(r.identity.name); setEmail(r.identity.email) }).catch(() => {})
  }, [sessionIds.join(',')]) // eslint-disable-line react-hooks/exhaustive-deps

  const modelBytes = (sessions ?? []).reduce((a, s) => a + s.runs.reduce((b, r) => b + (r.storage?.model_bytes ?? 0)
    + (models === 'both' ? r.storage?.final_bytes ?? 0 : 0), 0), 0)
  const notDone = (sessions ?? []).filter((s) => !s.runs.some((r) => r.status === 'completed'))

  const submit = async () => {
    setBusy(true); setErr(undefined)
    try {
      const r = await api.post<{ name: string }>('/exports', {
        session_ids: sessionIds, title, description: desc, author, email, license, include_models: models, include_videos: videos, overrides: over,
      })
      nav(`/exports?open=${r.name}`)
    } catch (e) { setErr(e); setBusy(false) }
  }

  return (
    <Modal title={`Export ${sessionIds.length} session${sessionIds.length > 1 ? 's' : ''} as a portfolio repository`} onClose={onClose}>
      <div className="col" style={{ gap: 12 }}>
        <div className="small text-2">
          Creates a self-contained git repository: a README with results tables and plots, one config per experiment, the original
          metrics, and the training code with a <code>train.py</code> so anyone can re-run every experiment with <code>pip install</code> —
          no dashboard needed. Datasets that can't be downloaded (live sources, uploads) are included as frozen snapshots. Nothing is
          published until you push it from the Exports page.
        </div>
        <ErrorBox error={err} />
        {notDone.length > 0 && <div className="notice small">{notDone.map((s) => s.name).join(', ')} {notDone.length > 1 ? 'have' : 'has'} no completed seeds — exported as config only.</div>}
        <label className="field"><span className="lbl">Repository title</span><input value={title} onChange={(e) => setTitle(e.target.value)} /></label>
        <label className="field"><span className="lbl">Intro (README)</span>
          <textarea value={desc} onChange={(e) => setDesc(e.target.value)} placeholder="What these experiments explore, what you found…" /></label>
        {sessions && (
          <details open={sessions.length <= 4}>
            <summary className="small">Experiment names & descriptions ({sessions.length}) — rename for the README without renaming the sessions</summary>
            <div className="col" style={{ gap: 8, marginTop: 8 }}>
              {sessions.map((s) => (
                <div key={s.id} className="grid-2" style={{ gap: 8 }}>
                  <input value={over[s.id]?.title ?? ''} onChange={(e) => setOver((o) => ({ ...o, [s.id]: { ...o[s.id], title: e.target.value } }))} />
                  <input value={over[s.id]?.description ?? ''} placeholder="description (optional)"
                    onChange={(e) => setOver((o) => ({ ...o, [s.id]: { ...o[s.id], description: e.target.value } }))} />
                </div>
              ))}
            </div>
          </details>
        )}
        <div className="grid-3" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(190px, 1fr))' }}>
          <label className="field"><span className="lbl">Trained weights</span>
            <select value={models} onChange={(e) => setModels(e.target.value)}>
              <option value="none">don't include</option><option value="best">best checkpoint</option><option value="both">best + final</option>
            </select>
            {models !== 'none' && <span className="help">≈ {fmtBytes(modelBytes)} · files over 100 MB are skipped (GitHub limit)</span>}</label>
          <label className="field"><span className="lbl">License</span>
            <select value={license} onChange={(e) => setLicense(e.target.value)}><option>MIT</option><option>Apache-2.0</option><option value="none">none</option></select></label>
          <label className="field"><span className="lbl">RL episode videos</span>
            <select value={videos ? 'y' : 'n'} onChange={(e) => setVideos(e.target.value === 'y')}><option value="y">include (≤ 8 MB)</option><option value="n">skip</option></select></label>
        </div>
        <div className="grid-2">
          <label className="field"><span className="lbl">Commit author</span><input value={author} onChange={(e) => setAuthor(e.target.value)} /></label>
          <label className="field"><span className="lbl">Commit email</span><input value={email} onChange={(e) => setEmail(e.target.value)} />
            <span className="help">Visible in the public commit history. GitHub's private <code>…@users.noreply.github.com</code> address works too.</span></label>
        </div>
        <button className="primary" style={{ alignSelf: 'flex-start' }} disabled={busy || !title.trim() || !sessions} onClick={submit}>
          {busy ? 'Building (snapshotting data, rendering plots)…' : 'Build repository'}</button>
      </div>
    </Modal>
  )
}
