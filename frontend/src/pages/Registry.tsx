import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api, fmtBytes, fmtMetric, fmtTime, fmtValue } from '../api'
import { ConfirmButton, ErrorBox, Modal, StageBadge, usePoll } from '../components/ui'
import { type RegisteredModel, type Version, type VersionDetail } from '../mlops'
import { usePluginNames } from './Sessions'

const KEY_METRICS = ['test/acc', 'test/f1', 'test/rmse', 'test/r2', 'test/bleu', 'final/mean_reward']

export default function Registry() {
  const [models, err, reload] = usePoll(() => api.get<RegisteredModel[]>('/registry'), 10000)
  const [detail, setDetail] = useState<string>()
  const [deployFor, setDeployFor] = useState<Version>()
  const name = usePluginNames()

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>Model registry</h1>
          <div className="sub">Immutable, versioned copies of trained models, each with its fitted preprocessing, training-data profile and model card.
            Register a model from any session's <b>Register</b> button.</div>
        </div>
        <Link className="btn" to="/deployments">Deployments →</Link>
      </div>
      <ErrorBox error={err} />
      {models && !models.length && <div className="card empty">No registered models yet. Open a completed session with a saved model and click <b>Register</b>.</div>}
      <div className="col" style={{ gap: 16 }}>
        {(models ?? []).map((m) => (
          <div className="card" key={m.name}>
            <div className="card-head" style={{ flexWrap: 'wrap' }}>
              <div>
                <h2 className="mono">{m.name}</h2>
                <div className="small text-2">{name(m.model_id)} · {m.task} · {m.modality} · {m.versions.length} version{m.versions.length === 1 ? '' : 's'}</div>
              </div>
              <div className="chips">
                {m.deployments.map((d) => <Link key={d.id} className="tag" to={`/deployments/${d.id}`}>deployed: {d.name}</Link>)}
              </div>
            </div>
            <div className="table-wrap">
              <table>
                <thead><tr><th>Version</th><th>Stage</th>{KEY_METRICS.filter((k) => m.versions.some((v) => k in v.metrics)).map((k) => <th key={k} className="num">{k}</th>)}<th>Created</th><th>Source</th><th></th></tr></thead>
                <tbody>
                  {m.versions.map((v) => (
                    <tr key={v.id} className="clickable" onClick={() => setDetail(v.id)}>
                      <td className="mono" style={{ fontWeight: 600 }}>v{v.version}</td>
                      <td><StageBadge stage={v.stage} />{v.created_by === 'automation' && <span className="tag" style={{ marginLeft: 6 }}>auto</span>}</td>
                      {KEY_METRICS.filter((k) => m.versions.some((x) => k in x.metrics)).map((k) => <td key={k} className="num">{fmtMetric(k, v.metrics[k])}</td>)}
                      <td className="small">{fmtTime(v.created_at)}</td>
                      <td className="small" onClick={(e) => e.stopPropagation()}>{v.source_session_id ? <Link to={`/sessions/${v.source_session_id}`}>session</Link> : '—'}</td>
                      <td style={{ textAlign: 'right' }} onClick={(e) => e.stopPropagation()}>
                        <div className="row" style={{ justifyContent: 'flex-end', gap: 4 }}>
                          <button className="sm" onClick={() => setDeployFor(v)}>Deploy</button>
                          <a className="btn sm ghost" href={`/api/registry/versions/${v.id}/export`}>Export</a>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ))}
      </div>
      {detail && <VersionModal id={detail} onClose={() => { setDetail(undefined); reload() }} />}
      {deployFor && <DeployModal version={deployFor} onClose={() => setDeployFor(undefined)} />}
    </div>
  )
}

function VersionModal({ id, onClose }: { id: string; onClose: () => void }) {
  const [v, err, reload] = usePoll(() => api.get<VersionDetail>(`/registry/versions/${id}`), 0, [id])
  const [actErr, setActErr] = useState<unknown>()
  const act = async (fn: () => Promise<unknown>) => { setActErr(undefined); try { await fn(); reload() } catch (e) { setActErr(e) } }
  return (
    <Modal title={v ? `${v.model_name} v${v.version}` : 'Model version'} onClose={onClose}>
      <ErrorBox error={err ?? actErr} />
      {v && (
        <div className="col" style={{ gap: 14 }}>
          <div className="row">
            <StageBadge stage={v.stage} />
            <span className="small text-2">{v.model_id} · {v.framework} · {fmtBytes(v.size_bytes)} · created {fmtTime(v.created_at)} by {v.created_by}</span>
          </div>
          {v.description && <div className="notice small">{v.description}</div>}
          <div className="row">
            <span className="small text-2">Stage:</span>
            {['staging', 'archived', 'none'].map((s) => <button key={s} className="sm" disabled={v.stage === s || v.stage === 'production'} onClick={() => act(() => api.patch(`/registry/versions/${id}`, { stage: s }))}>{s === 'none' ? 'unstage' : s}</button>)}
            <span className="small muted">(production is set by promoting in a deployment)</span>
          </div>
          {v.deployed_in.length > 0 && <div className="small">Used by: {v.deployed_in.map((d) => <Link key={d.id} to={`/deployments/${d.id}`} style={{ marginRight: 8 }}>{d.name} ({d.role})</Link>)}</div>}
          <div className="grid-2">
            <div className="card card-pad">
              <h3 style={{ marginBottom: 8 }}>Offline metrics</h3>
              <div className="kv small">{Object.entries(v.metrics).filter(([k]) => k.includes('/')).map(([k, x]) => <div key={k} style={{ display: 'contents' }}><div className="mono">{k}</div><div className="num">{fmtMetric(k, x)}</div></div>)}</div>
            </div>
            <div className="card card-pad">
              <h3 style={{ marginBottom: 8 }}>Model card</h3>
              <div className="kv small">
                <div>Trained on</div><div className="mono">{v.card.trained_on?.problem}</div>
                <div>Seed</div><div>{v.card.seed}</div>
                <div>Reference profile</div><div>{v.profile.n_reference} examples · {Object.keys(v.profile.features).length} monitored features</div>
                {Object.entries(v.card.versions ?? {}).slice(0, 5).map(([k, x]) => <div key={k} style={{ display: 'contents' }}><div>{k}</div><div className="mono">{String(x)}</div></div>)}
              </div>
            </div>
          </div>
          {v.profile.schema && (
            <div className="card card-pad">
              <h3 style={{ marginBottom: 8 }}>Input schema (raw features accepted at serving time)</h3>
              <div className="table-wrap"><table><tbody>
                {Object.entries(v.profile.schema).map(([k, t]) => {
                  const f = v.profile.features[k]
                  return <tr key={k}><td className="mono">{k}</td><td className="small">{t}</td>
                    <td className="small text-2">{f?.type === 'numeric' ? `range ${fmtValue(f.min)} – ${fmtValue(f.max)}, mean ${fmtValue(f.mean)}` : f?.categories?.slice(0, 8).join(', ')}</td></tr>
                })}
              </tbody></table></div>
            </div>
          )}
          <details><summary>Hyperparameters</summary><pre className="log mono small">{JSON.stringify(v.card.hyperparameters, null, 2)}</pre></details>
          <div className="row">
            <a className="btn" href={`/api/registry/versions/${id}/export`}>Download serving package (.zip)</a>
            <ConfirmButton className="danger" disabled={v.deployed_in.length > 0} onConfirm={() => act(async () => { await api.del(`/registry/versions/${id}`); onClose() })}>Delete version</ConfirmButton>
          </div>
        </div>
      )}
    </Modal>
  )
}

export function DeployModal({ version, onClose }: { version: Version; onClose: () => void }) {
  const [name, setName] = useState(version.model_name)
  const [kind, setKind] = useState('internal')
  const [err, setErr] = useState<unknown>()
  const nav = useNavigate()
  const submit = async () => {
    try {
      const r = await api.post<{ id: string }>('/deployments', { name, version_id: version.id, kind })
      nav(`/deployments/${r.id}`)
    } catch (e) { setErr(e) }
  }
  return (
    <Modal title={`Deploy ${version.model_name} v${version.version}`} onClose={onClose}>
      <div className="col" style={{ gap: 12 }}>
        <ErrorBox error={err} />
        <label className="field"><span className="lbl">Deployment name</span><input value={name} onChange={(e) => setName(e.target.value)} />
          <span className="help">Used in the endpoint URL (lowercase, dashes).</span></label>
        <div className="col" style={{ gap: 6 }}>
          <label className="check"><input type="radio" checked={kind === 'internal'} onChange={() => setKind('internal')} />
            <span><b>Serve here</b> — this dashboard hosts a prediction endpoint; every request is logged and monitored.</span></label>
          <label className="check"><input type="radio" checked={kind === 'external'} onChange={() => setKind('external')} />
            <span><b>Hosted elsewhere</b> — run the exported package (or your own service) anywhere; it pushes prediction logs here with the SDK.</span></label>
        </div>
        <button className="primary" style={{ alignSelf: 'flex-start' }} onClick={submit}>Create deployment</button>
      </div>
    </Modal>
  )
}
