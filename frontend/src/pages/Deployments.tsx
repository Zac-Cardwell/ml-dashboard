import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api, fmtMetric, fmtNum, fmtTime } from '../api'
import { ErrorBox, HealthBadge, usePoll } from '../components/ui'
import { type Alert, type Deployment } from '../mlops'

export default function Deployments() {
  const [deps, err] = usePoll(() => api.get<Deployment[]>('/deployments'), 5000)
  const [alerts] = usePoll(() => api.get<Alert[]>('/alerts'), 5000)
  const nav = useNavigate()

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>Deployments</h1>
          <div className="sub">Models in production: health, drift, live performance and automated retraining. Deploy from the <Link to="/registry">registry</Link>.</div>
        </div>
      </div>
      <ErrorBox error={err} />
      <DemoCard deps={deps} />
      {(alerts ?? []).length > 0 && (
        <div className="card section">
          <div className="card-head"><h2>Open alerts</h2><span className="small text-2">{alerts!.length}</span></div>
          <div className="table-wrap" style={{ maxHeight: 260 }}>
            <table><tbody>
              {alerts!.map((a) => (
                <tr key={a.id} className="clickable" onClick={() => nav(`/deployments/${a.deployment_id}`)}>
                  <td style={{ width: 110 }}><span className={`badge ${a.severity === 'critical' ? 'failed' : 'queued'}`}><span className="dot" />{a.severity}</span></td>
                  <td className="small">{a.deployment_name}</td><td>{a.message}</td><td className="small muted">{fmtTime(a.updated_at)}</td>
                </tr>
              ))}
            </tbody></table>
          </div>
        </div>
      )}
      <div className="grid-3 section" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(330px, 1fr))' }}>
        {(deps ?? []).map((d) => (
          <Link key={d.id} to={`/deployments/${d.id}`} className="card card-pad col" style={{ gap: 8, color: 'inherit', textDecoration: 'none' }}>
            <div className="row" style={{ justifyContent: 'space-between' }}>
              <h2>{d.name}</h2>
              <HealthBadge health={d.status === 'paused' ? 'no_data' : d.health} />
            </div>
            <div className="small text-2">
              {d.model_name} v{d.champion?.version ?? '–'}{d.challenger && <> · challenger v{d.challenger.version} ({d.challenger_mode})</>}
              {' · '}{d.kind === 'internal' ? 'served here' : 'external'}{d.status === 'paused' && ' · paused'}
            </div>
            <div className="kv small" style={{ marginTop: 4 }}>
              <div>Requests (24h)</div><div className="num">{d.requests_24h.toLocaleString()} <span className="muted">· {d.labeled_24h.toLocaleString()} labeled</span></div>
              <div>Live {d.latest.perf?.key ?? 'metric'}</div>
              <div className="num">{d.latest.perf ? <>{fmtMetric(d.latest.perf.key, d.latest.perf.value)} <span className="muted">vs {fmtMetric(d.latest.perf.key, d.latest.perf.baseline ?? undefined)} baseline</span></> : '—'}</div>
              <div>Max feature PSI</div><div className="num">{d.latest.max_psi != null ? fmtNum(d.latest.max_psi, 3) : '—'}</div>
              <div>p95 latency</div><div className="num">{d.latest.latency_p95 != null ? `${d.latest.latency_p95.toFixed(1)} ms` : '—'}</div>
            </div>
            {d.health_reasons.length > 0 && <div className="small" style={{ color: d.health === 'critical' ? 'var(--critical-text)' : 'var(--text-2)' }}>{d.health_reasons[0].message}{d.health_reasons.length > 1 && ` (+${d.health_reasons.length - 1} more)`}</div>}
            {d.traffic && <span className="tag" style={{ alignSelf: 'flex-start' }}>demo traffic: {d.traffic.scenario}</span>}
          </Link>
        ))}
      </div>
      {deps && !deps.length && <div className="card empty section">No deployments yet. Try the demo above, or register a model and deploy it.</div>}
    </div>
  )
}

function DemoCard({ deps }: { deps?: Deployment[] }) {
  const [state, , reload] = usePoll(() => api.get<{ deployment: string | null; session_id: string | null; session_status: string | null }>('/mlops/demo'), 4000)
  const [err, setErr] = useState<unknown>()
  if (!state || state.deployment || deps?.some((d) => d.id === 'loan-default')) return null
  const busy = !!state.session_id
  return (
    <div className="card card-pad col" style={{ gap: 8 }}>
      <h2>Try the MLOps demo</h2>
      <div className="small text-2">
        One click trains an XGBoost credit-risk model on synthetic loan applications, registers it (with its training-data profile)
        and deploys it here with automated retraining enabled. Then start simulated traffic and switch on a drift scenario —
        an economic downturn, a new customer segment, a broken upstream feature, or concept drift — and watch monitoring, alerts and
        retraining react.
      </div>
      <ErrorBox error={err} />
      <button className="primary" style={{ alignSelf: 'flex-start' }} disabled={busy}
        onClick={() => api.post('/mlops/demo/setup').then(reload).catch(setErr)}>
        {busy ? `Training demo model… (${state.session_status ?? 'queued'})` : 'Set up demo deployment'}
      </button>
    </div>
  )
}
