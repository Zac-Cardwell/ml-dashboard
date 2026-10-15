import { Link } from 'react-router-dom'
import { api, fmtValue } from '../api'
import { baseLayout, Plot, useTheme, withAlpha } from './Chart'
import { ConfirmButton, Progress, usePoll } from './ui'

type Trial = { trial_number: number; session_id: string | null; session_name: string | null; state: string; value: number | null; params: Record<string, unknown> }
type Search = { id: string; name: string; status: string; metric: string; direction: string; best_session_id: string | null; best_value: number | null
  config: { n_trials: number; parallel: number; pruning: boolean; sampler: string; space: Record<string, any> }; trials: Trial[]; importance: Record<string, number> }

const short = (k: string) => k.split('.').slice(1).join('.')

/** Optuna search results: progress, best trial, optimization history, parameter importance, trials. */
export function SearchPanel({ id }: { id: string }) {
  const theme = useTheme()
  const [s, , reload] = usePoll(() => api.get<Search>(`/searches/${id}`), 5000, [id])
  if (!s) return null
  const done = s.trials.filter((t) => ['complete', 'pruned', 'failed'].includes(t.state)).length
  const complete = s.trials.filter((t) => t.state === 'complete' && t.value != null).sort((a, b) => a.trial_number - b.trial_number)
  const maxi = s.direction === 'maximize'
  let best = maxi ? -Infinity : Infinity
  const bestSoFar = complete.map((t) => (best = maxi ? Math.max(best, t.value!) : Math.min(best, t.value!)))
  const bestTrial = s.trials.find((t) => t.session_id === s.best_session_id)
  const imp = Object.entries(s.importance).sort((a, b) => b[1] - a[1])
  const keys = Object.keys(s.config.space)
  const stateColor: Record<string, string> = { complete: 'completed', running: 'running', pruning: 'stopping', pruned: 'stopped', failed: 'failed' }

  const layout = { ...baseLayout(theme, 'trial', s.metric), height: 280, hovermode: 'closest' as const }
  return (
    <div className="card">
      <div className="card-head" style={{ flexWrap: 'wrap' }}>
        <div>
          <h2>{s.name}</h2>
          <div className="small text-2">Optuna {s.config.sampler.toUpperCase()} · {maxi ? 'maximize' : 'minimize'} <span className="mono">{s.metric}</span>
            {s.config.pruning && ' · median pruning'} · {s.config.parallel} in parallel</div>
        </div>
        <div className="row">
          <span className={`badge ${s.status === 'running' ? 'running' : s.status === 'completed' ? 'completed' : 'stopped'}`}><span className="dot" />{s.status}</span>
          {s.status === 'running' && <ConfirmButton className="sm danger" onConfirm={() => api.post(`/searches/${id}/stop`).then(reload)}>Stop search</ConfirmButton>}
        </div>
      </div>
      <div className="card-pad col" style={{ gap: 14 }}>
        <div className="row" style={{ flexWrap: 'nowrap' }}>
          <span className="small num" style={{ whiteSpace: 'nowrap' }}>{done}/{s.config.n_trials} trials</span>
          <div className="grow"><Progress value={done / s.config.n_trials} /></div>
          <span className="small text-2" style={{ whiteSpace: 'nowrap' }}>{s.trials.filter((t) => t.state === 'pruned').length} pruned</span>
        </div>
        {bestTrial && s.best_value != null && (
          <div className="notice">
            <b>Best so far:</b> {s.metric} = <b>{s.best_value.toFixed(4)}</b> (trial {bestTrial.trial_number}) —{' '}
            {Object.entries(bestTrial.params).map(([k, v]) => `${short(k)} = ${typeof v === 'number' ? Number(v.toPrecision(4)) : v}`).join(', ')}
            {' · '}<Link to={`/sessions/${bestTrial.session_id}`}>open session</Link>
            {' · '}<Link to={`/new?from=${bestTrial.session_id}`}>train with these settings</Link>
          </div>
        )}
        <div className="grid-2">
          <div className="card chart-card">
            <h3>Optimization history</h3>
            <div style={{ height: 280 }}>
              <Plot useResizeHandler style={{ width: '100%', height: '100%' }} config={{ displaylogo: false, responsive: true }} layout={layout}
                data={[
                  { type: 'scatter', mode: 'markers', name: 'trial', x: complete.map((t) => t.trial_number), y: complete.map((t) => t.value),
                    marker: { size: 9, color: withAlpha(theme.series[0], 0.75), line: { color: theme.surface, width: 1 } },
                    text: complete.map((t) => Object.entries(t.params).map(([k, v]) => `${short(k)}=${typeof v === 'number' ? Number(v.toPrecision(3)) : v}`).join('<br>')) as any,
                    hovertemplate: 'trial %{x}: %{y:.4f}<br>%{text}<extra></extra>' },
                  { type: 'scatter', mode: 'lines', name: 'best so far', x: complete.map((t) => t.trial_number), y: bestSoFar,
                    line: { color: theme.series[1], width: 2, shape: 'hv' }, hoverinfo: 'skip' },
                ]} />
            </div>
          </div>
          <div className="card chart-card">
            <h3>Parameter importance</h3>
            {imp.length ? (
              <div style={{ height: Math.max(160, 40 + imp.length * 30) }}>
                <Plot useResizeHandler style={{ width: '100%', height: '100%' }} config={{ displaylogo: false, responsive: true }}
                  layout={{ ...baseLayout(theme, 'share of variance in the objective'), height: Math.max(160, 40 + imp.length * 30), showlegend: false, hovermode: 'closest',
                    margin: { l: 10, r: 20, t: 10, b: 40 }, yaxis: { ...baseLayout(theme, '').yaxis, automargin: true, type: 'category', autorange: 'reversed' } }}
                  data={[{ type: 'bar', orientation: 'h', y: imp.map(([k]) => short(k)), x: imp.map(([, v]) => v), marker: { color: theme.series[0] }, hovertemplate: '%{y}: %{x:.2f}<extra></extra>' }]} />
              </div>
            ) : <div className="empty small">Appears after 4 completed trials (needs 2+ searched parameters).</div>}
          </div>
        </div>
        <div className="table-wrap">
          <table>
            <thead><tr><th className="num">#</th><th>State</th>{keys.map((k) => <th key={k} className="mono">{short(k)}</th>)}<th className="num">{s.metric}</th><th>Session</th></tr></thead>
            <tbody>{[...s.trials].sort((a, b) => (b.value ?? -Infinity) * (maxi ? 1 : -1) - (a.value ?? -Infinity) * (maxi ? 1 : -1)).map((t) => (
              <tr key={t.trial_number} style={t.session_id === s.best_session_id ? { background: 'var(--accent-soft)' } : undefined}>
                <td className="num">{t.trial_number}</td>
                <td><span className={`badge ${stateColor[t.state] ?? ''}`}><span className="dot" />{t.state}</span></td>
                {keys.map((k) => <td key={k} className="mono small">{typeof t.params[k] === 'number' ? Number((t.params[k] as number).toPrecision(4)) : fmtValue(t.params[k])}</td>)}
                <td className="num">{t.value != null ? t.value.toFixed(4) : '—'}</td>
                <td className="small">{t.session_id && <Link to={`/sessions/${t.session_id}`}>open</Link>}</td>
              </tr>))}</tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
