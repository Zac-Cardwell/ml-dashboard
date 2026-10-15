import { useEffect, useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { api, fmtMetric, fmtValue, type SeriesByRun, type SessionBrief, type SessionDetail } from '../api'
import { LineChart, useTheme, type Line } from '../components/Chart'
import { ErrorBox, StatusBadge, usePoll } from '../components/ui'
import { allKeys, groupMetricKeys } from '../metrics'
import { GroupResults, rankMembers, type Group } from '../components/GroupResults'
import { SearchPanel } from '../components/SearchPanel'
import { ChartControls } from './SessionDetail'
import { usePluginNames } from './Sessions'

type Loaded = { detail: SessionDetail; series: SeriesByRun }
const MAX = 8 // categorical palette size; more sessions would need cycled colors

const DEFAULT_KEYS = ['val/loss', 'val/acc', 'val/f1', 'val/rmse', 'val/bleu', 'eval/mean_reward', 'episode/reward', 'eval/episode_reward', 'train/loss']

function flatConfig(d: SessionDetail): Record<string, unknown> {
  return {
    problem: d.problem_id, model: d.model_id, seeds: d.seeds, device: d.device,
    ...Object.fromEntries(Object.entries(d.problem_params).map(([k, v]) => [`problem.${k}`, v])),
    ...Object.fromEntries(Object.entries(d.model_params).map(([k, v]) => [`model.${k}`, v])),
  }
}

export default function Compare() {
  const [sp, setSp] = useSearchParams()
  const groupId = sp.get('group')
  const name = usePluginNames()
  const [group] = usePoll(() => (groupId ? api.get<Group>(`/groups/${groupId}`) : Promise.resolve(undefined)), groupId ? 5000 : 0, [groupId])
  const explicit = (sp.get('ids') ?? '').split(',').filter(Boolean)
  // for a group, overlay the best MAX members unless sessions were picked explicitly
  const groupIds = group ? rankMembers(group.sessions).slice(0, MAX).map((m) => m.id) : []
  const ids = (explicit.length ? explicit : groupIds).slice(0, MAX)
  const idsKey = ids.join(',') + (group ? group.sessions.map((s) => s.status).join() : '')
  const [loaded, setLoaded] = useState<Record<string, Loaded>>({})
  const [err, setErr] = useState<unknown>()
  const [keys, setKeys] = useState<string[] | null>(null)
  const [showSeeds, setShowSeeds] = useState(false)
  const [smoothing, setSmoothing] = useState(0)
  const [all] = usePoll(() => api.get<SessionBrief[]>('/sessions'), 0)
  const theme = useTheme()

  useEffect(() => {
    let alive = true
    Promise.all(ids.map(async (id) => {
      const [detail, m] = await Promise.all([
        api.get<SessionDetail>(`/sessions/${id}`),
        api.get<{ series: SeriesByRun }>(`/sessions/${id}/metrics?aggregate=false`),
      ])
      return [id, { detail, series: m.series }] as const
    })).then((xs) => alive && setLoaded(Object.fromEntries(xs))).catch((e) => alive && setErr(e))
    return () => { alive = false }
  }, [idsKey]) // eslint-disable-line react-hooks/exhaustive-deps

  const sessions = ids.map((id) => loaded[id]).filter(Boolean)
  const unionKeys = useMemo(() => {
    const s = new Set<string>()
    for (const l of sessions) for (const k of allKeys(l.series)) s.add(k)
    return groupMetricKeys([...s]).flatMap((g) => g.keys)
  }, [sessions])
  const plotKeys = keys ?? DEFAULT_KEYS.filter((k) => unionKeys.includes(k)).slice(0, 4)

  const setIds = (next: string[]) => setSp(groupId ? { group: groupId, ids: next.join(',') } : { ids: next.join(',') })
  const configs = sessions.map((s) => flatConfig(s.detail))
  const diffKeys = [...new Set(configs.flatMap(Object.keys))].filter((k) => new Set(configs.map((c) => JSON.stringify(c[k]))).size > 1).sort()
  const metricKeys = [...new Set(sessions.flatMap((s) => Object.keys(s.detail.summary_stats)))].sort()

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>Compare sessions</h1>
          <div className="sub">Overlay curves (seed mean ± std) and diff configurations. Up to {MAX} sessions.</div>
        </div>
        <select style={{ width: 320 }} value="" onChange={(e) => e.target.value && setIds([...ids, e.target.value])} disabled={ids.length >= MAX}>
          <option value="">+ Add session…</option>
          {(all ?? []).filter((s) => !ids.includes(s.id)).map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
        </select>
      </div>
      <ErrorBox error={err} />
      {group && <div style={{ marginBottom: 16 }}>{group.kind === 'search' ? <SearchPanel id={group.group_id} /> : <GroupResults group={group} name={name} />}</div>}
      {group && group.sessions.length > MAX && !explicit.length && <div className="small text-2" style={{ marginBottom: 10 }}>Curves below show the top {MAX} members.</div>}
      {!ids.length && !groupId && <div className="card empty">Select sessions on the Sessions page (checkboxes) or add them above.</div>}

      {sessions.length > 0 && (
        <div className="col" style={{ gap: 16 }}>
          <div className="card card-pad col" style={{ gap: 8 }}>
            {sessions.map((s, i) => (
              <div key={s.detail.id} className="row">
                <span style={{ width: 14, height: 4, borderRadius: 2, background: theme.series[i] }} />
                <Link to={`/sessions/${s.detail.id}`} style={{ fontWeight: 550 }}>{s.detail.name}</Link>
                <StatusBadge status={s.detail.status} />
                <span className="muted small">{s.detail.n_seeds} seed{s.detail.n_seeds > 1 ? 's' : ''}</span>
                <button className="ghost sm" onClick={() => setIds(ids.filter((x) => x !== s.detail.id))} aria-label="remove">✕</button>
              </div>
            ))}
          </div>

          <div className="card card-pad col">
            <div className="small text-2">Metrics to plot</div>
            <div className="chips">
              {unionKeys.map((k) => (
                <label key={k} className="check tag" style={{ padding: '3px 8px' }}>
                  <input type="checkbox" checked={plotKeys.includes(k)} onChange={(e) => setKeys(e.target.checked ? [...plotKeys, k] : plotKeys.filter((x) => x !== k))} />
                  <span className="mono">{k}</span>
                </label>
              ))}
            </div>
            <ChartControls showSeeds={showSeeds} setShowSeeds={setShowSeeds} smoothing={smoothing} setSmoothing={setSmoothing} multi />
          </div>

          <div className="grid-2">
            {plotKeys.map((k) => {
              const lines: Line[] = sessions.map((s, i) => {
                const seedOf = Object.fromEntries(s.detail.runs.map((r) => [r.id, r.seed]))
                return { label: s.detail.name, color: theme.series[i], runs: Object.entries(s.series).filter(([, m]) => m[k]).map(([rid, m]) => ({ seed: seedOf[rid], ...m[k] })) }
              })
              return (
                <div className="card chart-card" key={k}>
                  <h3 className="mono">{k}</h3>
                  <LineChart lines={lines} xTitle="step" showSeeds={showSeeds} smoothing={smoothing} legend={false} />
                </div>
              )
            })}
          </div>

          <div className="card">
            <div className="card-head"><h2>Final metrics</h2><span className="muted small">mean ± std across seeds</span></div>
            <div className="table-wrap">
              <table>
                <thead><tr><th>Metric</th>{sessions.map((s, i) => <th key={s.detail.id} className="num"><span style={{ color: theme.series[i] }}>■</span> {s.detail.name}</th>)}</tr></thead>
                <tbody>
                  {metricKeys.map((k) => {
                    const vals = sessions.map((s) => s.detail.summary_stats[k]?.mean)
                    const nums = vals.filter((v): v is number => typeof v === 'number')
                    const lowerBetter = /loss|mse|rmse|mae|time|std/.test(k)
                    const best = nums.length > 1 ? (lowerBetter ? Math.min(...nums) : Math.max(...nums)) : undefined
                    return (
                      <tr key={k}>
                        <td className="mono">{k}</td>
                        {sessions.map((s, i) => {
                          const st = s.detail.summary_stats[k]
                          return (
                            <td key={s.detail.id} className="num" style={vals[i] === best && !/time|n_params|epoch|steps/.test(k) ? { fontWeight: 700 } : undefined}>
                              {st ? <>{fmtMetric(k, st.mean)}{st.n > 1 && <span className="muted small"> ± {fmtMetric(k, st.std)}</span>}</> : '—'}
                            </td>
                          )
                        })}
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>

          <div className="card">
            <div className="card-head"><h2>Configuration differences</h2></div>
            {diffKeys.length === 0 ? <div className="empty">All configurations are identical.</div> : (
              <div className="table-wrap">
                <table>
                  <thead><tr><th>Parameter</th>{sessions.map((s, i) => <th key={s.detail.id}><span style={{ color: theme.series[i] }}>■</span> {s.detail.name}</th>)}</tr></thead>
                  <tbody>{diffKeys.map((k) => <tr key={k}><td className="mono">{k}</td>{configs.map((c, i) => <td key={i} className="mono">{fmtValue(c[k])}</td>)}</tr>)}</tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
