import { Link } from 'react-router-dom'
import { fmtMetric, fmtValue, type Headline, type SessionBrief } from '../api'
import { baseLayout, Plot, useTheme } from './Chart'
import { Progress, StatusBadge } from './ui'

export type GroupMember = SessionBrief & {
  group_params: Record<string, unknown>
  model_params: Record<string, unknown>
  final_metrics: Record<string, { mean: number; std: number; n: number }>
}
export type Group = { group_id: string; kind: string; swept: string[]; sessions: GroupMember[] }

const short = (k: string) => k.split('.').slice(1).join('.') || k

export function rankMembers(members: GroupMember[]): GroupMember[] {
  const hib = members.find((m) => m.headline)?.headline?.higher_is_better ?? true
  return [...members].sort((a, b) => {
    if (!a.headline) return 1
    if (!b.headline) return -1
    return hib ? b.headline.mean - a.headline.mean : a.headline.mean - b.headline.mean
  })
}

/** Ranked results for a sweep / benchmark / multi-model group, plus a chart of the headline metric. */
export function GroupResults({ group, name }: { group: Group; name: (id: string) => string }) {
  const ranked = rankMembers(group.sessions)
  const key = ranked.find((m) => m.headline)?.headline?.key
  const active = group.sessions.filter((s) => ['running', 'queued', 'stopping'].includes(s.status)).length
  const done = group.sessions.filter((s) => s.status === 'completed').length
  const title = group.kind === 'sweep' ? `Sweep over ${group.swept.map(short).join(' × ')}`
    : group.kind === 'benchmark' ? `Benchmark · ${group.sessions.length} models on ${name(group.sessions[0]?.problem_id)}`
      : `${group.sessions.length} models on ${name(group.sessions[0]?.problem_id)}`
  const labelOf = (m: GroupMember) => group.kind === 'sweep'
    ? group.swept.map((k) => `${short(k)}=${fmtValue(m.group_params[k])}`).join(', ')
    : m.name

  return (
    <div className="card">
      <div className="card-head" style={{ flexWrap: 'wrap' }}>
        <h2>{title}</h2>
        <span className="small text-2">{done}/{group.sessions.length} completed{active ? ` · ${active} active` : ''}{key ? ` · ranked by ${key}` : ''}</span>
      </div>
      <div className="card-pad">
        <GroupChart group={group} ranked={ranked} labelOf={labelOf} />
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th className="num">#</th>
              {group.kind === 'sweep' ? group.swept.map((k) => <th key={k} className="mono">{short(k)}</th>) : <th>Session</th>}
              <th>Status</th><th className="num">{key ?? 'Result'}</th><th>Session</th>
            </tr>
          </thead>
          <tbody>
            {ranked.map((m, i) => (
              <tr key={m.id}>
                <td className="num">{m.headline ? i + 1 : '—'}</td>
                {group.kind === 'sweep'
                  ? group.swept.map((k) => <td key={k} className="mono">{fmtValue(m.group_params[k])}</td>)
                  : <td>{name(m.model_id)}{m.mode === 'evaluate' && <span className="muted small"> · eval</span>}</td>}
                <td>{['running', 'queued'].includes(m.status) ? <div style={{ minWidth: 90 }}><Progress value={m.progress} /></div> : <StatusBadge status={m.status} />}</td>
                <td className="num" style={i === 0 && m.headline ? { fontWeight: 700 } : undefined}>{fmtHead(m.headline)}</td>
                <td><Link to={`/sessions/${m.id}`}>{m.name}</Link></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function fmtHead(h: Headline | null) {
  if (!h) return '—'
  return <>{fmtMetric(h.key, h.mean)}{h.n > 1 && <span className="muted small"> ± {fmtMetric(h.key, h.std)}</span>}</>
}

function GroupChart({ group, ranked, labelOf }: { group: Group; ranked: GroupMember[]; labelOf: (m: GroupMember) => string }) {
  const theme = useTheme()
  const withH = ranked.filter((m) => m.headline)
  if (!withH.length) return <div className="muted small">The chart appears once members complete.</div>
  const key = withH[0].headline!.key
  const numeric = (k: string) => group.sessions.every((m) => typeof m.group_params[k] === 'number')
  const height = 300

  // one numeric swept parameter -> metric vs value
  if (group.kind === 'sweep' && group.swept.length === 1 && numeric(group.swept[0])) {
    const k = group.swept[0]
    const pts = [...withH].sort((a, b) => (a.group_params[k] as number) - (b.group_params[k] as number))
    const xs = pts.map((m) => m.group_params[k] as number)
    const logX = xs.every((x) => x > 0) && Math.max(...xs) / Math.min(...xs) >= 20
    const layout = { ...baseLayout(theme, short(k), key), height, hovermode: 'closest' as const, showlegend: false }
    layout.xaxis = { ...layout.xaxis, type: logX ? 'log' : 'linear', tickvals: xs, ticktext: xs.map((x) => fmtValue(x)) }
    return (
      <div style={{ height }}>
        <Plot useResizeHandler style={{ width: '100%', height: '100%' }} config={{ displaylogo: false, responsive: true }} layout={layout}
          data={[{ x: xs, y: pts.map((m) => m.headline!.mean), error_y: { type: 'data', array: pts.map((m) => m.headline!.std), visible: true, color: theme.series[0] },
            type: 'scatter', mode: 'lines+markers', line: { color: theme.series[0], width: 2 }, marker: { size: 9, color: theme.series[0], line: { color: theme.surface, width: 2 } },
            hovertemplate: `${short(k)}=%{x}<br>${key}: %{y:.4g}<extra></extra>` }]} />
      </div>
    )
  }
  // two swept parameters -> heatmap
  if (group.kind === 'sweep' && group.swept.length === 2) {
    const [a, b] = group.swept
    const xa = [...new Set(group.sessions.map((m) => fmtValue(m.group_params[a])))]
    const yb = [...new Set(group.sessions.map((m) => fmtValue(m.group_params[b])))]
    const z = yb.map((yv) => xa.map((xv) => withH.find((m) => fmtValue(m.group_params[a]) === xv && fmtValue(m.group_params[b]) === yv)?.headline?.mean ?? null))
    const hib = withH[0].headline!.higher_is_better
    const layout = { ...baseLayout(theme, short(a), short(b)), height, hovermode: 'closest' as const }
    layout.xaxis = { ...layout.xaxis, type: 'category' }
    layout.yaxis = { ...layout.yaxis, type: 'category' }
    return (
      <div style={{ height }}>
        <Plot useResizeHandler style={{ width: '100%', height: '100%' }} config={{ displaylogo: false, responsive: true }} layout={layout}
          data={[{ type: 'heatmap', x: xa, y: yb, z, colorscale: [[0, '#cde2fb'], [1, '#104281']], reversescale: !hib,
            texttemplate: '%{z:.4g}', hovertemplate: `${short(a)}=%{x}, ${short(b)}=%{y}<br>${key}: %{z:.4g}<extra></extra>`,
            colorbar: { thickness: 10, tickfont: { color: theme.muted, size: 10 } } } as Plotly.Data]} />
      </div>
    )
  }
  // otherwise: ranked horizontal bars
  const bars = [...withH].reverse()
  const h = Math.max(160, 40 + bars.length * 28)
  const layout = { ...baseLayout(theme, key), height: h, hovermode: 'closest' as const, showlegend: false, margin: { l: 10, r: 20, t: 10, b: 40 } }
  layout.yaxis = { ...layout.yaxis, automargin: true, type: 'category' }
  return (
    <div style={{ height: h }}>
      <Plot useResizeHandler style={{ width: '100%', height: '100%' }} config={{ displaylogo: false, responsive: true }} layout={layout}
        data={[{ type: 'bar', orientation: 'h', y: bars.map(labelOf), x: bars.map((m) => m.headline!.mean),
          error_x: { type: 'data', array: bars.map((m) => m.headline!.std), visible: true, color: theme.text2 },
          marker: { color: theme.series[0] }, hovertemplate: `%{y}<br>${key}: %{x:.4g}<extra></extra>` }]} />
    </div>
  )
}
