import { baseLayout, Plot, useTheme, withAlpha } from '../components/Chart'
import type { ChartResult } from './types'

const cfg = { displaylogo: false, responsive: true }
const centers = (edges: number[]) => edges.slice(0, -1).map((e, i) => (e + edges[i + 1]) / 2)

/** Render a /studio/chart result. Several series = several datasets (or color groups), in the fixed palette order. */
export function ChartView({ result, height = 340 }: { result: ChartResult; height?: number }) {
  const theme = useTheme()
  const c = (i: number) => theme.series[i % theme.series.length]
  const many = result.series.length > 1
  const yTitle = result.yTitle ?? (result.normalized ? 'share of rows' : result.agg && result.agg !== 'count' ? `${result.agg}(${result.y ?? ''})` : 'rows')
  let data: any[] = []
  const layout: any = { ...baseLayout(theme, result.x ?? '', yTitle), height, showlegend: many }
  switch (result.type) {
    case 'histogram':
      data = result.series.map((s, i) => ({ type: 'bar', x: centers(s.edges), y: s.values, name: `${s.name} (n=${s.n.toLocaleString()})`,
        marker: { color: many ? withAlpha(c(i), 0.55) : c(i), line: { width: 0 } }, hovertemplate: `%{x:.4g}: %{y:${result.normalized ? '.2%' : ','}}<extra>${s.name}</extra>` }))
      layout.barmode = 'overlay'; layout.bargap = 0.02
      if (result.normalized) layout.yaxis = { ...layout.yaxis, tickformat: '.0%' }
      break
    case 'ecdf':
      data = result.series.map((s, i) => ({ type: 'scatter', mode: 'lines', x: s.x, y: s.y, name: `${s.name} (n=${s.n.toLocaleString()})`, line: { color: c(i), width: 2, shape: 'hv' } }))
      layout.yaxis = { ...layout.yaxis, tickformat: '.0%', title: { text: 'share of rows ≤ x' } }
      break
    case 'box':
      data = result.series.map((s, i) => ({ type: 'box', name: s.name, x: [s.name], q1: [s.q[1]], median: [s.q[2]], q3: [s.q[3]], lowerfence: [s.q[0]], upperfence: [s.q[4]],
        mean: [s.mean], marker: { color: c(i) }, line: { color: c(i), width: 1.5 }, fillcolor: withAlpha(c(i), 0.18), boxmean: true,
        hovertemplate: `n=${s.n?.toLocaleString()}<extra>${s.name}</extra>` }))
      layout.yaxis = { ...layout.yaxis, title: { text: `${result.y || result.x} (whiskers: 5th–95th percentile)` } }
      layout.xaxis = { ...layout.xaxis, title: { text: '' } }
      layout.hovermode = 'closest'
      break
    case 'bar':
      data = result.series.map((s, i) => ({ type: 'bar', x: s.categories, y: s.values, name: s.name, marker: { color: c(i) },
        hovertemplate: `%{x}: %{y:${result.normalized ? '.2%' : ',.4g'}}<extra>${s.name}</extra>` }))
      layout.barmode = 'group'; layout.bargap = 0.15; layout.bargroupgap = 0.04
      layout.xaxis = { ...layout.xaxis, type: 'category' }
      if (result.normalized) layout.yaxis = { ...layout.yaxis, tickformat: '.0%' }
      break
    case 'scatter':
      data = result.series.map((s, i) => ({ type: 'scattergl', mode: 'markers', x: s.x, y: s.y, name: s.name,
        marker: { color: withAlpha(c(i), 0.55), size: 5, line: { width: 0 } } }))
      layout.yaxis = { ...layout.yaxis, title: { text: result.y } }
      layout.hovermode = 'closest'
      break
    case 'line':
      data = result.series.map((s, i) => ({ type: 'scatter', mode: 'lines', x: s.x, y: s.y, name: s.name, line: { color: c(i), width: 2 } }))
      break
    case 'heatmap': {
      const s = result.series[0]
      data = [{ type: 'heatmap', x: s.xs, y: s.ys, z: s.z, colorscale: [[0, withAlpha(theme.series[0], 0.05)], [1, theme.series[0]]],
        hovertemplate: '%{x} · %{y}: %{z:,.4g}<extra></extra>', colorbar: { thickness: 10, outlinewidth: 0, tickfont: { color: theme.muted } } }]
      layout.xaxis = { ...layout.xaxis, type: 'category' }
      layout.yaxis = { ...layout.yaxis, type: 'category', title: { text: result.y } }
      layout.hovermode = 'closest'
      break
    }
  }
  return (
    <div style={{ height }}>
      <Plot data={data} layout={layout} useResizeHandler style={{ width: '100%', height: '100%' }} config={cfg} />
    </div>
  )
}

/** Diverging matrix (correlations): blue = positive, orange = negative, neutral at 0. */
export function CorrMatrix({ columns, matrix, height }: { columns: string[]; matrix: (number | null)[][]; height?: number }) {
  const theme = useTheme()
  const h = height ?? Math.min(640, 120 + columns.length * 22)
  const layout: any = { ...baseLayout(theme, '', ''), height: h, hovermode: 'closest', margin: { l: 140, r: 20, t: 10, b: 120 } }
  layout.xaxis = { ...layout.xaxis, type: 'category', tickangle: -45 }
  layout.yaxis = { ...layout.yaxis, type: 'category', autorange: 'reversed' }
  return (
    <div style={{ height: h }}>
      <Plot data={[{ type: 'heatmap', x: columns, y: columns, z: matrix, zmin: -1, zmax: 1, zmid: 0,
        colorscale: [[0, theme.series[1]], [0.5, theme.grid], [1, theme.series[0]]],
        hovertemplate: '%{y} × %{x}: r = %{z:.3f}<extra></extra>', colorbar: { thickness: 10, outlinewidth: 0, tickfont: { color: theme.muted } } } as any]}
        layout={layout} useResizeHandler style={{ width: '100%', height: '100%' }} config={cfg} />
    </div>
  )
}

/** Tiny inline distribution for table cells. */
export function Sparkbars({ counts, width = 96, height = 22 }: { counts: number[]; width?: number; height?: number }) {
  const max = Math.max(1, ...counts)
  const w = width / Math.max(counts.length, 1)
  return (
    <svg width={width} height={height} aria-hidden="true" style={{ display: 'block' }}>
      {counts.map((c, i) => {
        const h = Math.max(c > 0 ? 1 : 0, (c / max) * (height - 1))
        return <rect key={i} x={i * w} y={height - h} width={Math.max(w - 1, 1)} height={h} rx={0.5} fill="var(--s1)" opacity={0.85} />
      })}
    </svg>
  )
}

/** Horizontal share bars (top categories) for table cells. */
export function TopBars({ items, width = 140 }: { items: { value: string; share: number }[]; width?: number }) {
  return (
    <div className="col" style={{ gap: 2, width }}>
      {items.slice(0, 3).map((t) => (
        <div key={t.value} className="row small" style={{ gap: 4, flexWrap: 'nowrap' }} title={`${t.value}: ${(t.share * 100).toFixed(1)}%`}>
          <span style={{ width: Math.max(2, t.share * 60), height: 6, background: 'var(--s1)', borderRadius: 2, flex: 'none' }} />
          <span className="muted" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{t.value}</span>
        </div>
      ))}
    </div>
  )
}

/** A simple labeled bar chart (values per label), e.g. missing share or mutual information. */
export function HBars({ labels, values, format = '.3f', height, title }: { labels: string[]; values: number[]; format?: string; height?: number; title?: string }) {
  const theme = useTheme()
  const h = height ?? Math.max(140, 40 + labels.length * 22)
  const layout: any = { ...baseLayout(theme, title ?? '', ''), height: h, hovermode: 'closest', showlegend: false, margin: { l: 10, r: 50, t: 8, b: 36 } }
  layout.yaxis = { ...layout.yaxis, type: 'category', autorange: 'reversed', automargin: true }
  if (format.includes('%')) layout.xaxis = { ...layout.xaxis, tickformat: '.0%' }
  return (
    <div style={{ height: h }}>
      <Plot data={[{ type: 'bar', orientation: 'h', y: labels, x: values, marker: { color: theme.series[0] }, hovertemplate: `%{y}: %{x:${format}}<extra></extra>` } as any]}
        layout={layout} useResizeHandler style={{ width: '100%', height: '100%' }} config={cfg} />
    </div>
  )
}

/** 2-D embedding colored by a label (clusters / anomaly flag / class). */
export function EmbeddingPlot({ x, y, label, height = 380, title }: { x: number[]; y: number[]; label?: (string | number)[] | null; height?: number; title?: string }) {
  const theme = useTheme()
  const groups = new Map<string, number[]>()
  x.forEach((_, i) => { const k = label ? String(label[i]) : 'rows'; if (!groups.has(k)) groups.set(k, []); groups.get(k)!.push(i) })
  const keys = [...groups.keys()].sort((a, b) => (Number(a) - Number(b)) || a.localeCompare(b))
  const data = keys.map((k, gi) => {
    const idx = groups.get(k)!
    const noise = k === '-1'
    return { type: 'scattergl', mode: 'markers', name: noise ? 'noise' : k, x: idx.map((i) => x[i]), y: idx.map((i) => y[i]),
      marker: { size: 5, color: noise ? theme.muted : withAlpha(theme.series[gi % theme.series.length], 0.65), line: { width: 0 } } }
  })
  const layout: any = { ...baseLayout(theme, 'component 1', 'component 2'), height, hovermode: 'closest', showlegend: keys.length > 1 && keys.length <= 20,
    title: title ? { text: title, font: { size: 12 } } : undefined }
  return <div style={{ height }}><Plot data={data as any} layout={layout} useResizeHandler style={{ width: '100%', height: '100%' }} config={cfg} /></div>
}

/** Small line chart for sweeps (one metric per chart — never two y-axes). */
export function MiniLine({ x, y, xTitle, yTitle, height = 200, mark }: { x: number[]; y: (number | null)[]; xTitle: string; yTitle: string; height?: number; mark?: number }) {
  const theme = useTheme()
  const layout: any = { ...baseLayout(theme, xTitle, yTitle), height, showlegend: false, margin: { l: 52, r: 12, t: 12, b: 40 } }
  if (mark != null) layout.shapes = [{ type: 'line', x0: mark, x1: mark, yref: 'paper', y0: 0, y1: 1, line: { color: theme.axis, dash: 'dot', width: 1 } }]
  return <div style={{ height }}><Plot data={[{ type: 'scatter', mode: 'lines+markers', x, y, line: { color: theme.series[0], width: 2 }, marker: { size: 7, color: theme.series[0] } } as any]}
    layout={layout} useResizeHandler style={{ width: '100%', height: '100%' }} config={cfg} /></div>
}
