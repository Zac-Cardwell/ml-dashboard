import Plotly from 'plotly.js-dist-min'
import createPlotlyComponent from 'react-plotly.js/factory'
import { useEffect, useMemo, useState } from 'react'
import type { SeriesByRun } from '../api'
import { aggregate, smooth, SERIES_VARS } from '../metrics'

export const Plot = createPlotlyComponent(Plotly)

export type Theme = { surface: string; text: string; text2: string; muted: string; grid: string; axis: string; series: string[] }

function readTheme(): Theme {
  const cs = getComputedStyle(document.documentElement)
  const v = (n: string) => cs.getPropertyValue(n).trim()
  return { surface: v('--surface'), text: v('--text'), text2: v('--text-2'), muted: v('--muted'), grid: v('--grid'),
    axis: v('--axis'), series: SERIES_VARS.map(v) }
}

export function useTheme(): Theme {
  const [theme, setTheme] = useState(readTheme)
  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    const on = () => setTheme(readTheme())
    mq.addEventListener('change', on)
    return () => mq.removeEventListener('change', on)
  }, [])
  return theme
}

export function withAlpha(hex: string, a: number) {
  const h = hex.replace('#', '')
  const n = parseInt(h.length === 3 ? h.split('').map((c) => c + c).join('') : h, 16)
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`
}

export const CHART_HEIGHT = 280

// Note: Plotly mutates the layout object it is given, so always pass a fresh copy and never read sizes back from it.
export function baseLayout(theme: Theme, xTitle: string, yTitle?: string): Partial<Plotly.Layout> {
  const axis = { gridcolor: theme.grid, linecolor: theme.axis, zerolinecolor: theme.axis, tickfont: { color: theme.muted, size: 11 },
    title: { font: { color: theme.text2, size: 11 } } }
  return {
    autosize: true, height: CHART_HEIGHT, margin: { l: 52, r: 12, t: 30, b: 40 },
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'system-ui, -apple-system, Segoe UI, sans-serif', color: theme.text2, size: 12 },
    hovermode: 'x unified', hoverlabel: { bgcolor: theme.surface, bordercolor: theme.axis, font: { color: theme.text } },
    legend: { orientation: 'h', y: 1.12, x: 0, font: { size: 11, color: theme.text2 } },
    xaxis: { ...axis, title: { ...axis.title, text: xTitle } },
    yaxis: { ...axis, title: { ...axis.title, text: yTitle ?? '' } },
  }
}

export type Line = { label: string; color: string; runs: { seed: number; steps: number[]; values: number[] }[] }

/**
 * One chart = several lines (e.g. train/val/test of one metric, or several sessions).
 * Each line with >1 run shows the seed mean with a ±1 std band; individual seeds optional.
 */
export function LineChart({ lines, xTitle, showSeeds, smoothing, logY, legend = true }: {
  lines: Line[]; xTitle: string; showSeeds: boolean; smoothing: number; logY?: boolean; legend?: boolean
}) {
  const theme = useTheme()
  const data = useMemo(() => {
    const traces: Plotly.Data[] = []
    for (const line of lines) {
      const runs = line.runs.filter((r) => r.steps.length)
      if (!runs.length) continue
      const single = runs.length === 1
      if (showSeeds && !single) {
        runs.forEach((r) =>
          traces.push({ x: r.steps, y: smooth(r.values, smoothing), type: 'scatter', mode: 'lines', name: `${line.label} · seed ${r.seed}`,
            line: { color: withAlpha(line.color, 0.4), width: 1 }, legendgroup: line.label, showlegend: false,
            hovertemplate: `seed ${r.seed}: %{y:.4g}<extra></extra>` }),
        )
      }
      if (single) {
        const r = runs[0]
        const onePoint = r.steps.length === 1
        traces.push({ x: r.steps, y: smooth(r.values, smoothing), type: 'scatter', mode: onePoint ? 'markers' : 'lines', name: line.label,
          line: { color: line.color, width: 2 }, marker: { size: 9, color: line.color }, legendgroup: line.label,
          hovertemplate: `${line.label}: %{y:.4g}<extra></extra>` })
        continue
      }
      const agg = aggregate(runs)
      if (!agg) continue
      const mean = smooth(agg.mean, smoothing)
      const upper = mean.map((m, i) => m + agg.std[i])
      const lower = mean.map((m, i) => m - agg.std[i])
      traces.push({ x: agg.steps, y: upper, type: 'scatter', mode: 'lines', line: { width: 0 }, hoverinfo: 'skip', showlegend: false,
        legendgroup: line.label })
      traces.push({ x: agg.steps, y: lower, type: 'scatter', mode: 'lines', line: { width: 0 }, fill: 'tonexty',
        fillcolor: withAlpha(line.color, 0.16), hoverinfo: 'skip', showlegend: false, legendgroup: line.label })
      traces.push({ x: agg.steps, y: mean, type: 'scatter', mode: agg.steps.length === 1 ? 'markers' : 'lines',
        name: `${line.label} (mean of ${agg.n})`, line: { color: line.color, width: 2 }, marker: { size: 9, color: line.color },
        legendgroup: line.label, customdata: agg.std, hovertemplate: `${line.label}: %{y:.4g} ± %{customdata:.3g}<extra></extra>` })
    }
    return traces
  }, [lines, showSeeds, smoothing])

  const layout = useMemo(() => {
    const l = baseLayout(theme, xTitle)
    if (logY) l.yaxis = { ...l.yaxis, type: 'log' }
    if (!legend) { l.showlegend = false; l.margin = { ...l.margin, t: 12 } }
    return l
  }, [theme, xTitle, logY, legend])

  if (!data.length) return <div className="empty small">No data yet</div>
  return (
    <div style={{ height: CHART_HEIGHT }}>
      <Plot data={data} layout={{ ...layout }} useResizeHandler style={{ width: '100%', height: '100%' }}
        config={{ displaylogo: false, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d'] }} />
    </div>
  )
}

/** Turn per-run series for some metric keys into chart lines colored by key (split). */
export function linesForKeys(keys: string[], series: SeriesByRun, seedOf: Record<string, number>, theme: Theme): Line[] {
  const SPLIT_COLOR: Record<string, number> = { train: 0, val: 1, test: 2, eval: 1, rollout: 0, episode: 0, final: 2 }
  const used = new Set<number>()
  return keys.map((k, i) => {
    const split = k.split('/')[0]
    let ci = SPLIT_COLOR[split] ?? -1
    if (ci < 0 || used.has(ci)) ci = [0, 1, 2, 3, 4, 5, 6, 7].find((c) => !used.has(c)) ?? i % 8
    used.add(ci)
    return {
      label: k,
      color: theme.series[ci],
      runs: Object.entries(series).filter(([, s]) => s[k]).map(([rid, s]) => ({ seed: seedOf[rid] ?? 0, ...s[k] })),
    }
  })
}

export function Heatmap({ labels, matrix, title }: { labels: string[]; matrix: number[][]; title?: string }) {
  const theme = useTheme()
  const max = Math.max(1, ...matrix.flat())
  const height = Math.min(560, 160 + labels.length * 30)
  const layout = {
    ...baseLayout(theme, 'Predicted', 'True'), hovermode: 'closest' as const, title: title ? { text: title, font: { size: 13 } } : undefined,
    height, margin: { l: 90, r: 10, t: title ? 40 : 14, b: 70 },
  }
  layout.yaxis = { ...layout.yaxis, autorange: 'reversed', type: 'category' }
  layout.xaxis = { ...layout.xaxis, type: 'category' }
  return (
    <div style={{ height }}><Plot
      data={[{ z: matrix, x: labels, y: labels, type: 'heatmap', colorscale: [[0, withAlpha(theme.series[0], 0.04)], [1, theme.series[0]]],
        zmin: 0, zmax: max, showscale: false, hovertemplate: 'true %{y} → pred %{x}: %{z}<extra></extra>',
        text: matrix.map((r) => r.map(String)) as any, texttemplate: labels.length <= 15 ? '%{text}' : '', textfont: { size: 11 } } as Plotly.Data]}
      layout={layout} useResizeHandler style={{ width: '100%', height: '100%' }} config={{ displaylogo: false, responsive: true }}
    /></div>
  )
}
