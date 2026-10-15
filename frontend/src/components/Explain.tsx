import { useEffect, useState } from 'react'
import { api, fmtValue } from '../api'
import { baseLayout, Plot, useTheme, withAlpha } from './Chart'
import { ErrorBox } from './ui'

export type ShapGlobal = {
  method: string; explains: string; base_value: number; features: string[]; importance: number[]; direction: (number | null)[]
  beeswarm: Record<string, { shap: number[]; color: number[]; value: unknown[] }>
  examples: { record: Record<string, unknown>; prediction: unknown; output: number; label: unknown }[]
  n: number; n_perm: number
}
export type Waterfall = { explains: string; base_value: number; output: number; prediction: unknown
  contributions: { feature: string; value: unknown; shap: number }[]; n_perm: number }
export type GradCam = { method: string; items: { image: string; overlay: string; prediction: string; confidence: number; label: string | null
  top3: { class: string; p: number }[] }[] }

const fmt = (v: number) => (Math.abs(v) < 0.001 ? v.toExponential(1) : v.toFixed(3))

/** Mean |SHAP| per raw feature. */
export function ImportanceChart({ g }: { g: ShapGlobal }) {
  const theme = useTheme()
  const n = g.features.length
  const h = Math.max(200, 40 + n * 26)
  const layout = { ...baseLayout(theme, `mean |SHAP| — average impact on ${g.explains}`), height: h, hovermode: 'closest' as const,
    showlegend: false, margin: { l: 10, r: 110, t: 10, b: 44 } }
  layout.yaxis = { ...layout.yaxis, automargin: true, type: 'category', autorange: 'reversed' }
  return (
    <div style={{ height: h }}>
      <Plot useResizeHandler style={{ width: '100%', height: '100%' }} config={{ displaylogo: false, responsive: true }} layout={layout}
        data={[{ type: 'bar', orientation: 'h', y: g.features, x: g.importance, marker: { color: theme.series[0] },
          text: g.direction.map((d) => (d == null ? '' : d > 0.2 ? 'higher → raises' : d < -0.2 ? 'higher → lowers' : 'mixed')) as any,
          textposition: 'outside', textfont: { size: 10, color: theme.text2 }, cliponaxis: false,
          hovertemplate: '%{y}: %{x:.4f}<extra></extra>' }]} />
    </div>
  )
}

/** One dot per explained example: x = SHAP value, color = the feature's value (low → high). */
export function Beeswarm({ g }: { g: ShapGlobal }) {
  const theme = useTheme()
  const feats = Object.keys(g.beeswarm)
  const h = Math.max(240, 50 + feats.length * 34)
  const traces: Plotly.Data[] = feats.map((f, i) => {
    const b = g.beeswarm[f]
    const jitter = b.shap.map((_, k) => i + (((k * 7919) % 100) / 100 - 0.5) * 0.55)
    const numeric = b.color.some((c) => c !== null && c >= 0)
    return {
      type: 'scattergl', mode: 'markers', x: b.shap, y: jitter, name: f, showlegend: false,
      marker: numeric
        ? { size: 6, color: b.color.map((c) => (c == null || c < 0 ? 0.5 : c)), colorscale: [[0, '#2a78d6'], [0.5, '#c3c2b7'], [1, '#e34948']], cmin: 0, cmax: 1, opacity: 0.85,
          showscale: i === 0, colorbar: { title: { text: 'feature value', side: 'right' }, tickvals: [0, 1], ticktext: ['low', 'high'], thickness: 10, len: 0.6 } }
        : { size: 6, color: withAlpha(theme.muted, 0.8) },
      customdata: b.value as any, hovertemplate: `${f} = %{customdata}<br>SHAP %{x:.4f}<extra></extra>`,
    } as Plotly.Data
  })
  const layout = { ...baseLayout(theme, `SHAP value (impact on ${g.explains})`), height: h, hovermode: 'closest' as const, margin: { l: 10, r: 60, t: 10, b: 44 } }
  layout.yaxis = { ...layout.yaxis, tickvals: feats.map((_, i) => i), ticktext: feats, automargin: true, autorange: 'reversed', zeroline: false }
  layout.shapes = [{ type: 'line', x0: 0, x1: 0, yref: 'paper', y0: 0, y1: 1, line: { color: theme.axis, width: 1 } }]
  return <div style={{ height: h }}><Plot useResizeHandler style={{ width: '100%', height: '100%' }} config={{ displaylogo: false, responsive: true }} layout={layout} data={traces} /></div>
}

/** How each feature pushed one prediction away from the average prediction. */
export function WaterfallChart({ w, top = 10 }: { w: Waterfall; top?: number }) {
  const theme = useTheme()
  const shown = w.contributions.slice(0, top)
  const rest = w.contributions.slice(top).reduce((a, c) => a + c.shap, 0)
  const items = [...shown, ...(w.contributions.length > top ? [{ feature: `${w.contributions.length - top} other features`, value: '', shap: rest }] : [])]
  const labels = items.map((c) => (c.value === '' ? c.feature : `${c.feature} = ${fmtValue(c.value)}`))
  const h = Math.max(220, 80 + items.length * 28)
  const layout = { ...baseLayout(theme, w.explains), height: h, hovermode: 'closest' as const, showlegend: false, margin: { l: 10, r: 30, t: 10, b: 44 } }
  layout.yaxis = { ...layout.yaxis, automargin: true, type: 'category', autorange: 'reversed' }
  return (
    <div style={{ height: h }}>
      <Plot useResizeHandler style={{ width: '100%', height: '100%' }} config={{ displaylogo: false, responsive: true }} layout={layout}
        data={[{
          type: 'waterfall', orientation: 'h', measure: ['absolute', ...items.map(() => 'relative'), 'total'] as any,
          y: ['average prediction', ...labels, 'this prediction'], x: [w.base_value, ...items.map((c) => c.shap), 0],
          text: [fmt(w.base_value), ...items.map((c) => `${c.shap >= 0 ? '+' : ''}${fmt(c.shap)}`), fmt(w.output)] as any, textposition: 'outside',
          increasing: { marker: { color: theme.series[7] } }, decreasing: { marker: { color: theme.series[0] } }, totals: { marker: { color: theme.muted } },
          connector: { line: { color: theme.axis, width: 1 } }, cliponaxis: false,
        } as Plotly.Data]} />
    </div>
  )
}

export function RecordWaterfall({ path, record, onClose }: { path: string; record?: Record<string, unknown>; onClose?: () => void }) {
  const [w, setW] = useState<Waterfall>()
  const [err, setErr] = useState<unknown>()
  const [busy, setBusy] = useState(false)
  const key = path + JSON.stringify(record ?? null)
  useEffect(() => {
    let alive = true
    setBusy(true); setErr(undefined); setW(undefined)
    ;(record ? api.post<Waterfall>(path, { record }) : api.get<Waterfall>(path))
      .then((r) => alive && setW(r)).catch((e) => alive && setErr(e)).finally(() => alive && setBusy(false))
    return () => { alive = false }
  }, [key]) // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <div className="col" style={{ gap: 8 }}>
      <ErrorBox error={err} />
      {busy && <div className="muted small">Computing Shapley values…</div>}
      {w && <>
        <div className="small text-2">Prediction <b>{fmtValue(w.prediction)}</b> · {w.explains} = <b>{w.output.toFixed(3)}</b> vs. {w.base_value.toFixed(3)} on average.
          Red pushes {w.explains} up, blue pushes it down.</div>
        <WaterfallChart w={w} />
      </>}
      {onClose && <button className="sm" style={{ alignSelf: 'flex-start' }} onClick={onClose}>Close</button>}
    </div>
  )
}

export function GradCamGallery({ g }: { g: GradCam }) {
  return (
    <div className="col" style={{ gap: 8 }}>
      <div className="small text-2">{g.method}. Bright regions influenced the predicted class most.</div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(230px, 1fr))', gap: 12 }}>
        {g.items.map((it, i) => {
          const wrong = it.label && it.label !== it.prediction
          return (
            <figure key={i} className="card" style={{ margin: 0, padding: 8 }}>
              <div className="row" style={{ gap: 4, flexWrap: 'nowrap' }}>
                <img src={`data:image/png;base64,${it.image}`} alt="input" style={{ width: '50%', imageRendering: 'pixelated', borderRadius: 4 }} />
                <img src={`data:image/png;base64,${it.overlay}`} alt="Grad-CAM" style={{ width: '50%', imageRendering: 'pixelated', borderRadius: 4 }} />
              </div>
              <figcaption className="small" style={{ marginTop: 6 }}>
                <b style={{ color: wrong ? 'var(--critical-text)' : 'var(--good-text)' }}>{wrong ? '✗' : '✓'} {it.prediction}</b> {(it.confidence * 100).toFixed(0)}%
                {it.label && wrong && <span className="muted"> · true: {it.label}</span>}
                <div className="muted">{it.top3.slice(1).map((t) => `${t.class} ${(t.p * 100).toFixed(0)}%`).join(' · ')}</div>
              </figcaption>
            </figure>
          )
        })}
      </div>
    </div>
  )
}
