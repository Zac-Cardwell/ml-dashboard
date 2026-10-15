import { useEffect, useState } from 'react'
import { ErrorBox } from '../components/ui'
import { colsOf } from './DataGrid'
import { studio, type ChartResult, type ChartSpec, type Dataset } from './types'
import { ChartView } from './viz'

/** Which fields each chart type uses, and what it's for (shown as guidance). */
export const CHART_TYPES: { type: string; label: string; fields: string[]; help: string }[] = [
  { type: 'histogram', label: 'Histogram', fields: ['x', 'color', 'bins'], help: 'Distribution of one numeric column. Several datasets/groups share bin edges and are normalized to shares so different sizes compare fairly.' },
  { type: 'ecdf', label: 'ECDF', fields: ['x', 'color'], help: 'Cumulative distribution — the most honest way to compare distributions (no binning choices).' },
  { type: 'box', label: 'Box plot', fields: ['x', 'color'], help: 'Quartiles and mean per group; whiskers at the 5th and 95th percentiles.' },
  { type: 'bar', label: 'Bar', fields: ['x', 'y', 'agg', 'color', 'top'], help: 'Count or aggregate of a measure per category (top categories, rest dropped).' },
  { type: 'scatter', label: 'Scatter', fields: ['x', 'y', 'color'], help: 'Relationship between two numeric columns (random sample of points).' },
  { type: 'line', label: 'Line (over time)', fields: ['x', 'y', 'agg', 'bucket', 'color'], help: 'An aggregate per time bucket.' },
  { type: 'heatmap', label: 'Heatmap', fields: ['x', 'y', 'agg', 'value'], help: 'Count or aggregate for each pair of two categorical columns.' },
]
const AGGS = ['count', 'sum', 'avg', 'median', 'min', 'max']

/** Chart builder. With extra datasets selected, each dataset becomes a series (subsets compared in one plot). */
export function ChartBuilder({ base, list, initial, lockDatasets }: { base: Dataset[]; list: Dataset[]; initial?: ChartSpec; lockDatasets?: boolean }) {
  const [spec, setSpec] = useState<ChartSpec>(initial ?? { type: 'histogram', x: colsOf(base[0]?.schema ?? [], true)[0] ?? colsOf(base[0]?.schema ?? [])[0], agg: 'count', bins: 40 })
  const [extra, setExtra] = useState<string[]>([])
  const [res, setRes] = useState<ChartResult>()
  const [err, setErr] = useState<unknown>()
  const [busy, setBusy] = useState(false)
  const ids = [...base.map((b) => b.id), ...extra]
  const meta = CHART_TYPES.find((t) => t.type === spec.type)!
  const schema = base[0]?.schema ?? []
  const all = colsOf(schema)
  const num = colsOf(schema, true)
  const set = (k: keyof ChartSpec, v: any) => setSpec({ ...spec, [k]: v === '' ? undefined : v })
  const key = JSON.stringify([spec, ids])
  useEffect(() => {
    if (!spec.x) return
    if (['scatter', 'heatmap'].includes(spec.type) && !spec.y) return
    const t = setTimeout(() => {
      setBusy(true); setErr(undefined)
      studio.chart(ids, spec).then(setRes).catch(setErr).finally(() => setBusy(false))
    }, 300)
    return () => clearTimeout(t)
  }, [key]) // eslint-disable-line react-hooks/exhaustive-deps
  const xOptions = ['histogram', 'ecdf', 'box', 'scatter'].includes(spec.type) ? num : all
  const yOptions = spec.type === 'scatter' ? num : spec.type === 'heatmap' ? all : num
  const needY = spec.type === 'scatter' || spec.type === 'heatmap' || (spec.agg && spec.agg !== 'count')
  return (
    <div className="chart-builder">
      <div className="card card-pad col chart-controls" style={{ gap: 10 }}>
        <label className="field"><span className="lbl">Chart</span>
          <select value={spec.type} onChange={(e) => set('type', e.target.value)}>{CHART_TYPES.map((t) => <option key={t.type} value={t.type}>{t.label}</option>)}</select></label>
        <div className="small muted">{meta.help}</div>
        <label className="field"><span className="lbl">{spec.type === 'line' ? 'Time column' : spec.type === 'box' ? 'Value' : 'X'}</span>
          <select value={spec.x ?? ''} onChange={(e) => set('x', e.target.value)}><option value="">—</option>{xOptions.map((c) => <option key={c}>{c}</option>)}</select></label>
        {meta.fields.includes('agg') && <label className="field"><span className="lbl">Aggregate</span>
          <select value={spec.agg ?? 'count'} onChange={(e) => set('agg', e.target.value)}>{AGGS.map((a) => <option key={a}>{a}</option>)}</select></label>}
        {meta.fields.includes('y') && (spec.type !== 'bar' && spec.type !== 'line' || needY) && <label className="field"><span className="lbl">{spec.type === 'heatmap' ? 'Y (category)' : spec.type === 'scatter' ? 'Y' : 'Measure'}</span>
          <select value={spec.y ?? ''} onChange={(e) => set('y', e.target.value)}><option value="">—</option>{yOptions.map((c) => <option key={c}>{c}</option>)}</select></label>}
        {spec.type === 'heatmap' && spec.agg && spec.agg !== 'count' && <label className="field"><span className="lbl">Measure</span>
          <select value={spec.value ?? ''} onChange={(e) => set('value', e.target.value)}><option value="">—</option>{num.map((c) => <option key={c}>{c}</option>)}</select></label>}
        {meta.fields.includes('color') && <label className="field"><span className="lbl">Group / color by</span>
          <select value={spec.color ?? ''} disabled={ids.length > 1} onChange={(e) => set('color', e.target.value)}><option value="">—</option>{all.map((c) => <option key={c}>{c}</option>)}</select>
          {ids.length > 1 && <span className="help">Comparing datasets: each dataset is a series</span>}</label>}
        {meta.fields.includes('bins') && <label className="field"><span className="lbl">Bins</span><input type="number" min={5} max={200} value={spec.bins ?? 40} onChange={(e) => set('bins', parseInt(e.target.value) || 40)} /></label>}
        {meta.fields.includes('bucket') && <label className="field"><span className="lbl">Bucket</span>
          <select value={spec.bucket ?? 'day'} onChange={(e) => set('bucket', e.target.value)}>{['minute', 'hour', 'day', 'week', 'month', 'quarter', 'year'].map((b) => <option key={b}>{b}</option>)}</select></label>}
        {meta.fields.includes('top') && <label className="field"><span className="lbl">Top categories</span><input type="number" min={2} max={100} value={spec.top ?? 20} onChange={(e) => set('top', parseInt(e.target.value) || 20)} /></label>}
        {(spec.type === 'histogram' || spec.type === 'bar') && <label className="check small"><input type="checkbox" checked={spec.normalize ?? ids.length > 1} onChange={(e) => set('normalize', e.target.checked)} /> Show shares instead of counts</label>}
        <label className="field"><span className="lbl">Filter (SQL WHERE)</span>
          <input className="mono" placeholder="optional" value={spec.filter ?? ''} onChange={(e) => set('filter', e.target.value)} /></label>
        {!lockDatasets && <div className="field"><span className="lbl">Compare with datasets</span>
          <div className="chips">{list.filter((d) => d.version > 0 && !base.some((b) => b.id === d.id)).map((d) => {
            const on = extra.includes(d.id)
            return <button key={d.id} type="button" className={`chip toggle${on ? ' on' : ''}`} onClick={() => setExtra(on ? extra.filter((x) => x !== d.id) : [...extra, d.id])}>{d.name}</button>
          })}</div><span className="help">Each selected dataset is overlaid as its own series (the column must exist in each).</span></div>}
      </div>
      <div className="card card-pad" style={{ minWidth: 0 }}>
        <div className="row small" style={{ justifyContent: 'space-between', marginBottom: 6 }}>
          <span className="muted">{busy ? 'Computing over all rows…' : res?.note ?? ''}</span>
        </div>
        <ErrorBox error={err} />
        {res && res.series.length > 0 ? <ChartView result={res} height={460} /> : !err && <div className="empty">Pick columns to plot.</div>}
      </div>
    </div>
  )
}

export function ChartsTab({ ds, list }: { ds: Dataset; list: Dataset[] }) {
  return <ChartBuilder key={ds.id + ds.version} base={[ds]} list={list} />
}
