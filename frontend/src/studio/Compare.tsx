import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { ErrorBox } from '../components/ui'
import { colsOf } from './DataGrid'
import { fmtInt, isNumericType, pct, short, studio, type ChartResult, type Dataset } from './types'
import { ChartView } from './viz'

const effectLabel = (d: number | null) => (d == null ? '' : Math.abs(d) < 0.2 ? 'negligible' : Math.abs(d) < 0.5 ? 'small' : Math.abs(d) < 0.8 ? 'medium' : 'large')
const cramerLabel = (v: number | null) => (v == null ? '' : v < 0.1 ? 'negligible' : v < 0.3 ? 'small' : v < 0.5 ? 'medium' : 'large')
const pFmt = (p: number | null) => (p == null ? '—' : p < 1e-4 ? '< 0.0001' : p.toFixed(4))

/** Compare one column across several datasets/subsets in one plot, with sizes, intervals, tests and effect sizes. */
export function CompareView({ list }: { list: Dataset[] }) {
  const [params, setParams] = useSearchParams()
  const ready = list.filter((d) => d.version > 0)
  const [ids, setIds] = useState<string[]>(() => (params.get('ds') ?? '').split(',').filter(Boolean))
  const [column, setColumn] = useState(params.get('col') ?? '')
  const [type, setType] = useState('auto')
  const [filter, setFilter] = useState('')
  const [applied, setApplied] = useState('')
  const [chart, setChart] = useState<ChartResult>()
  const [stats, setStats] = useState<any>()
  const [err, setErr] = useState<unknown>()
  const [busy, setBusy] = useState(false)
  const chosen = ids.map((i) => ready.find((d) => d.id === i)).filter(Boolean) as Dataset[]
  const common = useMemo(() => {
    if (!chosen.length) return []
    const sets = chosen.map((d) => new Set(colsOf(d.schema)))
    return colsOf(chosen[0].schema).filter((c) => sets.every((s) => s.has(c)))
  }, [ids.join(), ready.length]) // eslint-disable-line react-hooks/exhaustive-deps
  const colType = chosen[0]?.schema.find((c) => c.name === column)?.type
  const numeric = isNumericType(colType)
  const chartType = type === 'auto' ? (numeric ? 'histogram' : 'bar') : type

  useEffect(() => { setParams({ ...(ids.length ? { ds: ids.join(',') } : {}), ...(column ? { col: column } : {}) }, { replace: true }) }, [ids.join(), column]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (common.length && column && !common.includes(column)) setColumn(common[0]) }, [common.join()]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (chosen.length < 2 || !column) { setChart(undefined); setStats(undefined); return }
    setBusy(true); setErr(undefined)
    const f = applied || undefined
    Promise.all([studio.chart(ids, { type: chartType, x: column, filter: f, normalize: true }), studio.compare(ids, column, f)])
      .then(([c, s]) => { setChart(c); setStats(s) }).catch(setErr).finally(() => setBusy(false))
  }, [ids.join(), column, chartType, applied]) // eslint-disable-line react-hooks/exhaustive-deps

  const toggle = (id: string) => setIds(ids.includes(id) ? ids.filter((x) => x !== id) : [...ids, id])
  return (
    <div className="col" style={{ gap: 14 }}>
      <div className="page-head" style={{ marginBottom: 0 }}>
        <div><h1>Compare subsets</h1>
          <div className="sub">One column across datasets — distributions overlaid on shared bins and normalized to shares, with tests against the first (reference) dataset.</div></div>
      </div>
      <div className="card card-pad col" style={{ gap: 10 }}>
        <div className="field"><span className="lbl">Datasets (in order — the first is the reference)</span>
          <div className="chips">{ready.map((d) => {
            const i = ids.indexOf(d.id)
            return <button key={d.id} type="button" className={`chip toggle${i >= 0 ? ' on' : ''}`} onClick={() => toggle(d.id)}>
              {i >= 0 && <span className="chip-n" style={{ background: `var(--s${(i % 8) + 1})` }}>{i + 1}</span>}{d.name}<span className="muted small"> {fmtInt(d.rows)}</span></button>
          })}</div></div>
        <div className="row" style={{ alignItems: 'flex-end' }}>
          <label className="field" style={{ minWidth: 220 }}><span className="lbl">Column{chosen.length > 1 ? ` (${common.length} shared)` : ''}</span>
            <select value={column} onChange={(e) => setColumn(e.target.value)} disabled={!common.length}><option value="">—</option>{common.map((c) => <option key={c}>{c}</option>)}</select></label>
          <label className="field"><span className="lbl">Plot</span>
            <select value={type} onChange={(e) => setType(e.target.value)}>
              <option value="auto">auto</option>{numeric && <><option value="histogram">histogram</option><option value="ecdf">ECDF</option><option value="box">box</option></>}<option value="bar">bar (categories)</option></select></label>
          <form className="field" style={{ flex: 1, minWidth: 240 }} onSubmit={(e) => { e.preventDefault(); setApplied(filter.trim()) }}><span className="lbl">Filter applied to every dataset (SQL WHERE)</span>
            <input className="mono" placeholder="optional, Enter to apply" value={filter} onChange={(e) => setFilter(e.target.value)} /></form>
        </div>
      </div>
      <ErrorBox error={err} />
      {chosen.length < 2 ? <div className="card empty">Pick at least two datasets. Make subsets with a Filter recipe, a SQL query (“Save result as dataset”), or by saving a cluster in Explore.</div>
        : !column ? <div className="card empty">{common.length ? 'Pick a column.' : 'These datasets share no columns.'}</div> : (
          <>
            <div className="card card-pad">
              <div className="small muted" style={{ marginBottom: 4 }}>{busy ? 'Computing…' : chartType === 'histogram' || chartType === 'bar' ? 'Shares within each dataset, so different sizes compare fairly.' : ''}</div>
              {chart && <ChartView result={chart} height={420} />}
            </div>
            {stats?.warnings?.map((w: string, i: number) => <div key={i} className="callout warning small"><b>Not comparable as-is:</b> {w}</div>)}
            {stats && <StatsTable s={stats} />}
          </>
        )}
    </div>
  )
}

function StatsTable({ s }: { s: any }) {
  const test = (name: string) => s.tests.find((t: any) => t.dataset === name)
  return (
    <div className="card">
      <div className="card-head"><h3>Statistics</h3><span className="small muted mono">{s.column}</span></div>
      <div className="table-wrap">
        {s.numeric ? (
          <table className="dense"><thead><tr><th>Dataset</th><th className="num">n</th><th className="num">Missing</th><th className="num">Mean ± 95% CI</th><th className="num">Std</th>
            <th className="num">Median [IQR]</th><th className="num">Δ mean</th><th className="num">Cohen's d</th><th className="num">Welch t p</th><th className="num">KS p</th></tr></thead>
            <tbody>{s.rows.map((r: any, i: number) => {
              const t = test(r.dataset)
              if (r.missing_column) return <tr key={r.dataset}><td>{r.dataset}</td><td colSpan={9} className="muted">column missing</td></tr>
              return <tr key={r.dataset}><td><span className="swatch" style={{ background: `var(--s${(i % 8) + 1})` }} />{r.dataset}{i === 0 && <span className="tag" style={{ marginLeft: 6 }}>reference</span>}</td>
                <td className="num">{fmtInt(r.n)}</td><td className="num">{fmtInt(r.missing)}</td><td className="num">{short(r.mean)} ± {short(r.ci95)}</td><td className="num">{short(r.std)}</td>
                <td className="num">{short(r.median)} [{short(r.q1)}, {short(r.q3)}]</td>
                <td className="num">{t ? short(t.mean_diff) : ''}</td>
                <td className="num">{t ? <>{short(t.effect_size_d)} <span className="muted small">{effectLabel(t.effect_size_d)}</span></> : ''}</td>
                <td className="num">{t ? pFmt(t.welch_t_p) : ''}</td><td className="num">{t ? pFmt(t.ks_p) : ''}</td></tr>
            })}</tbody></table>
        ) : (
          <table className="dense"><thead><tr><th>Dataset</th><th className="num">n</th>{s.categories.map((c: string) => <th key={c} className="num">{c}</th>)}<th className="num">other</th>
            <th className="num">χ² p</th><th className="num">Cramér's V</th></tr></thead>
            <tbody>{s.rows.map((r: any, i: number) => {
              const t = test(r.dataset)
              if (r.missing_column) return <tr key={r.dataset}><td>{r.dataset}</td><td colSpan={s.categories.length + 4} className="muted">column missing</td></tr>
              return <tr key={r.dataset}><td><span className="swatch" style={{ background: `var(--s${(i % 8) + 1})` }} />{r.dataset}{i === 0 && <span className="tag" style={{ marginLeft: 6 }}>reference</span>}</td>
                <td className="num">{fmtInt(r.n)}</td>{s.categories.map((c: string) => <td key={c} className="num">{pct(r.shares[c])}</td>)}<td className="num">{pct(r.other)}</td>
                <td className="num">{t ? pFmt(t.chi2_p) : ''}</td><td className="num">{t ? <>{short(t.cramers_v)} <span className="muted small">{cramerLabel(t.cramers_v)}</span></> : ''}</td></tr>
            })}</tbody></table>
        )}
      </div>
      <div className="card-pad small text-2" style={{ borderTop: '1px solid var(--border)' }}>
        With large samples almost any difference is “significant” — read the effect size (Cohen's d / Cramér's V) for whether it matters. {s.note}
      </div>
    </div>
  )
}
