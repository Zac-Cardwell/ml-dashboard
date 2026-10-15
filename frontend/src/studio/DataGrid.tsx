import { useEffect, useState } from 'react'
import { ErrorBox } from '../components/ui'
import { fmtInt, isNumericType, studio, type Col, type Dataset } from './types'

const PAGE = 100

/** Server-paged grid over a whole dataset: sort by any column, filter with a SQL WHERE expression. */
export function DataGrid({ ds }: { ds: Dataset }) {
  const [offset, setOffset] = useState(0)
  const [sort, setSort] = useState<{ col: string; desc: boolean }>()
  const [where, setWhere] = useState('')
  const [applied, setApplied] = useState('')
  const [data, setData] = useState<{ total: number; columns: string[]; data: any[][] }>()
  const [err, setErr] = useState<unknown>()
  const [busy, setBusy] = useState(false)
  useEffect(() => {
    const q = new URLSearchParams({ offset: String(offset), limit: String(PAGE) })
    if (sort) { q.set('sort', sort.col); q.set('desc', String(sort.desc)) }
    if (applied) q.set('where', applied)
    setBusy(true); setErr(undefined)
    studio.rows(ds.id, q.toString()).then(setData).catch(setErr).finally(() => setBusy(false))
  }, [ds.id, ds.version, offset, sort, applied])
  const types = Object.fromEntries(ds.schema.map((c) => [c.name, c.type]))
  const total = data?.total ?? 0
  const clickSort = (col: string) => { setOffset(0); setSort(sort?.col === col ? (sort.desc ? undefined : { col, desc: true }) : { col, desc: false }) }
  return (
    <div className="col" style={{ gap: 10 }}>
      <form className="row" onSubmit={(e) => { e.preventDefault(); setOffset(0); setApplied(where.trim()) }}>
        <span className="small muted mono">WHERE</span>
        <input className="mono" style={{ flex: 1, minWidth: 240 }} placeholder="e.g. income > 50000 AND region = 'west'" value={where} onChange={(e) => setWhere(e.target.value)} />
        <button type="submit">Apply</button>
        {applied && <button type="button" className="ghost" onClick={() => { setWhere(''); setApplied(''); setOffset(0) }}>Clear</button>}
      </form>
      <ErrorBox error={err} />
      <div className="card">
        <div className="table-wrap grid-scroll">
          <table className="dense data-grid">
            <thead><tr><th className="rownum">#</th>{(data?.columns ?? ds.schema.map((c) => c.name)).map((c) => (
              <th key={c} className={isNumericType(types[c]) ? 'num sortable' : 'sortable'} onClick={() => clickSort(c)} title={`${types[c] ?? ''} — click to sort`}>
                {c}{sort?.col === c ? (sort.desc ? ' ↓' : ' ↑') : ''}<div className="coltype">{shortType(types[c])}</div></th>))}</tr></thead>
            <tbody>{data?.data.map((r, i) => (
              <tr key={i}><td className="rownum">{fmtInt(offset + i + 1)}</td>{r.map((v, j) => <Cell key={j} v={v} numeric={isNumericType(types[data.columns[j]])} />)}</tr>))}</tbody>
          </table>
        </div>
        <div className="card-foot row small">
          <span className="muted">{busy ? 'Loading…' : `${fmtInt(total)} rows${applied ? ' match' : ''}`}</span>
          <span style={{ marginLeft: 'auto' }} className="muted">{total ? `${fmtInt(offset + 1)}–${fmtInt(Math.min(offset + PAGE, total))}` : ''}</span>
          <button className="sm" disabled={offset === 0} onClick={() => setOffset(0)}>«</button>
          <button className="sm" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>Prev</button>
          <button className="sm" disabled={offset + PAGE >= total} onClick={() => setOffset(offset + PAGE)}>Next</button>
        </div>
      </div>
    </div>
  )
}

export function shortType(t?: string) {
  if (!t) return ''
  return t.replace('VARCHAR', 'text').replace('DOUBLE', 'float').replace('BIGINT', 'int').replace('INTEGER', 'int').replace('TIMESTAMP', 'time').replace('BOOLEAN', 'bool').toLowerCase()
}

export function Cell({ v, numeric }: { v: any; numeric?: boolean }) {
  if (v == null) return <td className="null">null</td>
  if (typeof v === 'number') return <td className="num">{Number.isInteger(v) ? v.toLocaleString() : Math.abs(v) >= 1e6 || Math.abs(v) < 1e-4 ? v.toExponential(3) : v.toFixed(4).replace(/\.?0+$/, '')}</td>
  if (typeof v === 'boolean') return <td className="mono">{String(v)}</td>
  const s = String(v)
  return <td className={numeric ? 'num' : ''} title={s.length > 60 ? s : undefined}>{s.length > 60 ? s.slice(0, 60) + '…' : s}</td>
}

/** Plain result table (SQL results, previews). */
export function ResultTable({ columns, rows, types, maxHeight = 460, index = true }: { columns: string[]; rows: any[][]; types?: (string | undefined)[]; maxHeight?: number; index?: boolean }) {
  return (
    <div className="table-wrap grid-scroll" style={{ maxHeight }}>
      <table className="dense data-grid">
        <thead><tr>{index && <th className="rownum">#</th>}{columns.map((c, i) => <th key={c + i} className={isNumericType(types?.[i]) ? 'num' : ''}>{c}
          {types?.[i] && <div className="coltype">{shortType(types[i])}</div>}</th>)}</tr></thead>
        <tbody>{rows.map((r, i) => <tr key={i}>{index && <td className="rownum">{i + 1}</td>}{r.map((v, j) => <Cell key={j} v={v} numeric={isNumericType(types?.[j])} />)}</tr>)}</tbody>
      </table>
    </div>
  )
}

export const colsOf = (schema: Col[], numericOnly = false) => schema.filter((c) => c.name !== '__split' && (!numericOnly || isNumericType(c.type))).map((c) => c.name)
