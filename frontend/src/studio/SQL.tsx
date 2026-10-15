import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { fmtTime } from '../api'
import { ErrorBox, Modal } from '../components/ui'
import { shortType, ResultTable } from './DataGrid'
import { SqlEditor } from './Recipe'
import { fmtInt, isNumericType, studio, type Col, type Dataset } from './types'
import { ChartView } from './viz'

type Result = Awaited<ReturnType<typeof studio.sql>>
type Table = Awaited<ReturnType<typeof studio.schema>>[number]
const HISTORY_KEY = 'studio.sql.history'
const loadHistory = (): { sql: string; t: number }[] => { try { return JSON.parse(localStorage.getItem(HISTORY_KEY) ?? '[]') } catch { return [] } }

/** SQL workspace: every built Studio dataset is a table (by slug), plus read-only dashboard tables. One SELECT at a time. */
export function SQLWorkspace({ list, reload }: { list: Dataset[]; reload: () => void }) {
  const first = list.find((d) => d.version > 0)
  const [sql, setSql] = useState(() => sessionStorage.getItem('studio.sql.draft') ?? (first ? `SELECT *\nFROM ${first.slug}\nLIMIT 100` : 'SELECT 42 AS answer'))
  const [res, setRes] = useState<Result>()
  const [err, setErr] = useState<unknown>()
  const [busy, setBusy] = useState(false)
  const [tables, setTables] = useState<Table[]>([])
  const [saved, setSaved] = useState<Awaited<ReturnType<typeof studio.queries>>>([])
  const [history, setHistory] = useState(loadHistory)
  const [side, setSide] = useState<'schema' | 'saved' | 'history'>('schema')
  const [open, setOpen] = useState<string | null>(null)
  const [dialog, setDialog] = useState<'query' | 'dataset' | null>(null)
  const [view, setView] = useState<'table' | 'chart'>('table')
  const nav = useNavigate()
  useEffect(() => { studio.schema().then(setTables).catch(() => {}); studio.queries().then(setSaved).catch(() => {}) }, [list.length])
  useEffect(() => { sessionStorage.setItem('studio.sql.draft', sql) }, [sql])
  useEffect(() => { if (first && sql === 'SELECT 42 AS answer') setSql(`SELECT *\nFROM ${first.slug}\nLIMIT 100`) }, [first?.slug]) // eslint-disable-line react-hooks/exhaustive-deps

  const run = () => {
    setBusy(true); setErr(undefined)
    studio.sql(sql, 5000).then((r) => {
      setRes(r)
      const h = [{ sql, t: Date.now() / 1000 }, ...history.filter((x) => x.sql !== sql)].slice(0, 30)
      setHistory(h); try { localStorage.setItem(HISTORY_KEY, JSON.stringify(h)) } catch { /* storage unavailable */ }
    }).catch(setErr).finally(() => setBusy(false))
  }
  const insert = (text: string) => setSql((s) => (s.endsWith('\n') || !s ? s : s + ' ') + text)

  return (
    <div className="sql-ws">
      <aside className="card sql-side">
        <div className="segmented sm" style={{ margin: 10 }}>{(['schema', 'saved', 'history'] as const).map((k) => <button key={k} className={side === k ? 'on' : ''} onClick={() => setSide(k)}>{k}</button>)}</div>
        <div className="sql-side-body">
          {side === 'schema' && tables.map((t) => (
            <div key={t.name}>
              <button className="schema-table" onClick={() => setOpen(open === t.name ? null : t.name)} title={t.label}>
                <span className="caret">{open === t.name ? '▾' : '▸'}</span><span className="mono">{t.name}</span>
                <span className="small muted" style={{ marginLeft: 'auto' }}>{t.kind === 'dashboard' ? 'dashboard' : t.rows != null ? fmtInt(t.rows) : ''}</span></button>
              {open === t.name && <div className="schema-cols">
                <button className="sm ghost" onClick={() => insert(`SELECT *\nFROM ${t.name}\nLIMIT 100`)}>insert SELECT</button>
                {t.kind === 'dashboard' && <div className="small muted" style={{ padding: '2px 6px' }}>{t.label}</div>}
                {t.columns.map((c: Col) => <button key={c.name} className="schema-col" onClick={() => insert(c.name)}><span className="mono">{c.name}</span><span className="muted small">{shortType(c.type)}</span></button>)}
              </div>}
            </div>
          ))}
          {side === 'saved' && (saved.length === 0 ? <div className="small muted" style={{ padding: 10 }}>No saved queries.</div> : saved.map((q) => (
            <div key={q.id} className="hist-item" onClick={() => setSql(q.sql)}>
              <div className="row" style={{ justifyContent: 'space-between' }}><b className="small">{q.name}</b>
                <button className="sm ghost danger" onClick={(e) => { e.stopPropagation(); studio.deleteQuery(q.id).then(() => studio.queries().then(setSaved)) }} aria-label="delete query">✕</button></div>
              <div className="mono small muted clamp">{q.sql}</div></div>)))}
          {side === 'history' && (history.length === 0 ? <div className="small muted" style={{ padding: 10 }}>Queries you run appear here (this browser only).</div> : history.map((h) => (
            <div key={h.t} className="hist-item" onClick={() => setSql(h.sql)}><div className="small muted">{fmtTime(h.t)}</div><div className="mono small clamp">{h.sql}</div></div>)))}
        </div>
      </aside>

      <div className="col" style={{ gap: 12, minWidth: 0 }}>
        <div className="card">
          <div className="card-head"><h3>SQL</h3><span className="small muted">DuckDB dialect · read-only sandbox · tables are dataset slugs · one statement per run</span></div>
          <div className="card-pad col" style={{ gap: 10 }}>
            <SqlEditor value={sql} onChange={setSql} onRun={run} rows={12} />
            <div className="row">
              <button className="primary" onClick={run} disabled={busy}>{busy ? 'Running…' : 'Run'} <span className="kbd">Ctrl ↵</span></button>
              <button onClick={() => setDialog('query')}>Save query</button>
              <button onClick={() => setDialog('dataset')} title="A query dataset: reproducible, versioned, rebuilt when its inputs change; can be profiled, charted and used to train models">
                Save result as dataset</button>
            </div>
            <ErrorBox error={err} />
          </div>
        </div>
        {res && (
          <div className="card">
            <div className="card-head">
              <div className="row small"><b>{fmtInt(res.row_count)}{res.truncated ? '+' : ''} rows</b><span className="muted">{res.seconds}s</span>
                {res.truncated && <span className="muted">— showing the first {fmtInt(res.row_count)}; save as a dataset to keep all rows</span>}</div>
              <div className="segmented sm">{(['table', 'chart'] as const).map((k) => <button key={k} className={view === k ? 'on' : ''} onClick={() => setView(k)}>{k}</button>)}</div>
            </div>
            {view === 'table' ? <ResultTable columns={res.columns} rows={res.rows} types={res.types} maxHeight={520} /> : <QuickChart res={res} />}
          </div>
        )}
      </div>
      {dialog && <SaveDialog kind={dialog} sql={sql} onClose={() => setDialog(null)} onSaved={(id) => {
        setDialog(null)
        if (dialog === 'query') studio.queries().then(setSaved)
        else if (id) { reload(); nav(`/studio/d/${id}`) }
      }} />}
    </div>
  )
}

function SaveDialog({ kind, sql, onClose, onSaved }: { kind: 'query' | 'dataset'; sql: string; onClose: () => void; onSaved: (id?: string) => void }) {
  const [name, setName] = useState('')
  const [err, setErr] = useState<unknown>()
  const go = async () => {
    try {
      if (kind === 'query') { await studio.saveQuery(name, sql); onSaved() }
      else { const d = await studio.create({ name, kind: 'query', query: sql }); onSaved(d.id) }
    } catch (e) { setErr(e) }
  }
  return (
    <Modal title={kind === 'query' ? 'Save query' : 'Save result as dataset'} onClose={onClose}>
      <div className="col" style={{ gap: 12 }}>
        {kind === 'dataset' && <div className="small text-2">Creates a <b>query dataset</b>: the full result (no row limit) is materialized as a versioned table, and it
          shows as stale when an input dataset changes. You can then profile it, chart it, compare it with other subsets and train models on it.</div>}
        <label className="field"><span className="lbl">Name</span><input autoFocus value={name} onChange={(e) => setName(e.target.value)} /></label>
        <ErrorBox error={err} />
        <button className="primary" style={{ alignSelf: 'flex-start' }} disabled={!name.trim()} onClick={go}>Save</button>
      </div>
    </Modal>
  )
}

/** Quick chart of a SQL result, computed client-side from the returned rows. */
function QuickChart({ res }: { res: Result }) {
  const num = res.columns.filter((_, i) => isNumericType(res.types?.[i]))
  const [x, setX] = useState(res.columns[0])
  const [y, setY] = useState(num.find((c) => c !== res.columns[0]) ?? num[0] ?? '')
  const [type, setType] = useState<'bar' | 'line' | 'scatter'>('bar')
  const result = useMemo(() => {
    const xi = res.columns.indexOf(x), yi = res.columns.indexOf(y)
    if (xi < 0 || yi < 0) return null
    const xs = res.rows.map((r) => r[xi]), ys = res.rows.map((r) => r[yi])
    if (type === 'bar') return { type: 'bar', x, y, yTitle: y, series: [{ name: y, categories: xs.map(String), values: ys }] }
    return { type, x, y, yTitle: y, series: [{ name: y, x: xs, y: ys }] }
  }, [res, x, y, type])
  return (
    <div className="card-pad col" style={{ gap: 8 }}>
      <div className="row small">
        <div className="segmented sm">{(['bar', 'line', 'scatter'] as const).map((k) => <button key={k} className={type === k ? 'on' : ''} onClick={() => setType(k)}>{k}</button>)}</div>
        <label className="row" style={{ gap: 4 }}>x <select style={{ width: 'auto' }} value={x} onChange={(e) => setX(e.target.value)}>{res.columns.map((c) => <option key={c}>{c}</option>)}</select></label>
        <label className="row" style={{ gap: 4 }}>y <select style={{ width: 'auto' }} value={y} onChange={(e) => setY(e.target.value)}>{num.map((c) => <option key={c}>{c}</option>)}</select></label>
        <span className="muted">Plots the {fmtInt(res.row_count)} returned rows as-is — aggregate in SQL (GROUP BY) for summaries.</span>
      </div>
      {result ? <ChartView result={result as any} height={380} /> : <div className="empty">The result needs a numeric column to plot.</div>}
    </div>
  )
}
