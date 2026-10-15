import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react'
import { Link } from 'react-router-dom'
import { ErrorBox } from '../components/ui'
import { colsOf, ResultTable } from './DataGrid'
import { fmtInt, studio, type Col, type Dataset, type Step, type StepSpec } from './types'

const OPS = ['=', '!=', '>', '>=', '<', '<=', 'between', 'in', 'not in', 'contains', 'starts with', 'is null', 'is not null']
const AGG_FNS = ['count', 'sum', 'avg', 'min', 'max', 'median', 'stddev', 'count_distinct', 'first', 'last']
const CATEGORY_ORDER = ['Rows', 'Columns', 'Cleaning', 'Features', 'Combine', 'Modeling', 'Advanced']

let specCache: Promise<StepSpec[]> | null = null
export const useSpecs = () => {
  const [specs, setSpecs] = useState<StepSpec[]>([])
  useEffect(() => { (specCache ??= studio.steps()).then(setSpecs).catch(() => { specCache = null }) }, [])
  return specs
}
const newId = () => Math.random().toString(36).slice(2, 12)

type Preview = { sampled: boolean; sample_rows: number; rows: number; schema: Col[]; warnings: { index?: number; kind: string; message: string }[]
  trace: { index: number; rows: number; columns: number }[]; fitted: Record<string, any>; head: { columns: string[]; data: any[][] } }

/** Visual recipe editor: ordered steps, live preview on a sample of the parent, save → versioned rebuild. */
export function RecipeTab({ ds, list, onSaved }: { ds: Dataset; list: Dataset[]; onSaved: () => void }) {
  const specs = useSpecs()
  const spec = useMemo(() => Object.fromEntries(specs.map((s) => [s.type, s])), [specs])
  const [recipe, setRecipe] = useState<Step[]>(ds.recipe)
  const [sel, setSel] = useState<number | null>(ds.recipe.length ? 0 : null)
  const [preview, setPreview] = useState<Preview>()
  const [pErr, setPErr] = useState<unknown>()
  const [busy, setBusy] = useState(false)
  const [saveErr, setSaveErr] = useState<unknown>()
  const [adding, setAdding] = useState(false)
  const dirty = JSON.stringify(recipe) !== JSON.stringify(ds.recipe)
  const parent = list.find((d) => d.id === ds.parent_id)
  useEffect(() => { if (!dirty) setRecipe(ds.recipe) }, [ds.version]) // eslint-disable-line react-hooks/exhaustive-deps

  // debounced preview of the whole recipe on a sample of the parent
  const seq = useRef(0)
  useEffect(() => {
    if (!ds.parent_id) return
    const n = ++seq.current
    const t = setTimeout(() => {
      setBusy(true)
      studio.preview({ parent_id: ds.parent_id, recipe, target: ds.target })
        .then((p) => { if (n === seq.current) { setPreview(p); setPErr(undefined) } })
        .catch((e) => { if (n === seq.current) setPErr(e) })
        .finally(() => { if (n === seq.current) setBusy(false) })
    }, 600)
    return () => clearTimeout(t)
  }, [JSON.stringify(recipe), ds.parent_id, ds.target]) // eslint-disable-line react-hooks/exhaustive-deps

  const columns = useMemo(() => {
    const s = new Set<string>(colsOf(parent?.schema ?? []))
    preview?.schema.forEach((c) => c.name !== '__split' && s.add(c.name))
    return [...s]
  }, [parent, preview])
  const numeric = useMemo(() => {
    const all = [...(parent?.schema ?? []), ...(preview?.schema ?? [])]
    return [...new Set(colsOf(all, true))]
  }, [parent, preview])

  const update = (i: number, s: Step) => setRecipe(recipe.map((x, j) => (j === i ? s : x)))
  const move = (i: number, d: number) => {
    const j = i + d
    if (j < 0 || j >= recipe.length) return
    const r = [...recipe]; [r[i], r[j]] = [r[j], r[i]]; setRecipe(r); setSel(j)
  }
  const add = (type: string) => {
    const s = spec[type]
    const step: Step = { id: newId(), type, enabled: true, params: Object.fromEntries(s.params.map((p) => [p.name, structuredClone(p.default)])) }
    const at = sel == null ? recipe.length : sel + 1
    setRecipe([...recipe.slice(0, at), step, ...recipe.slice(at)]); setSel(at); setAdding(false)
  }
  const save = async () => {
    setSaveErr(undefined)
    try { await studio.patch(ds.id, { recipe }); onSaved() } catch (e) { setSaveErr(e) }
  }
  const splitAt = recipe.findIndex((s) => s.type === 'split' && s.enabled !== false)
  const traceOf = (i: number) => preview?.trace.find((t) => t.index === i)
  const warnOf = (i: number) => preview?.warnings.filter((w) => w.index === i) ?? []
  const current = sel != null ? recipe[sel] : undefined

  return (
    <div className="recipe">
      <div className="card recipe-steps">
        <div className="card-head"><h3>Recipe</h3><span className="small muted">{recipe.length} step{recipe.length === 1 ? '' : 's'}</span></div>
        <div className="step source-step">
          <span className="step-n">⬚</span>
          <div style={{ minWidth: 0 }}><div className="step-title">{parent ? <Link to={`/studio/d/${parent.id}`}>{parent.name}</Link> : 'parent'}</div>
            <div className="small muted">{fmtInt(parent?.rows)} rows · {parent?.n_columns} cols{preview?.sampled ? ` · preview on ${fmtInt(preview.sample_rows)}-row sample` : ''}</div></div>
        </div>
        {recipe.map((s, i) => {
          const sp = spec[s.type]
          const tr = traceOf(i)
          const w = warnOf(i)
          const fitted = sp?.fitted && !(s.type === 'impute' && s.params.strategy === 'constant') && !(s.type === 'clip' && s.params.method === 'fixed')
          return (
            <div key={s.id} className={`step${sel === i ? ' on' : ''}${s.enabled === false ? ' off' : ''}`} onClick={() => setSel(i)}>
              <span className="step-n">{i + 1}</span>
              <div style={{ minWidth: 0, flex: 1 }}>
                <div className="step-title">{sp?.label ?? s.type}
                  {fitted && (splitAt >= 0 && splitAt < i
                    ? <span className="pill good" title="Learns statistics from the train split only">fit on train</span>
                    : <span className="pill warn" title="Learns statistics from all rows — add a split step before it for modeling">fit on all rows</span>)}
                  {sp?.python && <span className="pill">python</span>}
                </div>
                <div className="small muted step-sum">{summarize(s, sp)}</div>
                {tr && <div className="small text-2 num">→ {fmtInt(tr.rows)} rows · {tr.columns} cols</div>}
                {w.filter((x) => x.kind !== 'leakage').map((x, k) => <div key={k} className="small" style={{ color: 'var(--critical-text)' }}>{x.message}</div>)}
              </div>
              <div className="step-actions" onClick={(e) => e.stopPropagation()}>
                <button className="sm ghost" title="Move up" disabled={i === 0} onClick={() => move(i, -1)}>↑</button>
                <button className="sm ghost" title="Move down" disabled={i === recipe.length - 1} onClick={() => move(i, 1)}>↓</button>
                <button className="sm ghost" title={s.enabled === false ? 'Enable' : 'Disable (keep, but skip)'} onClick={() => update(i, { ...s, enabled: s.enabled === false })}>{s.enabled === false ? '○' : '●'}</button>
                <button className="sm ghost danger" title="Delete step" onClick={() => { setRecipe(recipe.filter((_, j) => j !== i)); setSel(recipe.length > 1 ? Math.max(0, i - 1) : null) }}>✕</button>
              </div>
            </div>
          )
        })}
        <div className="card-pad" style={{ paddingTop: 10 }}>
          {adding ? <StepPicker specs={specs} onPick={add} onCancel={() => setAdding(false)} />
            : <button style={{ width: '100%' }} onClick={() => setAdding(true)}>+ Add step{sel != null && sel < recipe.length - 1 ? ` after ${sel + 1}` : ''}</button>}
        </div>
        <div className="card-foot col" style={{ gap: 8 }}>
          <ErrorBox error={saveErr} />
          <div className="row">
            <button className="primary" disabled={!dirty || !!pErr} onClick={save} title={pErr ? 'Fix the preview error first' : ''}>Save & build v{ds.version + 1}</button>
            <button className="ghost" disabled={!dirty} onClick={() => setRecipe(ds.recipe)}>Discard changes</button>
          </div>
          <div className="small muted">Building runs the recipe over every row of the parent. Fitted steps then fit on the full train split.</div>
        </div>
      </div>

      <div className="col" style={{ gap: 14, minWidth: 0 }}>
        {current && spec[current.type] && (
          <div className="card">
            <div className="card-head"><h3>{sel! + 1}. {spec[current.type].label}</h3><span className="small muted">{spec[current.type].category}</span></div>
            <div className="card-pad col" style={{ gap: 12 }}>
              <div className="small text-2">{spec[current.type].help}</div>
              <StepForm step={current} spec={spec[current.type]} columns={columns} numeric={numeric} list={list.filter((d) => d.id !== ds.id && d.version > 0)}
                onChange={(s) => update(sel!, s)} />
              {preview?.fitted[current.id] && <details className="small"><summary>Fitted statistics (preview sample)</summary>
                <pre className="mono small fitted-json">{JSON.stringify(preview.fitted[current.id], null, 1).slice(0, 4000)}</pre></details>}
            </div>
          </div>
        )}
        {!current && <div className="card card-pad small text-2">
          Add steps to clean and engineer features. A typical modeling recipe: <b>Filter</b> → <b>Remove duplicates</b> → <b>Train / val / test split</b> →
          <b> Fill missing</b> → <b>Clip outliers</b> → <b>One-hot encode</b> → <b>Scale</b>. Putting the split first means every fitted step learns only from
          training rows — the same discipline a scikit-learn pipeline enforces.</div>}

        <div className="card">
          <div className="card-head"><h3>Preview</h3>
            <span className="small muted">{busy ? 'Running…' : preview ? `${fmtInt(preview.rows)} rows × ${preview.schema.length} columns${preview.sampled ? ' (on sample)' : ''}` : ''}</span></div>
          {pErr ? <div className="card-pad"><ErrorBox error={pErr} /></div> : preview && (
            <>
              {preview.warnings.some((w) => w.kind === 'leakage') && <div className="callout warning small" style={{ margin: 12 }}>
                <b>Leakage risk:</b> {(() => { const l = preview.warnings.filter((w) => w.kind === 'leakage'); return `${l.length > 1 ? 'Steps' : 'Step'} ${l.map((w) => (w.index ?? 0) + 1).join(', ')} ${l.length > 1 ? 'learn' : 'learns'}` })()} statistics from all rows.
                Fine for exploration; for a modeling dataset add a <b>Train / val / test split</b> step before them.</div>}
              <SchemaDiff before={parent?.schema ?? []} after={preview.schema} />
              <ResultTable columns={preview.head.columns} rows={preview.head.data} types={preview.head.columns.map((c) => preview.schema.find((s) => s.name === c)?.type)} maxHeight={420} />
            </>
          )}
        </div>
      </div>
    </div>
  )
}

function SchemaDiff({ before, after }: { before: Col[]; after: Col[] }) {
  const b = new Map(before.map((c) => [c.name, c.type]))
  const a = new Map(after.map((c) => [c.name, c.type]))
  const added = after.filter((c) => !b.has(c.name))
  const removed = before.filter((c) => !a.has(c.name))
  const changed = after.filter((c) => b.has(c.name) && b.get(c.name) !== c.type)
  if (!added.length && !removed.length && !changed.length) return null
  return (
    <div className="schema-diff small">
      {added.length > 0 && <div><span className="pill good">+{added.length}</span> {added.map((c) => c.name).join(', ')}</div>}
      {removed.length > 0 && <div><span className="pill bad">−{removed.length}</span> {removed.map((c) => c.name).join(', ')}</div>}
      {changed.length > 0 && <div><span className="pill">type</span> {changed.map((c) => `${c.name}: ${b.get(c.name)} → ${c.type}`).join(', ')}</div>}
    </div>
  )
}

function StepPicker({ specs, onPick, onCancel }: { specs: StepSpec[]; onPick: (t: string) => void; onCancel: () => void }) {
  const [q, setQ] = useState('')
  const ql = q.toLowerCase()
  const byName = specs.filter((s) => (s.label + ' ' + s.type).toLowerCase().includes(ql))
  // while searching: name matches first, then description matches, as one flat list
  const shown = !q ? specs : [...byName, ...specs.filter((s) => !byName.includes(s) && s.help.toLowerCase().includes(ql))]
  const cats = q ? ['Matches'] : CATEGORY_ORDER.filter((c) => shown.some((s) => s.category === c))
  return (
    <div className="step-picker">
      <div className="row" style={{ flexWrap: 'nowrap' }}><input autoFocus placeholder="Search steps…" value={q} onChange={(e) => setQ(e.target.value)}
        onKeyDown={(e) => { if (e.key === 'Enter' && shown[0]) onPick(shown[0].type); if (e.key === 'Escape') onCancel() }} />
        <button className="sm ghost" onClick={onCancel}>Cancel</button></div>
      <div className="step-picker-list">{shown.length === 0 && <div className="small muted" style={{ padding: 10 }}>No step matches.</div>}{cats.map((c) => (
        <div key={c}>{!q && <div className="picker-cat">{c}</div>}
          {shown.filter((s) => q || s.category === c).map((s) => (
            <button key={s.type} className="picker-item" onClick={() => onPick(s.type)}>
              <span>{s.label}{s.fitted && <span className="pill">fitted</span>}</span><span className="small muted">{s.help}</span></button>))}
        </div>))}</div>
    </div>
  )
}

function summarize(s: Step, sp?: StepSpec): string {
  const p: Record<string, any> = { ...Object.fromEntries((sp?.params ?? []).map((x) => [x.name, x.default])), ...s.params }
  const cols = (c?: string[]) => (c && c.length ? (c.length > 3 ? `${c.slice(0, 3).join(', ')} +${c.length - 3}` : c.join(', ')) : 'all')
  switch (s.type) {
    case 'filter': return [...(p.conditions ?? []).filter((c: any) => c.column).map((c: any) => `${c.column} ${c.op} ${c.op?.includes('null') ? '' : c.value ?? ''}`), p.sql].filter(Boolean).join(` ${p.combine} `) || 'no conditions'
    case 'split': return `${p.method}${p.column ? ` by ${p.column}` : ''} · val ${p.val_fraction} · test ${p.test_fraction}`
    case 'derive': return `${p.name} = ${p.expression || '…'}`
    case 'rename': return `${p.column || '…'} → ${p.to || '…'}`
    case 'join': return `${p.how} join on ${cols(p.left_on)}`
    case 'aggregate': return `by ${cols(p.group_by)} · ${(p.aggs ?? []).map((a: any) => `${a.fn}(${a.column})`).join(', ')}`
    case 'sql': return (p.sql ?? '').replace(/\s+/g, ' ').slice(0, 70)
    case 'impute': return `${p.strategy} · ${cols(p.columns)}`
    case 'scale': return `${p.method} · ${cols(p.columns)}`
    case 'clip': return `${p.method} · ${cols(p.columns)}`
    case 'cast': return `${cols(p.columns)} → ${p.to}`
    case 'pca': return p.n_components ? `${p.n_components} components` : `${(p.variance * 100).toFixed(0)}% variance`
    case 'cluster': return `${p.algorithm}${['kmeans', 'gmm'].includes(p.algorithm) ? ` k=${p.k}` : ''} → ${p.name}`
    case 'anomaly': return `${p.algorithm} · ${p.contamination} → ${p.name}`
    case 'sample': return p.fraction ? `${p.fraction * 100}%` : `${fmtInt(p.n)} rows`
    default: {
      if (Array.isArray(p.columns)) return cols(p.columns)
      if (p.column) return p.column
      return sp?.help ?? ''
    }
  }
}

// ------------------------------------------------------------------------- forms
function StepForm({ step, spec, columns, numeric, list, onChange }: {
  step: Step; spec: StepSpec; columns: string[]; numeric: string[]; list: Dataset[]; onChange: (s: Step) => void
}) {
  const set = (k: string, v: any) => onChange({ ...step, params: { ...step.params, [k]: v } })
  const other = list.find((d) => d.id === step.params.other)
  return (
    <div className="step-form">
      {spec.params.map((p) => {
        const v = step.params[p.name] ?? p.default
        let input
        switch (p.type) {
          case 'conditions': input = <Conditions value={v ?? []} columns={columns} onChange={(x) => set(p.name, x)} />; break
          case 'aggs': input = <Aggs value={v ?? []} columns={columns} onChange={(x) => set(p.name, x)} />; break
          case 'columns': input = <ColumnsPicker value={v ?? []} options={columns} numeric={numeric} onChange={(x) => set(p.name, x)} />; break
          case 'columns_other': input = <ColumnsPicker value={v ?? []} options={colsOf(other?.schema ?? [])} onChange={(x) => set(p.name, x)} />; break
          case 'column': input = <select value={v ?? ''} onChange={(e) => set(p.name, e.target.value)}><option value="">—</option>{columns.map((c) => <option key={c}>{c}</option>)}</select>; break
          case 'choice': input = <select value={v} onChange={(e) => set(p.name, e.target.value)}>{p.choices!.map((c) => <option key={c}>{c}</option>)}</select>; break
          case 'int': input = <input type="number" step={1} value={v ?? ''} onChange={(e) => set(p.name, e.target.value === '' ? 0 : parseInt(e.target.value))} />; break
          case 'float': input = <input type="number" step="any" value={v ?? ''} onChange={(e) => set(p.name, e.target.value === '' ? 0 : parseFloat(e.target.value))} />; break
          case 'bool': input = <label className="check"><input type="checkbox" checked={!!v} onChange={(e) => set(p.name, e.target.checked)} /> {p.label}</label>; break
          case 'sql_expr': input = <input className="mono" value={v ?? ''} placeholder={p.help} onChange={(e) => set(p.name, e.target.value)} />; break
          case 'sql': input = <SqlEditor value={v ?? ''} onChange={(x) => set(p.name, x)} rows={8} />; break
          case 'dataset': input = <select value={v ?? ''} onChange={(e) => set(p.name, e.target.value)}><option value="">—</option>{list.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}</select>; break
          case 'datasets': input = <Chips value={v ?? []} options={list.map((d) => ({ value: d.id, label: d.name }))} onChange={(x) => set(p.name, x)} />; break
          case 'multi': input = <Chips value={v ?? []} options={p.choices!.map((c) => ({ value: c, label: c }))} onChange={(x) => set(p.name, x)} all />; break
          default: input = <input value={v ?? ''} onChange={(e) => set(p.name, e.target.value)} />
        }
        const wide = ['conditions', 'aggs', 'columns', 'columns_other', 'sql', 'sql_expr', 'datasets', 'multi'].includes(p.type)
        const Wrap = ['conditions', 'aggs', 'columns', 'columns_other', 'datasets', 'multi', 'bool'].includes(p.type) ? 'div' : 'label'
        return (
          <Wrap key={p.name} className="field" style={wide ? { gridColumn: '1 / -1' } : undefined}>
            {p.type !== 'bool' && <span className="lbl">{p.label}</span>}
            {input}
            {p.help && p.type !== 'sql_expr' && <span className="help">{p.help}</span>}
          </Wrap>
        )
      })}
    </div>
  )
}

function ColumnsPicker({ value, options, numeric, onChange }: { value: string[]; options: string[]; numeric?: string[]; onChange: (v: string[]) => void }) {
  const [q, setQ] = useState('')
  const rest = options.filter((o) => !value.includes(o) && (!q || o.toLowerCase().includes(q.toLowerCase())))
  return (
    <div className="colpicker">
      <div className="chips">
        {value.length === 0 && <span className="small muted">none selected</span>}
        {value.map((c) => <span key={c} className="chip">{c}<button type="button" onClick={() => onChange(value.filter((x) => x !== c))} aria-label={`remove ${c}`}>×</button></span>)}
      </div>
      <div className="row" style={{ gap: 6, flexWrap: 'nowrap' }}>
        <input placeholder="filter columns…" value={q} onChange={(e) => setQ(e.target.value)} style={{ flex: 1 }} />
        {numeric && <button type="button" className="sm" onClick={() => onChange([...new Set([...value, ...numeric.filter((n) => options.includes(n))])])}>+ numeric</button>}
        <button type="button" className="sm" onClick={() => onChange([...value, ...rest])} disabled={!rest.length}>+ all{q ? ' shown' : ''}</button>
        {value.length > 0 && <button type="button" className="sm ghost" onClick={() => onChange([])}>clear</button>}
      </div>
      {rest.length > 0 && <div className="colpicker-opts">{rest.slice(0, 200).map((c) => <button type="button" key={c} className="opt" onClick={() => onChange([...value, c])}>{c}</button>)}</div>}
    </div>
  )
}

function Chips({ value, options, onChange, all }: { value: string[]; options: { value: string; label: string }[]; onChange: (v: string[]) => void; all?: boolean }) {
  return (
    <div className="chips">{options.map((o) => {
      const on = value.includes(o.value)
      return <button type="button" key={o.value} className={`chip toggle${on ? ' on' : ''}`} onClick={() => onChange(on ? value.filter((x) => x !== o.value) : [...value, o.value])}>{o.label}</button>
    })}{!all && options.length === 0 && <span className="small muted">no other built datasets</span>}</div>
  )
}

function Conditions({ value, columns, onChange }: { value: any[]; columns: string[]; onChange: (v: any[]) => void }) {
  const set = (i: number, k: string, v: any) => onChange(value.map((c, j) => (j === i ? { ...c, [k]: v } : c)))
  return (
    <div className="col" style={{ gap: 6 }}>
      {value.map((c, i) => (
        <div key={i} className="cond-row">
          <select value={c.column ?? ''} onChange={(e) => set(i, 'column', e.target.value)}><option value="">column…</option>{columns.map((x) => <option key={x}>{x}</option>)}</select>
          <select value={c.op ?? '='} onChange={(e) => set(i, 'op', e.target.value)}>{OPS.map((o) => <option key={o}>{o}</option>)}</select>
          {c.op?.includes('null') ? <span /> : <input className="mono" value={c.value ?? ''} placeholder={c.op === 'between' ? 'low, high' : c.op?.includes('in') ? 'a, b, c' : 'value'}
            onChange={(e) => set(i, 'value', e.target.value)} />}
          <button type="button" className="sm ghost" onClick={() => onChange(value.filter((_, j) => j !== i))} aria-label="remove condition">✕</button>
        </div>
      ))}
      <button type="button" className="sm" style={{ alignSelf: 'flex-start' }} onClick={() => onChange([...value, { column: '', op: '=', value: '' }])}>+ condition</button>
    </div>
  )
}

function Aggs({ value, columns, onChange }: { value: any[]; columns: string[]; onChange: (v: any[]) => void }) {
  const set = (i: number, k: string, v: any) => onChange(value.map((c, j) => (j === i ? { ...c, [k]: v } : c)))
  return (
    <div className="col" style={{ gap: 6 }}>
      {value.map((a, i) => (
        <div key={i} className="cond-row">
          <select value={a.fn} onChange={(e) => set(i, 'fn', e.target.value)}>{AGG_FNS.map((f) => <option key={f}>{f}</option>)}</select>
          <select value={a.column ?? '*'} onChange={(e) => set(i, 'column', e.target.value)}><option value="*">* (rows)</option>{columns.map((x) => <option key={x}>{x}</option>)}</select>
          <input placeholder="as name" value={a.as ?? ''} onChange={(e) => set(i, 'as', e.target.value)} />
          <button type="button" className="sm ghost" onClick={() => onChange(value.filter((_, j) => j !== i))} aria-label="remove aggregate">✕</button>
        </div>
      ))}
      <button type="button" className="sm" style={{ alignSelf: 'flex-start' }} onClick={() => onChange([...value, { column: '*', fn: 'count', as: '' }])}>+ aggregate</button>
    </div>
  )
}

/** Monospace SQL textarea: Tab indents, Ctrl/⌘+Enter runs. */
export function SqlEditor({ value, onChange, onRun, rows = 10, placeholder }: { value: string; onChange: (v: string) => void; onRun?: () => void; rows?: number; placeholder?: string }) {
  const key = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); onRun?.(); return }
    if (e.key === 'Tab' && !e.shiftKey) {
      e.preventDefault()
      const el = e.currentTarget, a = el.selectionStart, b = el.selectionEnd
      const v = value.slice(0, a) + '  ' + value.slice(b)
      onChange(v)
      requestAnimationFrame(() => { el.selectionStart = el.selectionEnd = a + 2 })
    }
  }
  return <textarea className="sql-editor" spellCheck={false} rows={rows} value={value} placeholder={placeholder} onChange={(e) => onChange(e.target.value)} onKeyDown={key} />
}

/** A query dataset: its SQL over other Studio datasets (by slug), rebuilt as a new version on save. */
export function QueryEditorTab({ ds, onSaved }: { ds: Dataset; onSaved: () => void }) {
  const [sql, setSql] = useState(ds.query ?? '')
  const [res, setRes] = useState<Awaited<ReturnType<typeof studio.sql>>>()
  const [err, setErr] = useState<unknown>()
  const [busy, setBusy] = useState(false)
  const run = () => { setBusy(true); setErr(undefined); studio.sql(sql, 200).then(setRes).catch(setErr).finally(() => setBusy(false)) }
  const save = async () => { setErr(undefined); try { await studio.patch(ds.id, { query: sql }); onSaved() } catch (e) { setErr(e) } }
  return (
    <div className="col" style={{ gap: 12 }}>
      <div className="card">
        <div className="card-head"><h3>Query</h3><span className="small muted">reference datasets by slug · <Link to="/studio/sql">open the SQL workspace</Link> for the schema browser</span></div>
        <div className="card-pad col" style={{ gap: 10 }}>
          <SqlEditor value={sql} onChange={setSql} onRun={run} rows={12} />
          <div className="row">
            <button onClick={run} disabled={busy}>{busy ? 'Running…' : 'Run preview'} <span className="kbd">Ctrl ↵</span></button>
            <button className="primary" disabled={sql.trim() === (ds.query ?? '').trim()} onClick={save}>Save & build v{ds.version + 1}</button>
          </div>
          <ErrorBox error={err} />
        </div>
      </div>
      {res && <div className="card"><div className="card-head"><h3>Preview</h3><span className="small muted">{fmtInt(res.row_count)}{res.truncated ? '+' : ''} rows · {res.seconds}s</span></div>
        <ResultTable columns={res.columns} rows={res.rows} types={res.types} /></div>}
    </div>
  )
}
