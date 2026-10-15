import type { Param } from '../api'

export function normalize(v: unknown): string {
  if (Array.isArray(v)) return v.join(',')
  if (v !== null && typeof v === 'object') return JSON.stringify(v)
  if (v === null || v === undefined) return ''
  return String(v).replace(/\s+/g, '')
}

/** Form values start from the declared defaults, overridden by any provided values. */
export function initialValues(params: Param[], from?: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const p of params) out[p.name] = from && p.name in from ? from[p.name] : p.default
  return out
}

function toInput(p: Param, v: unknown): string {
  if (p.type === 'json') return typeof v === 'string' ? v : JSON.stringify(v ?? {}, null, 0)
  if (p.type === 'int_list') return Array.isArray(v) ? v.join(', ') : String(v ?? '')
  return v === null || v === undefined ? '' : String(v)
}

/**
 * Renders a form from a plugin's parameter schema, grouped by `group`.
 * `baseline` (e.g. the parent session's values) highlights changed fields for "train again".
 */
export function ParamForm({ params, values, onChange, baseline }: {
  params: Param[]
  values: Record<string, unknown>
  onChange: (v: Record<string, unknown>) => void
  baseline?: Record<string, unknown>
}) {
  const visible = (p: Param) => {
    const c = p.show_if
    if (!c) return true
    const cur = values[c.param] ?? params.find((q) => q.name === c.param)?.default
    if (c.in && !c.in.includes(cur)) return false
    if (c.not_in && c.not_in.includes(cur)) return false
    return true
  }
  const groups = new Map<string, Param[]>()
  for (const p of params.filter(visible)) {
    if (!groups.has(p.group)) groups.set(p.group, [])
    groups.get(p.group)!.push(p)
  }
  const set = (name: string, v: unknown) => onChange({ ...values, [name]: v })

  if (!params.length) return <div className="muted small">No parameters.</div>
  return (
    <div className="col" style={{ gap: 18 }}>
      {[...groups.entries()].map(([g, ps]) => (
        <div key={g}>
          <h3 style={{ marginBottom: 10 }} className="text-2">{g}</h3>
          <div className="grid-3">
            {ps.map((p) => {
              const v = values[p.name]
              const changed = baseline !== undefined && p.name in baseline && normalize(baseline[p.name]) !== normalize(v)
              const id = `p-${p.name}`
              if (p.type === 'bool') {
                return (
                  <label key={p.name} className={`field ${changed ? 'changed' : ''}`} title={p.help}>
                    <span className="lbl">{p.name}{changed && <ChangedMark from={baseline![p.name]} />}</span>
                    <span className="check"><input type="checkbox" checked={!!v} onChange={(e) => set(p.name, e.target.checked)} />
                      <span className="small text-2">{p.help || 'enabled'}</span></span>
                  </label>
                )
              }
              return (
                <label key={p.name} className={`field ${changed ? 'changed' : ''}`}>
                  <span className="lbl">
                    {p.name}
                    {p.type === 'int_list' && <span className="muted small">list</span>}
                    {changed && <ChangedMark from={baseline![p.name]} />}
                  </span>
                  {p.type === 'choice' ? (
                    <ChoiceSelect p={p} value={String(v ?? '')} onChange={(x) => set(p.name, x)} />
                  ) : p.type === 'json' ? (
                    <textarea className="mono" rows={2} value={toInput(p, v)} onChange={(e) => set(p.name, e.target.value)} />
                  ) : (
                    <>
                      <input id={id} value={toInput(p, v)} list={p.suggestions ? `${id}-list` : undefined}
                        inputMode={p.type === 'float' || p.type === 'int' ? 'decimal' : undefined}
                        placeholder={p.type === 'int_list' ? 'e.g. 256, 128' : undefined}
                        onChange={(e) => set(p.name, e.target.value)} />
                      {p.suggestions && (
                        <datalist id={`${id}-list`}>{p.suggestions.map((s) => <option key={s} value={s} />)}</datalist>
                      )}
                    </>
                  )}
                  {(p.help || p.min !== undefined || p.max !== undefined) && (
                    <span className="help">
                      {p.help}
                      {(p.min !== undefined || p.max !== undefined) && (
                        <span> {p.help ? '· ' : ''}{p.min !== undefined ? `≥ ${p.min}` : ''}{p.min !== undefined && p.max !== undefined ? ', ' : ''}{p.max !== undefined ? `≤ ${p.max}` : ''}</span>
                      )}
                    </span>
                  )}
                </label>
              )
            })}
          </div>
        </div>
      ))}
    </div>
  )
}

function ChoiceSelect({ p, value, onChange }: { p: Param; value: string; onChange: (v: string) => void }) {
  const label = (c: string) => p.labels?.[c] ?? c
  const missing = value && !p.choices!.includes(value)
  if (!p.option_groups) {
    return (
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {missing && <option value={value}>{value} (unavailable)</option>}
        {p.choices!.map((c) => <option key={c} value={c}>{label(c)}</option>)}
      </select>
    )
  }
  const order: string[] = []
  const by = new Map<string, string[]>()
  for (const c of p.choices!) {
    const g = p.option_groups[c] ?? ''
    if (!by.has(g)) { by.set(g, []); order.push(g) }
    by.get(g)!.push(c)
  }
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)}>
      {missing && <option value={value}>{value} (unavailable)</option>}
      {order.map((g) => g ? <optgroup key={g} label={g}>{by.get(g)!.map((c) => <option key={c} value={c}>{label(c)}</option>)}</optgroup>
        : by.get(g)!.map((c) => <option key={c} value={c}>{label(c)}</option>))}
    </select>
  )
}

function ChangedMark({ from }: { from: unknown }) {
  return <span className="small" style={{ color: 'var(--s2)', fontWeight: 500 }} title={`was ${normalize(from)}`}>was {normalize(from) || '""'}</span>
}
