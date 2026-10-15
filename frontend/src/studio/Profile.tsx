import { useEffect, useState } from 'react'
import { fmtBytes } from '../api'
import { baseLayout, Plot, useTheme } from '../components/Chart'
import { ErrorBox, Modal } from '../components/ui'
import { Tile } from './Studio'
import { fmtInt, pct, short, studio, type ColProfile, type Dataset, type Profile } from './types'
import { ChartView, CorrMatrix, HBars, Sparkbars, TopBars } from './viz'

const SEV: Record<string, string> = { critical: 'failed', warning: 'queued', info: '' }

export function ProfileTab({ ds }: { ds: Dataset }) {
  const [p, setP] = useState<Profile>()
  const [err, setErr] = useState<unknown>()
  const [busy, setBusy] = useState(false)
  const [col, setCol] = useState<string>()
  const [corrKind, setCorrKind] = useState<'pearson' | 'spearman'>('pearson')
  const load = (refresh = false) => { setBusy(true); setErr(undefined); studio.profile(ds.id, refresh).then(setP).catch(setErr).finally(() => setBusy(false)) }
  useEffect(() => { setP(undefined); load() }, [ds.id, ds.version, ds.target]) // eslint-disable-line react-hooks/exhaustive-deps
  if (err) return <ErrorBox error={err} />
  if (!p) return <div className="card card-pad muted">{busy ? 'Profiling (column statistics run over every row; correlations on a sample)…' : ''}</div>
  const cols = Object.entries(p.columns)
  const ta = p.target_analysis
  return (
    <div className="col" style={{ gap: 14 }}>
      <div className="grid-6">
        <Tile k="Rows" v={fmtInt(p.rows)} s={p.split ? Object.entries(p.split).map(([k, v]) => `${k} ${fmtInt(v)}`).join(' · ') : 'no split column'} />
        <Tile k="Columns" v={cols.length} s={kindsSummary(p)} />
        <Tile k="Missing cells" v={pct(p.missing_cells)} s={`${pct(p.missingness.rows_with_missing)} of rows have a gap`} />
        <Tile k="Duplicate rows" v={fmtInt(p.duplicates)} s={pct(p.duplicates / Math.max(p.rows, 1))} />
        <Tile k="Target" v={<span className="mono" style={{ fontSize: '1rem' }}>{p.target ?? '—'}</span>} s={ta ? ta.task : 'set one to analyze features against it'} />
        <Tile k="Stored" v={p.bytes ? fmtBytes(p.bytes) : '—'} s={p.sampled ? `pairwise stats on ${fmtInt(p.sample_rows)}-row sample` : 'all rows'} />
      </div>

      <div className="card">
        <div className="card-head"><h3>Data quality</h3>
          <button className="sm ghost" onClick={() => load(true)} disabled={busy}>{busy ? 'Profiling…' : 'Re-profile'}</button></div>
        {p.alerts.length === 0 ? <div className="card-pad small">No issues found.</div> : (
          <div className="card-pad col" style={{ gap: 4 }}>{p.alerts.map((a, i) => (
            <div key={i} className="row small" style={{ flexWrap: 'nowrap', alignItems: 'flex-start' }}>
              <span className={`badge ${SEV[a.severity]}`} style={{ minWidth: 78 }}><span className="dot" />{a.severity}</span>
              <span>{a.message}</span>
              {a.column && p.columns[a.column] && <button className="sm ghost" style={{ marginLeft: 'auto' }} onClick={() => setCol(a.column!)}>inspect</button>}
            </div>))}</div>
        )}
      </div>

      <div className="card">
        <div className="card-head"><h3>Columns</h3><span className="small muted">click a row for its full distribution</span></div>
        <div className="table-wrap" style={{ maxHeight: 560 }}>
          <table className="dense">
            <thead><tr><th>Column</th><th>Kind</th><th>Missing</th><th className="num">Distinct</th><th>Distribution</th><th>Summary</th></tr></thead>
            <tbody>{cols.map(([name, c]) => (
              <tr key={name} className="clickable" onClick={() => setCol(name)}>
                <td><span className="mono">{name}</span>{name === p.target && <span className="tag" style={{ marginLeft: 6 }}>target</span>}
                  <div className="muted small">{c.type}</div></td>
                <td><span className="tag">{c.kind}</span></td>
                <td style={{ minWidth: 110 }}><MissBar v={c.missing} /></td>
                <td className="num">{fmtInt(c.unique)}</td>
                <td>{c.hist ? <Sparkbars counts={c.hist.counts} /> : c.top ? <TopBars items={c.top} /> : null}</td>
                <td className="small text-2">{summary(c)}</td>
              </tr>))}</tbody>
          </table>
        </div>
      </div>

      <div className="grid-2" style={{ alignItems: 'start' }}>
        {ta && (
          <div className="card">
            <div className="card-head"><h3>What predicts {p.target}?</h3><span className="small muted">mutual information (sample)</span></div>
            <div className="card-pad">
              <HBars labels={ta.associations.slice(0, 15).map((a) => a.column)} values={ta.associations.slice(0, 15).map((a) => a.mutual_info ?? 0)} title="mutual information (nats)" />
              <div className="small muted">Higher = the column alone tells more about the target. Catches non-linear relationships; doesn't account for
                redundancy between columns. Near-perfect scores usually mean leakage.</div>
            </div>
          </div>
        )}
        {ta?.balance && (
          <div className="card">
            <div className="card-head"><h3>Class balance</h3></div>
            <div className="card-pad"><HBars labels={Object.keys(ta.balance)} values={Object.values(ta.balance)} format=".1%" title="share of rows" /></div>
          </div>
        )}
        {p.correlations && (
          <div className="card" style={{ gridColumn: ta ? undefined : '1 / -1' }}>
            <div className="card-head"><h3>Correlations</h3>
              <div className="segmented sm">{(['pearson', 'spearman'] as const).map((k) => <button key={k} className={corrKind === k ? 'on' : ''} onClick={() => setCorrKind(k)}>{k}</button>)}</div></div>
            <div className="card-pad">
              <CorrMatrix columns={p.correlations.columns} matrix={p.correlations[corrKind]} />
              {p.correlations.strong_pairs.length > 0 && <div className="small" style={{ marginTop: 6 }}>Strongly related: {p.correlations.strong_pairs.slice(0, 6).map((s) =>
                <span key={s.a + s.b} className="tag mono" style={{ marginRight: 4 }}>{s.a} ~ {s.b} {s.r.toFixed(2)}</span>)}</div>}
            </div>
          </div>
        )}
        {Object.keys(p.missingness.columns).length > 0 && (
          <div className="card">
            <div className="card-head"><h3>Missing values</h3></div>
            <div className="card-pad">
              <HBars labels={Object.keys(p.missingness.columns)} values={Object.values(p.missingness.columns)} format=".1%" title="share missing" />
              {p.missingness.co_missing && <>
                <div className="small text-2" style={{ margin: '8px 0 4px' }}>Do gaps occur together? (correlation of missing-value indicators)</div>
                <CorrMatrix columns={p.missingness.co_missing.columns} matrix={p.missingness.co_missing.matrix} height={260} />
              </>}
            </div>
          </div>
        )}
        {p.time && <TimeCoverage t={p.time} />}
      </div>
      {col && <ColumnModal ds={ds} name={col} c={p.columns[col]} target={p.target} onClose={() => setCol(undefined)} />}
    </div>
  )
}

function kindsSummary(p: Profile) {
  const k: Record<string, number> = {}
  Object.values(p.columns).forEach((c) => { k[c.kind] = (k[c.kind] ?? 0) + 1 })
  return Object.entries(k).map(([a, b]) => `${b} ${a}`).join(' · ')
}

function MissBar({ v }: { v: number }) {
  const color = v >= 0.5 ? 'var(--critical)' : v >= 0.05 ? 'var(--warning)' : 'var(--s3)'
  return (
    <div className="row small" style={{ gap: 6, flexWrap: 'nowrap' }}>
      <span style={{ width: 54, height: 6, background: 'var(--surface-3)', borderRadius: 3, overflow: 'hidden', flex: 'none' }}>
        <span style={{ display: 'block', width: `${Math.max(v > 0 ? 3 : 0, v * 100)}%`, height: '100%', background: color }} /></span>
      <span className="num">{pct(v)}</span>
    </div>
  )
}

function summary(c: ColProfile) {
  if (c.mean != null) return <>μ {short(c.mean)} · median {short(c.median)} · σ {short(c.std)} · [{short(c.min)}, {short(c.max)}]{c.outliers ? ` · ${pct(c.outliers)} outliers` : ''}</>
  if (c.kind === 'datetime') return <>{String(c.min).slice(0, 19)} → {String(c.max).slice(0, 19)}</>
  if (c.top?.length) return <>top: {c.top.slice(0, 2).map((t) => `${t.value} (${pct(t.share, 0)})`).join(', ')}{c.avg_length ? ` · avg length ${c.avg_length.toFixed(0)}` : ''}</>
  return null
}

function TimeCoverage({ t }: { t: NonNullable<Profile['time']> }) {
  const theme = useTheme()
  const layout: any = { ...baseLayout(theme, `${t.column} (per ${t.unit})`, 'rows'), height: 240, showlegend: false }
  return (
    <div className="card">
      <div className="card-head"><h3>Rows over time</h3><span className="small muted mono">{t.column}</span></div>
      <div className="card-pad" style={{ height: 260 }}>
        <Plot data={[{ type: 'scatter', mode: 'lines', x: t.periods, y: t.counts, line: { color: theme.series[0], width: 2 }, fill: 'tozeroy', fillcolor: 'rgba(42,120,214,0.08)' } as any]}
          layout={layout} useResizeHandler style={{ width: '100%', height: '100%' }} config={{ displaylogo: false, responsive: true }} />
      </div>
    </div>
  )
}

function ColumnModal({ ds, name, c, target, onClose }: { ds: Dataset; name: string; c: ColProfile; target: string | null; onClose: () => void }) {
  const [d, setD] = useState<any>()
  const [by, setBy] = useState<string>(target && target !== name ? target : '')
  const [err, setErr] = useState<unknown>()
  useEffect(() => { setD(undefined); studio.column(ds.id, name, by || undefined).then(setD).catch(setErr) }, [name, by]) // eslint-disable-line react-hooks/exhaustive-deps
  const chart = d?.hist ? (d.by ? {
    type: 'histogram', x: name, normalized: true,
    series: Object.entries(d.by.groups).map(([g, h]: [string, any]) => {
      const tot = h.counts.reduce((a: number, b: number) => a + b, 0) || 1
      return { name: `${d.by.column} = ${g}`, edges: h.edges, n: tot, values: h.counts.map((x: number) => x / tot) }
    }),
  } : { type: 'histogram', x: name, normalized: false, series: [{ name, edges: d.hist.edges, n: d.hist.counts.reduce((a: number, b: number) => a + b, 0), values: d.hist.counts }] }) : null
  return (
    <Modal title={name} onClose={onClose}>
      <div className="col" style={{ gap: 12 }}>
        <div className="row small" style={{ gap: 8 }}>
          <span className="tag">{c.kind}</span><span className="muted">{c.type}</span>
          <span>missing {pct(c.missing)} · {fmtInt(c.unique)} distinct</span>
          {c.hist && <label className="row" style={{ marginLeft: 'auto', gap: 6 }}>split by
            <select style={{ width: 'auto' }} value={by} onChange={(e) => setBy(e.target.value)}><option value="">—</option>
              {ds.schema.filter((x) => x.name !== name).map((x) => <option key={x.name}>{x.name}</option>)}</select></label>}
        </div>
        <ErrorBox error={err} />
        {chart && <ChartView result={chart as any} height={320} />}
        {c.mean != null && (
          <div className="kv small" style={{ gridTemplateColumns: 'repeat(6, auto)' }}>
            {[['min', c.min], ['1%', c.p01], ['Q1', c.q1], ['median', c.median], ['Q3', c.q3], ['99%', c.p99], ['max', c.max], ['mean', c.mean], ['std', c.std],
              ['skew', c.skew], ['zeros', pct(c.zeros)], ['negative', pct(c.negative)]].map(([k, v]) => [<div key={String(k)} className="muted">{k}</div>,
              <div key={k + 'v'} className="num">{typeof v === 'number' ? short(v) : String(v ?? '—')}</div>])}
          </div>
        )}
        {d?.top && !d.hist && (
          <div className="table-wrap" style={{ maxHeight: 360 }}><table className="dense"><thead><tr><th>Value</th><th className="num">Rows</th><th className="num">Share</th></tr></thead>
            <tbody>{d.top.map((t: any) => <tr key={String(t.value)}><td className="mono small">{t.value ?? '∅'}</td><td className="num">{fmtInt(t.count)}</td>
              <td className="num">{pct(t.count / Math.max(ds.rows ?? 1, 1))}</td></tr>)}</tbody></table></div>
        )}
      </div>
    </Modal>
  )
}
