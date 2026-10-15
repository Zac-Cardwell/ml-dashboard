import { useEffect, useState } from 'react'
import { api, fmtNum, fmtTime } from '../api'
import { baseLayout, Plot, useTheme } from './Chart'
import { ErrorBox } from './ui'

export type ColProfile = { kind: string; missing: number; n_unique: number; mean?: number; std?: number; min?: number; max?: number; p50?: number
  zeros?: number; outliers?: number; hist?: { edges: number[]; counts: number[] } | null; top?: { value: string; share: number }[]; mean_length?: number }
export type Issue = { severity: string; kind: string; column: string | null; message: string }
type Report = { name: string; grade: string; task: string; modality: string; sizes: Record<string, number | null>; n_rows?: number; n_columns?: number
  columns: Record<string, ColProfile>; issues: Issue[]; target?: { kind: string; shares?: Record<string, number>; mean?: number }
  split_drift?: Record<string, { psi: number | null; status: string; psi_noise?: number }>; created_at: number; seconds: number }

const SEV: Record<string, string> = { critical: 'failed', warning: 'queued', info: '' }

export function IssueList({ issues }: { issues: Issue[] }) {
  if (!issues.length) return <div className="notice small">✓ No problems found.</div>
  return (
    <div className="col" style={{ gap: 4 }}>
      {issues.map((i, k) => (
        <div key={k} className="row small" style={{ flexWrap: 'nowrap', alignItems: 'flex-start' }}>
          <span className={`badge ${SEV[i.severity]}`} style={{ minWidth: 78 }}><span className="dot" />{i.severity}</span><span>{i.message}</span>
        </div>
      ))}
    </div>
  )
}

export function ShareBars({ shares, title }: { shares: Record<string, number>; title?: string }) {
  const theme = useTheme()
  const ks = Object.keys(shares)
  const h = Math.max(140, 40 + ks.length * 24)
  return (
    <div style={{ height: h }}>
      <Plot useResizeHandler style={{ width: '100%', height: '100%' }} config={{ displaylogo: false, responsive: true }}
        layout={{ ...baseLayout(theme, title ?? 'share'), height: h, showlegend: false, hovermode: 'closest', margin: { l: 10, r: 50, t: 8, b: 36 },
          xaxis: { ...baseLayout(theme, '').xaxis, tickformat: '.0%' }, yaxis: { ...baseLayout(theme, '').yaxis, automargin: true, type: 'category', autorange: 'reversed' } }}
        data={[{ type: 'bar', orientation: 'h', y: ks, x: ks.map((k) => shares[k]), marker: { color: theme.series[0] }, text: ks.map((k) => `${(shares[k] * 100).toFixed(1)}%`) as any,
          textposition: 'outside', cliponaxis: false, hovertemplate: '%{y}: %{x:.1%}<extra></extra>' }]} />
    </div>
  )
}

export function ColumnTable({ columns }: { columns: Record<string, ColProfile> }) {
  return (
    <div className="table-wrap" style={{ maxHeight: 420 }}>
      <table>
        <thead><tr><th>Column</th><th>Type</th><th className="num">Missing</th><th className="num">Unique</th><th>Summary</th></tr></thead>
        <tbody>{Object.entries(columns).map(([k, c]) => (
          <tr key={k}>
            <td className="mono">{k}</td><td className="small">{c.kind}</td>
            <td className="num" style={c.missing >= 0.05 ? { color: 'var(--critical-text)', fontWeight: 600 } : undefined}>{(c.missing * 100).toFixed(1)}%</td>
            <td className="num">{c.n_unique.toLocaleString()}</td>
            <td className="small text-2">{c.kind === 'numeric'
              ? <>mean {fmtNum(c.mean)} · median {fmtNum(c.p50)} · range {fmtNum(c.min)} – {fmtNum(c.max)}{c.outliers ? ` · ${(c.outliers * 100).toFixed(1)}% outliers` : ''}</>
              : (c.top ?? []).slice(0, 4).map((t) => `${t.value} (${(t.share * 100).toFixed(0)}%)`).join(', ')}</td>
          </tr>))}</tbody>
      </table>
    </div>
  )
}

export function QualityReport({ datasetId }: { datasetId: string }) {
  const [r, setR] = useState<Report>()
  const [err, setErr] = useState<unknown>()
  const [busy, setBusy] = useState(false)
  const load = (refresh = false) => { setBusy(true); setErr(undefined); api.post<Report>(`/data/static/${datasetId}/report`, { refresh }).then(setR).catch(setErr).finally(() => setBusy(false)) }
  useEffect(() => { load() }, [datasetId]) // eslint-disable-line react-hooks/exhaustive-deps
  if (err) return <ErrorBox error={err} />
  if (!r) return <div className="muted">{busy ? 'Analyzing (the first run may download the dataset)…' : ''}</div>
  const drifted = Object.entries(r.split_drift ?? {}).filter(([, d]) => d.psi != null).sort((a, b) => (b[1].psi ?? 0) - (a[1].psi ?? 0)).slice(0, 8)
  return (
    <div className="col" style={{ gap: 14 }}>
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <div className="row">
          <span className="tile" style={{ padding: 0 }}><span className="v" style={{ fontSize: '2rem' }}>{r.grade}</span></span>
          <div className="small text-2">{r.modality} · {r.task}{r.n_rows != null && ` · ${r.n_rows.toLocaleString()} rows`}{r.n_columns != null && ` · ${r.n_columns} features`}
            <div>splits: {Object.entries(r.sizes).map(([k, v]) => `${k} ${v?.toLocaleString() ?? '—'}`).join(' · ')}</div>
            <div className="muted">checked {fmtTime(r.created_at)} in {r.seconds}s</div></div>
        </div>
        <button className="sm" disabled={busy} onClick={() => load(true)}>{busy ? 'Re-checking…' : 'Re-check'}</button>
      </div>
      <IssueList issues={r.issues} />
      <div className="grid-2">
        {r.target?.shares && <div className="card chart-card"><h3>Target distribution</h3><ShareBars shares={r.target.shares} /></div>}
        {drifted.length > 0 && (
          <div className="card card-pad">
            <h3 style={{ marginBottom: 6 }}>Train vs. test split (PSI)</h3>
            <div className="small text-2" style={{ marginBottom: 6 }}>Flagged only above both 0.1 and the sampling-noise level for this split size.</div>
            <table><tbody>{drifted.map(([k, d]) => (
              <tr key={k}><td className="mono small">{k}</td><td className="num small">{d.psi!.toFixed(3)}</td><td className="small muted">noise ≤ {d.psi_noise?.toFixed(3)}</td>
                <td>{d.status !== 'ok' && <span className="badge queued"><span className="dot" />shifted</span>}</td></tr>))}</tbody></table>
          </div>
        )}
      </div>
      <div className="card"><div className="card-head"><h3>Columns</h3></div><ColumnTable columns={r.columns} /></div>
    </div>
  )
}
