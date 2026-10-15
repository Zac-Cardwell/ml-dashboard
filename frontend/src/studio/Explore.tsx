import { useState } from 'react'
import { ErrorBox } from '../components/ui'
import { colsOf, ResultTable } from './DataGrid'
import { fmtInt, pct, short, studio, type Dataset } from './types'
import { ChartView, EmbeddingPlot, MiniLine } from './viz'

const TASKS = [
  { key: 'cluster', label: 'Clusters', help: 'Find groups of similar rows. Features are standardized; missing values filled with the median.' },
  { key: 'anomaly', label: 'Anomalies', help: 'Score how unusual each row is; flag the top share as anomalies.' },
  { key: 'project', label: 'Reduce dims', help: 'PCA: how many directions carry the variance, and which columns drive them. t-SNE for a visual map only.' },
]
const ALGOS: Record<string, string[]> = { cluster: ['kmeans', 'gmm', 'hdbscan', 'dbscan', 'agglomerative'], anomaly: ['isolation_forest', 'lof'] }

/** Unsupervised workbench on a reproducible sample; a result can be saved as a recipe step (new derived dataset). */
export function ExploreTab({ ds, onCreated }: { ds: Dataset; onCreated: (id: string) => void }) {
  const numeric = colsOf(ds.schema, true).filter((c) => c !== ds.target)
  const [spec, setSpec] = useState<Record<string, any>>({ task: 'cluster', algorithm: 'kmeans', k: 4, sample_rows: 20000, projection: 'pca', columns: [], contamination: 0.01,
    min_cluster_size: 25, eps: 0.5, min_samples: 5, sweep: true, k_lo: 2, k_hi: 10, seed: 0, variance: 0.95, n_components: 0 })
  const [res, setRes] = useState<any>()
  const [err, setErr] = useState<unknown>()
  const [busy, setBusy] = useState(false)
  const [saveName, setSaveName] = useState('')
  const set = (k: string, v: any) => setSpec({ ...spec, [k]: v })
  const body = () => {
    const s: Record<string, any> = { ...spec }
    if (s.task === 'cluster' && s.sweep && ['kmeans', 'gmm'].includes(s.algorithm)) s.k_range = [s.k_lo, s.k_hi]
    if (s.task === 'project' && ds.target) s.color_by = ds.target
    return s
  }
  const run = () => { setBusy(true); setErr(undefined); studio.explore(ds.id, body()).then((r) => { setRes(r); setSaveName('') }).catch(setErr).finally(() => setBusy(false)) }
  const save = async (subset?: number) => {
    setErr(undefined)
    const label = subset == null ? (spec.task === 'cluster' ? 'clustered' : spec.task === 'anomaly' ? 'anomaly scored' : 'pca') : spec.task === 'cluster' ? `cluster ${subset}` : subset ? 'anomalies' : 'inliers'
    try {
      const d = await studio.apply({ dataset: ds.id, spec: { ...body(), name: spec.task === 'cluster' ? 'cluster' : 'anomaly' }, name: saveName || `${ds.name} · ${label}`, subset_value: subset ?? null })
      onCreated(d.id)
    } catch (e) { setErr(e) }
  }
  const algo = spec.algorithm
  return (
    <div className="explore">
      <div className="card card-pad col" style={{ gap: 10 }}>
        <div className="segmented" style={{ alignSelf: 'stretch' }}>{TASKS.map((t) => <button key={t.key} style={{ flex: 1 }} className={spec.task === t.key ? 'on' : ''}
          onClick={() => { setSpec({ ...spec, task: t.key, algorithm: ALGOS[t.key]?.[0] }); setRes(undefined) }}>{t.label}</button>)}</div>
        <div className="small muted">{TASKS.find((t) => t.key === spec.task)!.help}</div>
        {ALGOS[spec.task] && <label className="field"><span className="lbl">Algorithm</span>
          <select value={algo} onChange={(e) => set('algorithm', e.target.value)}>{ALGOS[spec.task].map((a) => <option key={a}>{a}</option>)}</select></label>}
        {spec.task === 'cluster' && ['kmeans', 'gmm', 'agglomerative'].includes(algo) && <label className="field"><span className="lbl">Clusters (k)</span>
          <input type="number" min={2} max={50} value={spec.k} onChange={(e) => set('k', parseInt(e.target.value) || 2)} /></label>}
        {spec.task === 'cluster' && ['kmeans', 'gmm'].includes(algo) && <div className="row small">
          <label className="check"><input type="checkbox" checked={spec.sweep} onChange={(e) => set('sweep', e.target.checked)} /> Sweep k from</label>
          <input type="number" className="sm-input" value={spec.k_lo} onChange={(e) => set('k_lo', parseInt(e.target.value) || 2)} /> to
          <input type="number" className="sm-input" value={spec.k_hi} onChange={(e) => set('k_hi', parseInt(e.target.value) || 10)} /></div>}
        {algo === 'hdbscan' && <label className="field"><span className="lbl">Min cluster size</span><input type="number" value={spec.min_cluster_size} onChange={(e) => set('min_cluster_size', parseInt(e.target.value) || 5)} /></label>}
        {algo === 'dbscan' && <div className="grid-tight"><label className="field"><span className="lbl">eps (z-units)</span><input type="number" step="any" value={spec.eps} onChange={(e) => set('eps', parseFloat(e.target.value) || 0.5)} /></label>
          <label className="field"><span className="lbl">Min samples</span><input type="number" value={spec.min_samples} onChange={(e) => set('min_samples', parseInt(e.target.value) || 5)} /></label></div>}
        {spec.task === 'anomaly' && <label className="field"><span className="lbl">Expected anomaly share</span><input type="number" step="any" min={0.001} max={0.5} value={spec.contamination} onChange={(e) => set('contamination', parseFloat(e.target.value) || 0.01)} /></label>}
        {spec.task === 'project' && <label className="field"><span className="lbl">PCA step keeps</span>
          <select value={spec.n_components ? 'n' : 'v'} onChange={(e) => set('n_components', e.target.value === 'n' ? 5 : 0)}><option value="v">enough components for a variance share</option><option value="n">a fixed number of components</option></select>
          {spec.n_components ? <input type="number" min={1} value={spec.n_components} onChange={(e) => set('n_components', parseInt(e.target.value) || 1)} />
            : <input type="number" step="any" min={0.5} max={0.999} value={spec.variance} onChange={(e) => set('variance', parseFloat(e.target.value) || 0.95)} />}</label>}
        <label className="field"><span className="lbl">2-D view</span>
          <select value={spec.projection} onChange={(e) => set('projection', e.target.value)}><option value="pca">PCA (fast, linear)</option><option value="tsne">t-SNE (slow, local structure — for looking only)</option></select></label>
        <div className="field"><span className="lbl">Features ({spec.columns.length || `all ${numeric.length} numeric`})</span>
          <div className="chips" style={{ maxHeight: 140, overflowY: 'auto' }}>{numeric.map((c) => {
            const on = spec.columns.includes(c)
            return <button key={c} type="button" className={`chip toggle${on ? ' on' : ''}`} onClick={() => set('columns', on ? spec.columns.filter((x: string) => x !== c) : [...spec.columns, c])}>{c}</button>
          })}</div>
          {numeric.length === 0 && <span className="help">No numeric columns — derive a dataset with a One-hot step first.</span>}</div>
        <label className="field"><span className="lbl">Sample rows</span><input type="number" min={500} max={200000} value={spec.sample_rows} onChange={(e) => set('sample_rows', parseInt(e.target.value) || 20000)} />
          <span className="help">Exploration runs on a reproducible random sample of {fmtInt(ds.rows)} rows.</span></label>
        <button className="primary" onClick={run} disabled={busy || numeric.length === 0}>{busy ? 'Running…' : 'Run'}</button>
        <ErrorBox error={err} />
      </div>

      <div className="col" style={{ gap: 14, minWidth: 0 }}>
        {!res && <div className="card empty">{busy ? 'Fitting…' : 'Configure and run an exploration.'}</div>}
        {res && <Result res={res} ds={ds} />}
        {res && (res.task !== 'cluster' || algo !== 'agglomerative') && (
          <div className="card card-pad col" style={{ gap: 8 }}>
            <h3>Keep this result</h3>
            <div className="small text-2">Saving creates a derived dataset whose recipe re-fits this {res.task === 'project' ? 'PCA' : algo} on the full table
              (training rows only, if the dataset has a split) and adds {res.task === 'cluster' ? 'a cluster column' : res.task === 'anomaly' ? 'score and flag columns' : 'component columns'}.
              Numbers can shift a little from this sample.</div>
            <div className="row">
              <input placeholder={`${ds.name} · …`} value={saveName} onChange={(e) => setSaveName(e.target.value)} style={{ flex: 1, minWidth: 200 }} />
              <button onClick={() => save()}>Save as dataset</button>
              {res.task === 'anomaly' && <><button onClick={() => save(1)}>Save anomalies only</button><button onClick={() => save(0)}>Save inliers only</button></>}
            </div>
            {res.task === 'cluster' && res.profiles && <div className="row small">Save one cluster as a subset:
              {res.profiles.filter((p: any) => p.cluster >= 0).map((p: any) => <button key={p.cluster} className="sm" onClick={() => save(p.cluster)}>cluster {p.cluster}</button>)}</div>}
          </div>
        )}
      </div>
    </div>
  )
}

function Metric({ k, v, hint }: { k: string; v: any; hint?: string }) {
  return <div className="metric" title={hint}><div className="k">{k}</div><div className="v num">{v == null ? '—' : typeof v === 'number' ? short(v) : v}</div></div>
}

function Result({ res, ds }: { res: any; ds: Dataset }) {
  const m = res.metrics ?? {}
  const sampleNote = `${fmtInt(res.sample_rows)} of ${fmtInt(res.total_rows)} rows · ${res.columns.length} features`
  if (res.task === 'cluster') {
    const best = res.sweep?.reduce((a: any, b: any) => ((b.silhouette ?? -1) > (a.silhouette ?? -1) ? b : a), res.sweep[0])
    return (
      <>
        <div className="card card-pad">
          <div className="metrics-row">
            <Metric k="Clusters" v={m.n_clusters} />
            <Metric k="Silhouette" v={m.silhouette} hint="−1…1, higher = tighter, better separated clusters. > 0.5 strong, 0.25–0.5 reasonable, < 0.25 weak" />
            <Metric k="Davies–Bouldin" v={m.davies_bouldin} hint="Lower is better" />
            <Metric k="Calinski–Harabasz" v={m.calinski_harabasz} hint="Higher is better" />
            {m.noise_share != null && m.noise_share > 0 && <Metric k="Noise" v={pct(m.noise_share)} hint="Rows HDBSCAN/DBSCAN left unassigned" />}
            {m.ari_vs_target != null && <Metric k={`ARI vs ${ds.target}`} v={m.ari_vs_target} hint="Agreement between clusters and the target classes: 0 = chance, 1 = identical" />}
          </div>
          <div className="small muted" style={{ marginTop: 6 }}>{sampleNote}</div>
        </div>
        {res.sweep && <div className="grid-2">
          <div className="card card-pad"><div className="small text-2">Silhouette by k (higher is better){best ? ` — best k = ${best.k}` : ''}</div>
            <MiniLine x={res.sweep.map((s: any) => s.k)} y={res.sweep.map((s: any) => s.silhouette)} xTitle="k" yTitle="silhouette" mark={best?.k} /></div>
          <div className="card card-pad"><div className="small text-2">{res.sweep[0]?.inertia != null ? 'Inertia by k (look for the elbow)' : 'BIC by k (lower is better)'}</div>
            <MiniLine x={res.sweep.map((s: any) => s.k)} y={res.sweep.map((s: any) => s.inertia ?? s.bic)} xTitle="k" yTitle={res.sweep[0]?.inertia != null ? 'inertia' : 'BIC'} /></div>
        </div>}
        <div className="card card-pad"><EmbeddingPlot x={res.points.x} y={res.points.y} label={res.points.label} title={`clusters in ${res.projection.toUpperCase()} space`} /></div>
        <div className="card">
          <div className="card-head"><h3>Cluster profiles</h3><span className="small muted">most distinguishing features (z-score of the cluster mean)</span></div>
          <div className="table-wrap"><table className="dense"><thead><tr><th>Cluster</th><th className="num">Rows</th><th className="num">Share</th><th>Distinguishing features</th></tr></thead>
            <tbody>{res.profiles.map((p: any) => <tr key={p.cluster}><td>{p.cluster < 0 ? <span className="muted">noise</span> : p.cluster}</td><td className="num">{fmtInt(p.size)}</td><td className="num">{pct(p.share)}</td>
              <td className="small">{p.distinguishing.map((d: any) => <span key={d.column} className={`zpill ${d.z > 0 ? 'pos' : 'neg'}`}>{d.column} {d.z > 0 ? '+' : ''}{d.z?.toFixed(2)}σ</span>)}</td></tr>)}</tbody></table></div>
        </div>
        {res.crosstab && <div className="card"><div className="card-head"><h3>Clusters vs {ds.target}</h3></div>
          <ResultTable index={false} columns={['cluster', ...res.crosstab.classes]} rows={res.crosstab.clusters.map((c: number, i: number) => [c, ...res.crosstab.counts[i]])} maxHeight={320} /></div>}
      </>
    )
  }
  if (res.task === 'anomaly') {
    const h = res.score_hist
    return (
      <>
        <div className="card card-pad">
          <div className="metrics-row">
            <Metric k="Flagged" v={fmtInt(m.flagged)} /><Metric k="Share" v={pct(m.flagged_share, 2)} />
            {m.auc_vs_minority_class != null && <Metric k={`AUC vs "${m.minority_class}"`} v={m.auc_vs_minority_class} hint="How well the score ranks the minority target class first (0.5 = chance)" />}
            {m.ap_vs_minority_class != null && <Metric k="Avg precision" v={m.ap_vs_minority_class} />}
          </div>
          <div className="small muted" style={{ marginTop: 6 }}>{sampleNote}</div>
        </div>
        <div className="grid-2">
          <div className="card card-pad"><div className="small text-2">Anomaly score distribution (right = more unusual)</div>
            <ChartView height={300} result={{ type: 'histogram', x: 'anomaly score', series: [{ name: 'rows', edges: h.edges, n: h.counts.reduce((a: number, b: number) => a + b, 0), values: h.counts }] }} /></div>
          <div className="card card-pad"><EmbeddingPlot x={res.points.x} y={res.points.y} label={res.points.label.map((l: number) => (l ? 'anomaly' : 'normal'))} height={300} /></div>
        </div>
        <div className="card"><div className="card-head"><h3>Most anomalous rows</h3></div>
          <ResultTable columns={res.top_rows.columns} rows={res.top_rows.rows} maxHeight={380} /></div>
      </>
    )
  }
  const ev: number[] = res.explained_variance_ratio
  const cum = ev.reduce((a: number[], v) => [...a, (a[a.length - 1] ?? 0) + v], [])
  const need = cum.findIndex((c) => c >= 0.95) + 1
  return (
    <>
      {res.note && <div className="callout warning small">{res.note}</div>}
      <div className="grid-2">
        <div className="card card-pad"><div className="small text-2">Cumulative explained variance — {need > 0 ? `${need} of ${res.columns.length} components reach 95%` : 'more components needed for 95%'}</div>
          <MiniLine x={cum.map((_, i) => i + 1)} y={cum} xTitle="components" yTitle="cumulative variance" height={260} mark={need || undefined} /></div>
        <div className="card card-pad"><EmbeddingPlot x={res.points.x} y={res.points.y} label={res.points.label} height={280} title={res.points.label ? `colored by ${ds.target}` : undefined} /></div>
      </div>
      <div className="card"><div className="card-head"><h3>Loadings</h3><span className="small muted">columns with the largest weight in each component</span></div>
        <div className="table-wrap"><table className="dense"><thead><tr><th>Component</th><th className="num">Variance</th><th>Top columns</th></tr></thead>
          <tbody>{res.loadings.map((l: any, i: number) => <tr key={l.component}><td className="mono">{l.component}</td><td className="num">{pct(ev[i])}</td>
            <td className="small">{l.top.map((t: any) => <span key={t.column} className={`zpill ${t.weight > 0 ? 'pos' : 'neg'}`}>{t.column} {t.weight.toFixed(2)}</span>)}</td></tr>)}</tbody></table></div>
      </div>
    </>
  )
}
