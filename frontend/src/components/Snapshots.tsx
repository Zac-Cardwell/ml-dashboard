import { AudioSet } from './Generate'
import { useEffect, useMemo, useState } from 'react'
import { api, fmtNum, type Media } from '../api'
import { baseLayout, Heatmap, Plot, useTheme, withAlpha } from './Chart'
import { ErrorBox, usePoll } from './ui'

const TITLES: Record<string, string> = {
  episode: 'Agent behaviour (evaluation episodes)',
  decision_boundary: 'Decision boundary',
  curve_fit: 'Fitted function',
  predictions: 'Sample predictions',
  samples: 'Model outputs on fixed validation examples',
  generated: 'Generated samples (same noise / prompts every snapshot)',
  final_samples: 'Final samples (saved weights)',
  denoising: 'Denoising process (noise → sample)',
  reconstructions: 'Reconstructions (top row: input · bottom row: model)',
  val_confusion: 'Validation confusion matrix',
}
const ORDER = ['generated', 'final_samples', 'denoising', 'reconstructions', 'episode', 'samples', 'decision_boundary', 'curve_fit', 'predictions', 'val_confusion']

/** Learning snapshots: one card per snapshot type, each with a seed picker and a step scrubber. */
export function Snapshots({ sessionId, version, stepLabel }: { sessionId: string; version: number; stepLabel: string }) {
  const [media, err] = usePoll(() => api.get<Media[]>(`/sessions/${sessionId}/media`), 0, [sessionId, version])
  const groups = useMemo(() => {
    const g = new Map<string, Media[]>()
    for (const m of media ?? []) {
      if (!g.has(m.name)) g.set(m.name, [])
      g.get(m.name)!.push(m)
    }
    return [...g.entries()].sort((a, b) => (ORDER.indexOf(a[0]) + 99) % 99 - (ORDER.indexOf(b[0]) + 99) % 99)
  }, [media])

  if (err) return <ErrorBox error={err} />
  if (!media) return <div className="muted">Loading…</div>
  if (!media.length) {
    return (
      <div className="card empty">
        No learning snapshots yet. They are recorded during training when “Record learning snapshots” is on:
        episode videos for RL, generated text for NLP, prediction galleries for images, and decision boundaries
        or fitted curves for 2-feature / 1-feature tabular data.
      </div>
    )
  }
  return <div className="col" style={{ gap: 16 }}>{groups.map(([name, items]) => <SnapshotCard key={name} name={name} items={items} stepLabel={stepLabel} />)}</div>
}

function SnapshotCard({ name, items, stepLabel }: { name: string; items: Media[]; stepLabel: string }) {
  const seeds = useMemo(() => [...new Set(items.map((m) => m.seed))].sort((a, b) => a - b), [items])
  const [seed, setSeed] = useState(seeds[0])
  const list = useMemo(() => items.filter((m) => m.seed === seed).sort((a, b) => a.step - b.step || a.id - b.id), [items, seed])
  const [idx, setIdx] = useState(list.length - 1)
  const [playing, setPlaying] = useState(false)
  const follow = idx >= list.length - 2   // stay on the newest snapshot while training adds more

  useEffect(() => { if (follow) setIdx(list.length - 1) }, [list.length]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!playing) return
    const isVideo = list[0]?.kind === 'video'
    const t = setInterval(() => setIdx((i) => {
      if (i >= list.length - 1) { setPlaying(false); return i }
      return i + 1
    }), isVideo ? 4000 : 900)
    return () => clearInterval(t)
  }, [playing, list])

  const cur = list[Math.min(idx, list.length - 1)]
  if (!cur) return null
  const label = cur.data?.label === 'final' ? 'final' : cur.data?.label?.startsWith?.('eval') ? `episode ${cur.step}` : `${stepLabel} ${fmtNum(cur.step)}`

  return (
    <div className="card">
      <div className="card-head" style={{ flexWrap: 'wrap' }}>
        <h3>{TITLES[name] ?? name.replace(/_/g, ' ')}</h3>
        <div className="row small">
          {seeds.length > 1 && (
            <select style={{ width: 'auto' }} value={seed} onChange={(e) => { setSeed(Number(e.target.value)); setIdx(0) }}>
              {seeds.map((s) => <option key={s} value={s}>seed {s}</option>)}
            </select>
          )}
        </div>
      </div>
      <div className="card-pad col" style={{ gap: 12 }}>
        {list.length > 1 && (
          <div className="row" style={{ flexWrap: 'nowrap' }}>
            <button className="sm" onClick={() => { if (idx >= list.length - 1) setIdx(0); setPlaying((p) => !p) }} aria-label={playing ? 'pause' : 'play'}>
              {playing ? '❚❚' : '▶'}
            </button>
            <input type="range" min={0} max={list.length - 1} value={Math.min(idx, list.length - 1)}
              onChange={(e) => { setPlaying(false); setIdx(Number(e.target.value)) }} style={{ flex: 1 }} aria-label="snapshot" />
            <span className="small num text-2" style={{ minWidth: 150, textAlign: 'right' }}>{label} · {idx + 1}/{list.length}</span>
          </div>
        )}
        <SnapshotView m={cur} />
      </div>
    </div>
  )
}

export function SnapshotView({ m }: { m: Media }) {
  switch (m.kind) {
    case 'video': return <VideoSnap m={m} />
    case 'decision_boundary': return <BoundarySnap data={m.data} />
    case 'curve_fit': return <CurveSnap data={m.data} />
    case 'image_predictions': return <ImageGrid data={m.data} />
    case 'text_samples': return <TextSamples data={m.data} />
    case 'confusion_matrix': return <Heatmap labels={m.data.labels} matrix={m.data.matrix} />
    case 'image': return <img src={`/api/runs/${m.run_id}/files/${m.path}?inline=true`} alt={m.name} style={{ width: 'auto', maxWidth: 'min(100%, 560px)', imageRendering: 'pixelated' }} />
    case 'audio': return <AudioSet spectrogram={`/api/runs/${m.run_id}/files/${m.path}?inline=true`}
      clips={(m.data?.clips ?? []).map((c: string) => `/api/runs/${m.run_id}/files/${c}?inline=true`)}
      specs={(m.data?.specs ?? []).map((c: string) => `/api/runs/${m.run_id}/files/${c}?inline=true`)} />
    default: return <pre className="log">{JSON.stringify(m.data, null, 2)}</pre>
  }
}

function VideoSnap({ m }: { m: Media }) {
  return (
    <div className="col" style={{ gap: 6, alignItems: 'flex-start' }}>
      <video key={m.id} src={`/api/runs/${m.run_id}/files/${m.path}?inline=true`} controls autoPlay muted playsInline loop
        style={{ maxWidth: '100%', maxHeight: 420, borderRadius: 8, background: '#000' }} />
      {m.data && <span className="small text-2 num">episode reward <b>{fmtNum(m.data.reward)}</b> · {m.data.length} steps</span>}
    </div>
  )
}

function BoundarySnap({ data }: { data: { x: number[]; y: number[]; z: number[][]; class_names: string[]; points: { x: number[]; y: number[]; label: number[] } } }) {
  const theme = useTheme()
  const k = data.class_names.length
  const colors = data.class_names.map((_, i) => theme.series[i % 8])
  const scale = colors.flatMap((c, i) => {
    const lo = k > 1 ? i / k : 0, hi = k > 1 ? (i + 1) / k : 1
    return [[lo, withAlpha(c, 0.22)], [hi, withAlpha(c, 0.22)]] as [number, string][]
  })
  const traces: Plotly.Data[] = [
    { type: 'heatmap', x: data.x, y: data.y, z: data.z, zmin: -0.5, zmax: k - 0.5, colorscale: scale, showscale: false, hoverinfo: 'skip' } as Plotly.Data,
    ...data.class_names.map((name, ci) => {
      const idx = data.points.label.map((l, i) => (l === ci ? i : -1)).filter((i) => i >= 0)
      return { type: 'scatter', mode: 'markers', name, x: idx.map((i) => data.points.x[i]), y: idx.map((i) => data.points.y[i]),
        marker: { size: 7, color: colors[ci], line: { color: theme.surface, width: 1 } } } as Plotly.Data
    }),
  ]
  const layout = { ...baseLayout(theme, 'x₀', 'x₁'), height: 380, hovermode: 'closest' as const }
  return <div style={{ height: 380, maxWidth: 560 }}><Plot data={traces} layout={layout} useResizeHandler style={{ width: '100%', height: '100%' }} config={{ displaylogo: false, responsive: true }} /></div>
}

function CurveSnap({ data }: { data: { x: number[]; pred: number[]; points: { x: number[]; y: number[] } } }) {
  const theme = useTheme()
  const traces: Plotly.Data[] = [
    { type: 'scatter', mode: 'markers', name: 'data', x: data.points.x, y: data.points.y, marker: { size: 5, color: withAlpha(theme.muted, 0.6) } },
    { type: 'scatter', mode: 'lines', name: 'model', x: data.x, y: data.pred, line: { color: theme.series[0], width: 2.5 } },
  ]
  return <div style={{ height: 340, maxWidth: 640 }}><Plot data={traces} layout={{ ...baseLayout(theme, 'x', 'y'), height: 340 }} useResizeHandler style={{ width: '100%', height: '100%' }} config={{ displaylogo: false, responsive: true }} /></div>
}

function ImageGrid({ data }: { data: { items: { img: string; label: string; pred: string; conf: number; correct: boolean }[] } }) {
  const right = data.items.filter((i) => i.correct).length
  return (
    <div className="col" style={{ gap: 8 }}>
      <span className="small text-2">{right}/{data.items.length} correct on these fixed examples</span>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(110px, 1fr))', gap: 10 }}>
        {data.items.map((it, i) => (
          <figure key={i} style={{ margin: 0, textAlign: 'center' }}>
            <img src={`data:image/png;base64,${it.img}`} alt={it.label} style={{ width: 96, height: 96, objectFit: 'contain', imageRendering: 'pixelated', borderRadius: 6, background: 'var(--surface-2)' }} />
            <figcaption className="small" style={{ lineHeight: 1.3 }}>
              <div style={{ color: it.correct ? 'var(--good-text)' : 'var(--critical-text)', fontWeight: 600 }}>{it.correct ? '✓' : '✗'} {it.pred}</div>
              <div className="muted">{(it.conf * 100).toFixed(0)}%{!it.correct && ` · true: ${it.label}`}</div>
            </figcaption>
          </figure>
        ))}
      </div>
    </div>
  )
}

function TextSamples({ data }: { data: { columns: string[]; rows: unknown[][] } }) {
  const predCol = data.columns.indexOf('prediction')
  return (
    <div className="table-wrap">
      <table>
        <thead><tr>{data.columns.map((c) => <th key={c}>{c}</th>)}</tr></thead>
        <tbody>
          {data.rows.map((r, i) => (
            <tr key={i}>{r.map((c, j) => <td key={j} className="small" style={j === predCol ? { fontWeight: 600 } : { color: 'var(--text-2)' }}>{String(c)}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/** Live frames from a running RL environment. */
export function LiveView({ runs }: { runs: { id: string; seed: number; status: string }[] }) {
  const running = runs.filter((r) => r.status === 'running')
  const [sel, setSel] = useState<string>()
  const [tick, setTick] = useState(0)
  const [ok, setOk] = useState(true)
  const run = running.find((r) => r.id === sel) ?? running[0]
  useEffect(() => {
    if (!run) return
    const t = setInterval(() => setTick((x) => x + 1), 250)
    return () => clearInterval(t)
  }, [run])
  if (!run) return null
  return (
    <div className="card">
      <div className="card-head">
        <h3>Live view <span className="badge running" style={{ marginLeft: 6 }}><span className="dot" />training env · seed {run.seed}</span></h3>
        {running.length > 1 && (
          <select style={{ width: 'auto' }} value={run.id} onChange={(e) => setSel(e.target.value)}>
            {running.map((r) => <option key={r.id} value={r.id}>seed {r.seed}</option>)}
          </select>
        )}
      </div>
      <div className="card-pad">
        <img src={`/api/runs/${run.id}/live.jpg?t=${tick}`} alt="live environment frame" onError={() => setOk(false)} onLoad={() => setOk(true)}
          style={{ maxWidth: '100%', maxHeight: 360, borderRadius: 8, background: '#000', display: ok ? 'block' : 'none' }} />
        {!ok && <div className="muted small">Waiting for frames… (live preview may be off for this session)</div>}
      </div>
    </div>
  )
}
