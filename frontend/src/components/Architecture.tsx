import { useEffect, useState } from 'react'
import { api, type Param } from '../api'
import { initialValues, ParamForm } from './ParamForm'
import { ErrorBox } from './ui'

type Node = { name: string; type: string; params: number; own_params: number; trainable: number; out_shape: unknown
  extra: string; children?: Node[]; collapsed?: number }
type Arch = { ok: boolean; error?: string; device?: string; data_shape?: number[]; modality?: string; total_params?: number
  trainable_params?: number; size_mb?: number; tree?: Node; warnings?: string[] }
type Details = { id: string; name: string; description: string; source: string; class_name: string | null; class_doc: string
  class_source: string | null; class_file: string | null; extra_classes: { name: string; source: string }[]; notes: string[]
  framework: string | null; modalities: string[]; params: Param[] }

const SHAPES: Record<string, string> = { image: '3, 32, 32', audio: '1, 8000', video: '1, 16, 32, 32', tabular: '16', text: '128' }
const fmtP = (n: number) => n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}k` : String(n)
const shapeStr = (s: unknown): string => Array.isArray(s) ? (Array.isArray(s[0]) ? s.map(shapeStr).join(', ') : `(${s.join(', ')})`) : '—'

/** Architecture reference for a model plugin: parameters (with types, defaults, help), layer tree, source. */
export function ArchitectureView({ modelId }: { modelId: string }) {
  const [d, setD] = useState<Details>()
  const [vals, setVals] = useState<Record<string, unknown>>({})
  const [modality, setModality] = useState('')
  const [shape, setShape] = useState('')
  const [depth, setDepth] = useState(3)
  const [arch, setArch] = useState<Arch>()
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<unknown>()
  const [showAll, setShowAll] = useState(false)

  useEffect(() => {
    setD(undefined); setArch(undefined)
    api.get<Details>(`/models/${modelId}/details`).then((x) => {
      setD(x); setVals(initialValues(x.params))
      const m = x.modalities.includes('image') ? 'image' : x.modalities[0] ?? 'image'
      setModality(m); setShape(SHAPES[m] ?? '3, 32, 32')
    }).catch(setErr)
  }, [modelId])

  const run = async () => {
    setBusy(true); setErr(undefined)
    try {
      setArch(await api.post<Arch>(`/models/${modelId}/architecture`, {
        params: vals, modality, max_depth: depth,
        data_shape: modality === 'text' ? null : shape.split(',').map((s) => parseInt(s.trim(), 10)).filter((x) => !Number.isNaN(x)),
      }))
    } catch (e) { setErr(e) } finally { setBusy(false) }
  }
  useEffect(() => { if (d) run() }, [d, depth]) // eslint-disable-line react-hooks/exhaustive-deps

  if (!d) return err ? <ErrorBox error={err} /> : <div className="muted">Loading…</div>
  const netParams = d.params.filter((p) => ['Network', 'Generator', 'Discriminator', 'Model', 'Diffusion', 'VAE', 'GAN', 'Text'].includes(p.group))
  const shown = showAll ? d.params : netParams.length ? netParams : d.params.slice(0, 12)

  return (
    <div className="col" style={{ gap: 14 }}>
      <div>
        <div className="small text-2">{d.class_doc || d.description}</div>
        <div className="mono small muted" style={{ marginTop: 4 }}>{d.id}{d.class_file && ` · ${d.class_file}`}</div>
        {d.notes.length > 0 && <div className="small muted" style={{ marginTop: 4 }}>{d.notes.map((n) => `• ${n}`).join('  ')}</div>}
      </div>

      <div className="grid-2" style={{ alignItems: 'start' }}>
        <div className="col" style={{ gap: 10 }}>
          <div className="row" style={{ justifyContent: 'space-between' }}>
            <h3>Parameters {netParams.length > 0 && !showAll && <span className="muted small">(architecture)</span>}</h3>
            <button className="ghost sm" onClick={() => setShowAll((x) => !x)}>{showAll ? 'architecture only' : `all ${d.params.length}`}</button>
          </div>
          <ParamForm params={shown} values={vals} onChange={setVals} />
          <div className="row" style={{ gap: 8, alignItems: 'flex-end' }}>
            {d.modalities.length > 1 && <label className="field" style={{ width: 120 }}><span className="lbl">Data</span>
              <select value={modality} onChange={(e) => { setModality(e.target.value); setShape(SHAPES[e.target.value] ?? shape) }}>
                {d.modalities.map((m) => <option key={m}>{m}</option>)}</select></label>}
            {modality !== 'text' && <label className="field" style={{ width: 150 }}><span className="lbl">Sample shape</span>
              <input value={shape} onChange={(e) => setShape(e.target.value)} placeholder="C, H, W" /></label>}
            <button className="primary" disabled={busy} onClick={run}>{busy ? 'Building…' : 'Update'}</button>
          </div>
        </div>

        <div className="col" style={{ gap: 8 }}>
          <ErrorBox error={err} />
          {arch && !arch.ok && <div className="error-box small" style={{ whiteSpace: 'pre-wrap' }}>Could not build the model with these settings:{'\n'}{arch.error}</div>}
          {arch?.ok && (
            <>
              <div className="row" style={{ gap: 18 }}>
                <span className="tile" style={{ padding: 0 }}><span className="k">Parameters</span><span className="v num">{fmtP(arch.total_params!)}</span></span>
                <span className="tile" style={{ padding: 0 }}><span className="k">Trainable</span><span className="v num">{fmtP(arch.trainable_params!)}</span></span>
                <span className="tile" style={{ padding: 0 }}><span className="k">Weights</span><span className="v num">{arch.size_mb! < 1 ? `${(arch.size_mb! * 1000).toFixed(0)} KB` : `${arch.size_mb!.toFixed(1)} MB`}</span></span>
              </div>
              <div className="small muted">input {arch.modality === 'text' ? 'token ids' : `(batch, ${arch.data_shape!.join(', ')})`} ·
                {' '}{arch.device === 'meta' ? 'built without allocating memory (meta device)' : 'built on CPU'}</div>
            </>
          )}
        </div>
      </div>

      {arch?.ok && arch.tree && (
        <div className="card">
          <div className="card-head"><h3>Layers</h3>
            <label className="row small">depth <select value={depth} onChange={(e) => setDepth(Number(e.target.value))} style={{ width: 'auto' }}>
              {[1, 2, 3, 4, 5, 6, 8].map((x) => <option key={x}>{x}</option>)}</select></label></div>
          <div className="table-wrap" style={{ maxHeight: 520 }}>
            <table>
              <thead><tr><th>Layer</th><th>Type</th><th>Output shape</th><th className="num">Params</th><th>Config</th></tr></thead>
              <tbody><TreeRows node={arch.tree} depth={0} total={arch.total_params!} /></tbody>
            </table>
          </div>
        </div>
      )}

      {d.class_source && (
        <details>
          <summary className="small">Source of {d.class_name}{d.extra_classes.length ? ` and ${d.extra_classes.map((c) => c.name).join(', ')}` : ''}</summary>
          <pre className="log mono small" style={{ maxHeight: 500 }}>{d.class_source}{d.extra_classes.map((c) => '\n\n' + c.source).join('')}</pre>
        </details>
      )}
    </div>
  )
}

function TreeRows({ node, depth, total }: { node: Node; depth: number; total: number }) {
  const [open, setOpen] = useState(depth < 2)
  const kids = node.children ?? []
  const share = total ? node.params / total : 0
  return (
    <>
      <tr>
        <td className="mono small" style={{ paddingLeft: 8 + depth * 16, whiteSpace: 'nowrap' }}>
          {kids.length ? <button className="ghost sm" style={{ padding: '0 4px', minWidth: 18 }} onClick={() => setOpen((o) => !o)}>{open ? '▾' : '▸'}</button>
            : <span style={{ display: 'inline-block', width: 22 }} />}
          {node.name}{node.collapsed ? <span className="muted"> (+{node.collapsed})</span> : null}</td>
        <td className="small">{node.type}</td>
        <td className="mono small">{shapeStr(node.out_shape)}</td>
        <td className="num small" title={`${node.params.toLocaleString()} parameters`}>
          {node.params ? fmtP(node.params) : '—'}
          {depth > 0 && share >= 0.01 && <span className="muted"> · {(share * 100).toFixed(0)}%</span>}</td>
        <td className="mono small muted" style={{ maxWidth: 320, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={node.extra}>{node.extra}</td>
      </tr>
      {open && kids.map((k) => <TreeRows key={k.name} node={k} depth={depth + 1} total={total} />)}
    </>
  )
}
