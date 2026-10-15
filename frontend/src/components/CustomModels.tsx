import { useState } from 'react'
import { api, type PluginMeta } from '../api'
import { usePlugins } from '../App'
import { ArchitectureView } from './Architecture'
import { ConfirmButton, ErrorBox, Modal, usePoll } from './ui'

type LinkedFile = { file: string; kind: string; plugins?: string[]; notes?: string[]; errors?: string[] }
type Linked = { path: string; name: string; exists?: boolean; files: LinkedFile[] }
type Scan = { path: string; files: { file: string; error?: string; classes?: { name: string; bases: string[]; torch: boolean }[]
  functions?: { name: string }[]; has_main?: boolean; suggest?: string | null; lines?: number }[] }
type Loss = { name: string; kind: string; doc: string; file?: string; source?: string }

const ROLE = (m: PluginMeta) => m.id.startsWith('gen.diffusion') ? 'diffusion' : m.id.startsWith('gen.vae') ? 'VAE'
  : m.id.startsWith('gen.gan') ? 'GAN' : m.id.startsWith('gen.lm') ? 'language model' : m.id.startsWith('gen.custom') ? 'custom generative'
  : m.framework === 'sb3' ? 'RL policy' : m.framework === 'torch' ? 'supervised' : m.framework ?? ''

/** Library tab: your own models (uploaded or from linked project folders) with architecture reference, losses. */
export function CustomModels() {
  const { plugins, reload } = usePlugins()
  const [links, , reloadLinks] = usePoll(() => api.get<Linked[]>('/plugins/linked'), 0)
  const [losses] = usePoll(() => api.get<{ losses: Loss[] }>('/losses'), 0)
  const [arch, setArch] = useState<PluginMeta>()
  const [adding, setAdding] = useState(false)
  const [lossSrc, setLossSrc] = useState<Loss>()
  const mine = (plugins?.models ?? []).filter((m) => m.source !== 'builtin')
  const builtinGen = (plugins?.models ?? []).filter((m) => m.source === 'builtin' && m.framework === 'generative')
  const refresh = () => { reload(); reloadLinks() }

  return (
    <div className="col" style={{ gap: 16 }}>
      <div className="card">
        <div className="card-head">
          <h3>Linked project folders</h3>
          <button className="sm primary" onClick={() => setAdding(true)}>+ Link a folder</button>
        </div>
        <div className="card-pad small text-2" style={{ paddingTop: 0 }}>
          Use models from an existing project without copying or changing it: chosen files are imported in place (the folder is
          put on the import path, so its own imports keep working) and re-read on every plugin reload. Nothing is written into the folder.
        </div>
        {!links?.length ? <div className="empty small">No linked folders.</div> : (
          <table><tbody>{links.map((l) => (
            <tr key={l.path}>
              <td><b>{l.name}</b><div className="mono small muted">{l.path}{l.exists === false && ' — not found'}</div></td>
              <td className="small">{l.files.map((f) => (
                <div key={f.file}><span className="mono">{f.file}</span> <span className="tag">{f.kind}</span>
                  {f.plugins?.length ? <span className="muted"> → {f.plugins.length} model{f.plugins.length > 1 ? 's' : ''}</span> : null}
                  {f.errors?.length ? <span style={{ color: 'var(--critical-text)' }}> · failed to load</span> : null}
                  {f.notes?.map((n) => <div key={n} className="muted" style={{ paddingLeft: 12 }}>{n}</div>)}
                  {f.errors?.map((e) => <pre key={e} className="mono small" style={{ whiteSpace: 'pre-wrap', color: 'var(--critical-text)' }}>{e.split('\n').slice(-4).join('\n')}</pre>)}
                </div>))}</td>
              <td style={{ textAlign: 'right' }}>
                <ConfirmButton className="ghost sm danger" onConfirm={() => api.put('/plugins/linked', { links: links.filter((x) => x.path !== l.path) }).then(refresh)}>Unlink</ConfirmButton>
              </td>
            </tr>))}</tbody></table>
        )}
      </div>

      <div>
        <h3 style={{ marginBottom: 8 }}>Your models <span className="muted small">({mine.length})</span></h3>
        {!mine.length ? <div className="card empty small">Upload a .py file (Add / upload) or link a project folder.</div> : (
          <div className="grid-3">{mine.map((m) => <ModelCard key={m.id} m={m} onArch={() => setArch(m)} />)}</div>
        )}
      </div>

      <div>
        <h3 style={{ marginBottom: 8 }}>Built-in generative methods</h3>
        <div className="small text-2" style={{ marginBottom: 8 }}>Your networks plug into these automatically: a module with <code>forward(x, t)</code> becomes a diffusion
          denoiser, one with <code>encode</code>/<code>decode</code> a VAE, <code>Generator</code> + <code>Discriminator</code> classes a GAN, a module taking
          <code> vocab_size</code> a language model, and a module with <code>losses()</code> + <code>sample()</code> a fully custom generative model.
          Multi-network models (GAN, VAE, latent diffusion, the encoder classifier) have interchangeable parts: each slot accepts a built-in network,
          one of yours (classes named <code>*Encoder</code>, <code>*Decoder</code>, <code>*Generator</code>, <code>*Discriminator</code> qualify), a part of a model trained here, or a pretrained download.</div>
        <div className="grid-3">{builtinGen.map((m) => <ModelCard key={m.id} m={m} onArch={() => setArch(m)} />)}</div>
      </div>

      <div className="card">
        <div className="card-head"><h3>Loss functions</h3><span className="small muted mono">user_plugins/losses/*.py · or a linked file of kind “losses”</span></div>
        <div className="card-pad small text-2" style={{ paddingTop: 0 }}>Any public function <code>fn(pred, target)</code> returning a scalar tensor becomes a choice in the
          loss fields of diffusion (prediction vs. target), VAE (reconstruction) and language models (logits vs. token ids).</div>
        <table><tbody>{(losses?.losses ?? []).map((l) => (
          <tr key={l.name}><td className="mono">{l.name}</td><td><span className="tag">{l.kind}</span></td><td className="small text-2">{l.doc}</td>
            <td style={{ textAlign: 'right' }}>{l.source && <button className="ghost sm" onClick={() => setLossSrc(l)}>Source</button>}</td></tr>))}</tbody></table>
      </div>

      {arch && <Modal title={`${arch.name} · architecture`} onClose={() => setArch(undefined)}><ArchitectureView modelId={arch.id} /></Modal>}
      {adding && <LinkModal existing={links ?? []} onClose={() => setAdding(false)} onSaved={() => { setAdding(false); refresh() }} />}
      {lossSrc && <Modal title={lossSrc.name} onClose={() => setLossSrc(undefined)}><div className="mono small muted">{lossSrc.file}</div><pre className="log mono small">{lossSrc.source}</pre></Modal>}
    </div>
  )
}

function ModelCard({ m, onArch }: { m: PluginMeta; onArch: () => void }) {
  return (
    <div className="card card-pad col" style={{ gap: 6 }}>
      <div className="row" style={{ justifyContent: 'space-between' }}><h3>{m.name}</h3><span className="tag">{ROLE(m)}</span></div>
      <div className="small text-2">{m.description}</div>
      <div className="mono small muted" style={{ wordBreak: 'break-all' }}>{m.source === 'builtin' ? m.id : m.source}</div>
      <div className="chips">{(m.modalities ?? []).map((x) => <span key={x} className="tag">{x}</span>)}
        <span className="tag">{m.params.length} params</span></div>
      {(m.framework === 'torch' || m.framework === 'generative')
        ? <button className="sm" style={{ alignSelf: 'flex-start' }} onClick={onArch}>Architecture & parameters</button>
        : <span className="small muted">no PyTorch network to inspect</span>}
    </div>
  )
}

function LinkModal({ existing, onClose, onSaved }: { existing: Linked[]; onClose: () => void; onSaved: () => void }) {
  const [path, setPath] = useState('')
  const [scan, setScan] = useState<Scan>()
  const [pick, setPick] = useState<Record<string, string>>({})
  const [err, setErr] = useState<unknown>()
  const [busy, setBusy] = useState(false)
  const doScan = async () => {
    setErr(undefined)
    try {
      const s = await api.post<Scan>('/plugins/linked/scan', { path })
      setScan(s)
      setPick(Object.fromEntries(s.files.filter((f) => f.suggest).map((f) => [f.file, f.suggest!])))
    } catch (e) { setErr(e) }
  }
  const save = async () => {
    if (!scan) return
    setBusy(true); setErr(undefined)
    const files = Object.entries(pick).filter(([, k]) => k).map(([file, kind]) => ({ file, kind }))
    try {
      await api.put('/plugins/linked', { links: [...existing.filter((l) => l.path !== scan.path), { path: scan.path, name: scan.path.split('/').pop(), files }] })
      onSaved()
    } catch (e) { setErr(e); setBusy(false) }
  }
  return (
    <Modal title="Link a project folder" onClose={onClose}>
      <div className="col" style={{ gap: 12 }}>
        <div className="row" style={{ flexWrap: 'nowrap' }}>
          <input value={path} onChange={(e) => setPath(e.target.value)} placeholder="/path/to/your/project" onKeyDown={(e) => e.key === 'Enter' && doScan()} />
          <button onClick={doScan} disabled={!path.trim()}>Scan</button>
        </div>
        <ErrorBox error={err} />
        {scan && (
          <>
            <div className="small text-2">Files are inspected without running them. Pick the ones to import: <b>model</b> files with your <code>nn.Module</code> classes,
              <b> losses</b> files with loss functions. Scripts with training loops or a <code>__main__</code> block don't need to be imported — only the model definitions.</div>
            <table>
              <thead><tr><th>File</th><th>Contains</th><th>Import as</th></tr></thead>
              <tbody>{scan.files.map((f) => (
                <tr key={f.file}>
                  <td className="mono small">{f.file}{f.has_main && <div className="muted">has __main__ (script)</div>}</td>
                  <td className="small">{f.error ? <span style={{ color: 'var(--critical-text)' }}>{f.error}</span> : <>
                    {(f.classes ?? []).filter((c) => c.torch).map((c) => <span key={c.name} className="tag" style={{ marginRight: 4 }}>{c.name}</span>)}
                    {(f.functions ?? []).length > 0 && <span className="muted"> {f.functions!.length} function{f.functions!.length > 1 ? 's' : ''}</span>}</>}</td>
                  <td><select value={pick[f.file] ?? ''} onChange={(e) => setPick((p) => ({ ...p, [f.file]: e.target.value }))} style={{ width: 120 }}>
                    <option value="">—</option><option value="model">model</option><option value="losses">losses</option>
                    <option value="dataset">dataset</option><option value="env">env</option></select></td>
                </tr>))}</tbody>
            </table>
            <button className="primary" style={{ alignSelf: 'flex-start' }} disabled={busy || !Object.values(pick).some(Boolean)} onClick={save}>
              {busy ? 'Linking & loading…' : 'Link folder'}</button>
          </>
        )}
      </div>
    </Modal>
  )
}
