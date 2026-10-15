import { useState } from 'react'
import { api, type SessionDetail } from '../api'
import { ErrorBox } from './ui'

type GenResult = { kind: string; seconds: number; image?: string; video?: string; clips?: string[]; specs?: string[]; spectrogram?: string
  texts?: string[]; prompt?: string }

/** Generation playground for a trained generative session: sample new images / audio / video / text on demand. */
export function GeneratePanel({ detail }: { detail: SessionDetail }) {
  const runs = detail.runs.filter((r) => r.storage?.has_model)
  const [rid, setRid] = useState(runs[0]?.id)
  const [variant, setVariant] = useState('best')
  const [n, setN] = useState(16)
  const [seed, setSeed] = useState(0)
  const [sampler, setSampler] = useState('')
  const [steps, setSteps] = useState(50)
  const [prompt, setPrompt] = useState('')
  const [length, setLength] = useState(200)
  const [temp, setTemp] = useState(0.8)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<unknown>()
  const [results, setResults] = useState<(GenResult & { key: string; rid: string; label: string })[]>([])
  const isText = detail.modality === 'text'
  const isDiffusion = detail.model_id.startsWith('gen.diffusion')
  const hasFinal = runs.find((r) => r.id === rid)?.storage?.has_final
  if (!runs.length) return <div className="card empty">Generation needs a saved model. Train with “Save model” on.</div>

  const go = async () => {
    if (!rid) return
    setBusy(true); setErr(undefined)
    try {
      const body = isText ? { n: Math.min(n, 8), seed, variant, prompt, length, temperature: temp }
        : { n, seed, variant, sampler: sampler || null, steps: sampler === 'ddim' ? steps : null }
      const r = await api.post<GenResult>(`/runs/${rid}/generate`, body)
      const seedRun = runs.find((x) => x.id === rid)?.seed
      const label = isText ? `seed ${seed} · T=${temp}` : `seed ${seed}${sampler ? ` · ${sampler}${sampler === 'ddim' ? ` ${steps} steps` : ''}` : ''}`
      setResults((xs) => [{ ...r, key: `${Date.now()}`, rid, label: `run seed ${seedRun} · ${variant} · ${label}` }, ...xs].slice(0, 12))
      setSeed((s) => s + 1)
    } catch (e) { setErr(e) } finally { setBusy(false) }
  }
  const file = (r: { rid: string }, p: string) => `/api/runs/${r.rid}/files/${p}?inline=true`

  return (
    <div className="col" style={{ gap: 14 }}>
      <div className="card card-pad col" style={{ gap: 10 }}>
        <div className="row" style={{ gap: 12, alignItems: 'flex-end' }}>
          {runs.length > 1 && <label className="field" style={{ width: 140 }}><span className="lbl">Model of</span>
            <select value={rid} onChange={(e) => setRid(e.target.value)}>{runs.map((r) => <option key={r.id} value={r.id}>seed {r.seed}</option>)}</select></label>}
          {hasFinal && <label className="field" style={{ width: 150 }}><span className="lbl">Weights</span>
            <select value={variant} onChange={(e) => setVariant(e.target.value)}><option value="best">best</option><option value="final">final</option></select></label>}
          <label className="field" style={{ width: 90 }}><span className="lbl">{isText ? 'Samples' : 'Count'}</span>
            <input type="number" min={1} max={isText ? 8 : 64} value={n} onChange={(e) => setN(Number(e.target.value))} /></label>
          <label className="field" style={{ width: 100 }}><span className="lbl">Seed</span>
            <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} /></label>
          {isDiffusion && <label className="field" style={{ width: 170 }}><span className="lbl">Sampler</span>
            <select value={sampler} onChange={(e) => setSampler(e.target.value)}><option value="">as trained</option>
              <option value="ddpm">DDPM (all steps)</option><option value="ddim">DDIM (fast)</option></select></label>}
          {isDiffusion && sampler === 'ddim' && <label className="field" style={{ width: 100 }}><span className="lbl">Steps</span>
            <input type="number" min={1} value={steps} onChange={(e) => setSteps(Number(e.target.value))} /></label>}
          {isText && <>
            <label className="field" style={{ width: 110 }}><span className="lbl">Length</span>
              <input type="number" min={1} max={2000} value={length} onChange={(e) => setLength(Number(e.target.value))} /></label>
            <label className="field" style={{ width: 110 }}><span className="lbl">Temperature</span>
              <input type="number" step={0.1} min={0} max={5} value={temp} onChange={(e) => setTemp(Number(e.target.value))} /></label>
          </>}
          <button className="primary" disabled={busy} onClick={go}>{busy ? 'Generating…' : 'Generate'}</button>
        </div>
        {isText && <label className="field"><span className="lbl">Prompt</span>
          <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="Start of the text (empty = unconditional)" style={{ minHeight: 60 }} /></label>}
        <div className="small muted">Runs on this machine's GPU if free. The seed increments after each generation; set it back to reproduce a result.</div>
      </div>
      <ErrorBox error={err} />
      {results.map((r) => (
        <div key={r.key} className="card card-pad col" style={{ gap: 8 }}>
          <div className="row small text-2" style={{ justifyContent: 'space-between' }}><span>{r.label}</span><span>{r.seconds}s</span></div>
          {r.image && <img src={file(r, r.image)} alt="generated" style={{ maxWidth: '100%', imageRendering: 'pixelated', alignSelf: 'flex-start' }} />}
          {r.video && <video src={file(r, r.video)} controls autoPlay loop muted playsInline style={{ maxWidth: '100%', alignSelf: 'flex-start' }} />}
          {r.spectrogram && <AudioSet spectrogram={file(r, r.spectrogram)} clips={(r.clips ?? []).map((c) => file(r, c))} specs={(r.specs ?? []).map((c) => file(r, c))} />}
          {r.texts && r.texts.map((t, i) => <pre key={i} className="log" style={{ whiteSpace: 'pre-wrap' }}><b>{r.prompt}</b>{t}</pre>)}
        </div>
      ))}
    </div>
  )
}

/** Generated audio: one tile per clip (spectrogram + player). Older snapshots stored one stacked spectrogram. */
export function AudioSet({ spectrogram, clips, specs }: { spectrogram: string; clips: string[]; specs?: string[] }) {
  if (!specs?.length) {
    return (
      <div className="row" style={{ gap: 14, alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <img src={spectrogram} alt="spectrograms" style={{ maxWidth: 240, width: '100%' }} />
        <div className="col" style={{ gap: 8 }}>{clips.map((c, i) => <audio key={c} src={c} controls preload="none" style={{ height: 32 }} aria-label={`clip ${i + 1}`} />)}</div>
      </div>
    )
  }
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: 12 }}>
      {clips.map((c, i) => (
        <div key={c} className="col" style={{ gap: 4 }}>
          <img src={specs[i]} alt={`spectrogram of clip ${i + 1}`} style={{ width: '100%', borderRadius: 4 }} />
          <audio src={c} controls preload="none" style={{ width: '100%', height: 32 }} aria-label={`clip ${i + 1}`} />
        </div>
      ))}
    </div>
  )
}
