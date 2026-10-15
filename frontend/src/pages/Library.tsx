import { useState, type FormEvent } from 'react'
import { api, fmtValue, type PluginMeta } from '../api'
import { usePlugins } from '../App'
import { ArchitectureView } from '../components/Architecture'
import { CustomModels } from '../components/CustomModels'
import { RemovedDatasets, RemoveDatasetModal } from '../components/DatasetRemoval'
import { ConfirmButton, ErrorBox, Modal, usePoll } from '../components/ui'

type Files = {
  plugin_files: { kind: string; name: string; size: number; modified: number }[]
  csv_datasets: { slug: string; name: string; target: string; task: string; columns: string[] }[]
}

const TABS = ['Models', 'Custom models', 'Datasets', 'Environments', 'Add / upload'] as const

export default function Library() {
  const { plugins, reload } = usePlugins()
  const [tab, setTab] = useState<(typeof TABS)[number]>('Models')
  const [q, setQ] = useState('')
  if (!plugins) return <div className="muted">Loading…</div>
  const list = tab === 'Models' ? plugins.models : tab === 'Datasets' ? plugins.datasets : plugins.envs
  const filtered = list.filter((p) => !q || `${p.name} ${p.id} ${p.description}`.toLowerCase().includes(q.toLowerCase()))

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>Library</h1>
          <div className="sub">{plugins.models.length} models · {plugins.datasets.length} datasets · {plugins.envs.length} environments</div>
        </div>
        <button onClick={() => api.post('/plugins/reload').then(reload)}>Reload plugins</button>
      </div>
      {plugins.errors.length > 0 && (
        <details className="error-box" style={{ marginBottom: 14 }} open>
          <summary style={{ color: 'inherit' }}>{plugins.errors.length} plugin file{plugins.errors.length > 1 ? 's' : ''} failed to load</summary>
          {plugins.errors.map((e, i) => <pre key={i} className="mono small" style={{ whiteSpace: 'pre-wrap' }}><b>{e.file}</b>{'\n'}{e.error}</pre>)}
        </details>
      )}
      <div className="tabs">{TABS.map((t) => <button key={t} className={tab === t ? 'on' : ''} onClick={() => setTab(t)}>{t}</button>)}</div>
      {tab === 'Custom models' ? <CustomModels /> : tab === 'Add / upload' ? <Uploads /> : (
        <>
          <input style={{ maxWidth: 300, marginBottom: 12 }} placeholder="Filter…" value={q} onChange={(e) => setQ(e.target.value)} />
          <div className="grid-3">{filtered.map((p) => <PluginCard key={p.id} p={p} onChanged={reload} />)}</div>
          {tab === 'Datasets' && <div style={{ marginTop: 16 }}><RemovedDatasets version={plugins.datasets.length} onRestored={reload} /></div>}
          {tab === 'Environments' && <div style={{ marginTop: 16 }}><RemovedDatasets kind="env" version={plugins.envs.length} onRestored={reload} /></div>}
        </>
      )}
    </div>
  )
}

function PluginCard({ p, onChanged }: { p: PluginMeta; onChanged: () => void }) {
  const [open, setOpen] = useState(false)
  const [removing, setRemoving] = useState(false)
  const [arch, setArch] = useState(false)
  const hasNet = p.kind === 'model' && (p.framework === 'torch' || p.framework === 'generative')
  return (
    <div className="card card-pad col" style={{ gap: 6 }}>
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <h3>{p.name}</h3>
        {p.source !== 'builtin' && <span className="tag">{p.source === 'upload' ? 'uploaded' : p.source === 'live' ? 'live' : p.source.includes('/optional/') ? 'optional' : 'custom'}</span>}
      </div>
      <div className="mono small muted">{p.id}</div>
      <div className="small text-2">{p.description}</div>
      <div className="chips">
        {p.framework && <span className="tag">{p.framework}</span>}
        {(p.tasks ?? (p.task ? [p.task] : [])).map((t) => <span key={t} className="tag">{t}</span>)}
        {(p.modalities ?? (p.modality ? [p.modality] : [])).map((t) => <span key={t} className="tag">{t}</span>)}
        {p.uses_gpu && <span className="tag">GPU</span>}
      </div>
      <div className="row" style={{ gap: 4 }}>
        {p.params.length > 0 && <button className="ghost sm" onClick={() => setOpen(true)}>{p.params.length} parameters</button>}
        {hasNet && <button className="ghost sm" onClick={() => setArch(true)}>Architecture</button>}
        {(p.kind === 'dataset' || p.kind === 'env') && <button className="ghost sm danger" onClick={() => setRemoving(true)}>Remove</button>}
      </div>
      {removing && <RemoveDatasetModal id={p.id} kind={p.kind === 'env' ? 'env' : 'dataset'} onClose={() => setRemoving(false)} onDone={() => { setRemoving(false); onChanged() }} />}
      {arch && <Modal title={`${p.name} · architecture`} onClose={() => setArch(false)}><ArchitectureView modelId={p.id} /></Modal>}
      {open && (
        <Modal title={`${p.name} · parameters`} onClose={() => setOpen(false)}>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Name</th><th>Type</th><th>Default</th><th>Group</th><th>Description</th></tr></thead>
              <tbody>{p.params.map((x) => (
                <tr key={x.name}><td className="mono">{x.name}</td><td className="mono small">{x.type}{x.choices ? `: ${x.choices.join(' | ')}` : ''}</td>
                  <td className="mono">{fmtValue(x.default)}</td><td className="small">{x.group}</td><td className="small text-2">{x.help}</td></tr>
              ))}</tbody>
            </table>
          </div>
          <div className="muted small" style={{ marginTop: 8 }}>Source: <span className="mono">{p.source}</span></div>
        </Modal>
      )}
    </div>
  )
}

function parseHeader(text: string): string[] {
  const line = text.split(/\r?\n/)[0] ?? ''
  const out: string[] = []
  let cur = '', quoted = false
  for (const ch of line) {
    if (ch === '"') quoted = !quoted
    else if (ch === ',' && !quoted) { out.push(cur.trim()); cur = '' }
    else cur += ch
  }
  out.push(cur.trim())
  return out.filter(Boolean)
}

function Uploads() {
  const { reload } = usePlugins()
  const [files, , reloadFiles] = usePoll(() => api.get<Files>('/plugins/files'), 0)
  const [kind, setKind] = useState('model')
  const [pyFile, setPyFile] = useState<File>()
  const [pyResult, setPyResult] = useState<{ added: PluginMeta[]; errors: { error: string }[] }>()
  const [err, setErr] = useState<unknown>()
  const [csv, setCsv] = useState<File>()
  const [cols, setCols] = useState<string[]>([])
  const [csvName, setCsvName] = useState('')
  const [target, setTarget] = useState('')
  const [task, setTask] = useState('classification')
  const [csvMsg, setCsvMsg] = useState('')
  const [viewing, setViewing] = useState<{ title: string; src: string }>()
  const refresh = () => { reloadFiles(); reload() }

  const uploadPy = async (e: FormEvent) => {
    e.preventDefault()
    if (!pyFile) return
    setErr(undefined); setPyResult(undefined)
    const fd = new FormData()
    fd.append('kind', kind); fd.append('file', pyFile)
    try { setPyResult(await api.post('/plugins/upload', fd)); refresh() } catch (e) { setErr(e) }
  }
  const pickCsv = async (f?: File) => {
    setCsv(f); setCols([]); setTarget('')
    if (!f) return
    const head = await f.slice(0, 64 * 1024).text()
    const c = parseHeader(head)
    setCols(c); setTarget(c[c.length - 1] ?? '')
    if (!csvName) setCsvName(f.name.replace(/\.csv$/i, ''))
  }
  const uploadCsv = async (e: FormEvent) => {
    e.preventDefault()
    if (!csv) return
    setErr(undefined); setCsvMsg('')
    const fd = new FormData()
    fd.append('file', csv); fd.append('name', csvName); fd.append('target', target); fd.append('task', task)
    try { const r = await api.post<{ id: string }>('/datasets/upload_csv', fd); setCsvMsg(`Added dataset ${r.id}`); refresh() } catch (e) { setErr(e) }
  }

  return (
    <div className="col" style={{ gap: 16 }}>
      <ErrorBox error={err} />
      <div className="grid-2">
        <form className="card card-pad col" onSubmit={uploadPy}>
          <h2>Upload a Python plugin</h2>
          <div className="small text-2">
            A <b>model</b> file can contain a plain <code>torch.nn.Module</code> (constructor args with defaults become form fields;
            <code> input_dim</code>, <code>input_shape</code>, <code>in_channels</code>, <code>num_classes</code>, <code>output_dim</code> are injected) or a
            <code> ModelPlugin</code> subclass with its own training loop. An <b>environment</b> file can contain a plain <code>gymnasium.Env</code>.
            A <b>dataset</b> file contains a <code>DatasetPlugin</code> subclass. See <code>examples/plugins/</code> for templates.
          </div>
          <div className="row" style={{ flexWrap: 'nowrap' }}>
            <select value={kind} onChange={(e) => setKind(e.target.value)} style={{ width: 160 }}>
              <option value="model">Model</option><option value="dataset">Dataset</option><option value="env">Environment</option>
            </select>
            <input type="file" accept=".py" onChange={(e) => setPyFile(e.target.files?.[0])} />
          </div>
          <div className="small muted">Uploaded code runs on this machine with your user's permissions. Only upload code you trust.</div>
          <button className="primary" style={{ alignSelf: 'flex-start' }} disabled={!pyFile}>Upload</button>
          {pyResult && (
            <div className={pyResult.errors.length ? 'error-box' : 'notice'}>
              {pyResult.added.length > 0 && <>Added: {pyResult.added.map((a) => a.name).join(', ')}{'\n'}</>}
              {pyResult.errors.map((e) => e.error).join('\n')}
            </div>
          )}
        </form>

        <form className="card card-pad col" onSubmit={uploadCsv}>
          <h2>Upload a CSV dataset</h2>
          <div className="small text-2">Categorical feature columns are one-hot encoded and missing values are filled with the median.
            To use a file already on this machine without uploading it, pick “CSV file on disk” when starting a training session.</div>
          <input type="file" accept=".csv,text/csv" onChange={(e) => pickCsv(e.target.files?.[0])} />
          {cols.length > 0 && (
            <div className="grid-3" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))' }}>
              <label className="field"><span className="lbl">Name</span><input value={csvName} onChange={(e) => setCsvName(e.target.value)} /></label>
              <label className="field"><span className="lbl">Target column</span>
                <select value={target} onChange={(e) => setTarget(e.target.value)}>{cols.map((c) => <option key={c}>{c}</option>)}</select></label>
              <label className="field"><span className="lbl">Task</span>
                <select value={task} onChange={(e) => setTask(e.target.value)}><option>classification</option><option>regression</option></select></label>
            </div>
          )}
          <button className="primary" style={{ alignSelf: 'flex-start' }} disabled={!csv || !target || !csvName}>Add dataset</button>
          {csvMsg && <div className="notice">{csvMsg}</div>}
        </form>
      </div>

      <div className="grid-2">
        <div className="card">
          <div className="card-head"><h3>Plugin files</h3><span className="muted small mono">user_plugins/</span></div>
          {!files?.plugin_files.length ? <div className="empty small">No custom plugin files.</div> : (
            <table><tbody>{files.plugin_files.map((f) => (
              <tr key={f.kind + f.name}>
                <td><span className="tag">{f.kind}</span></td><td className="mono">{f.name}</td>
                <td style={{ textAlign: 'right' }}>
                  <button className="ghost sm" onClick={() => api.get<string>(`/plugins/file/${f.kind}/${f.name}`).then((src) => setViewing({ title: f.name, src }))}>View</button>
                  <ConfirmButton className="ghost sm danger" onConfirm={() => api.del(`/plugins/file/${f.kind}/${f.name}`).then(refresh)}>Delete</ConfirmButton>
                </td>
              </tr>))}</tbody></table>
          )}
        </div>
        <div className="card">
          <div className="card-head"><h3>Uploaded CSV datasets</h3></div>
          {!files?.csv_datasets.length ? <div className="empty small">No uploaded CSVs.</div> : (
            <table><tbody>{files.csv_datasets.map((d) => (
              <tr key={d.slug}>
                <td>{d.name}<div className="muted small">target <span className="mono">{d.target}</span> · {d.task} · {d.columns.length} columns</div></td>
                <td style={{ textAlign: 'right' }}><ConfirmButton className="ghost sm danger" onConfirm={() => api.del(`/datasets/upload/${d.slug}`).then(refresh)}>Delete</ConfirmButton></td>
              </tr>))}</tbody></table>
          )}
        </div>
      </div>
      {viewing && <Modal title={viewing.title} onClose={() => setViewing(undefined)}><pre className="log mono" style={{ maxHeight: '70vh' }}>{viewing.src}</pre></Modal>}
    </div>
  )
}
