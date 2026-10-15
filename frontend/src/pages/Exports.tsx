import { useEffect, useState, type ReactNode } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { api, fmtBytes, fmtTime } from '../api'
import type { GitIdentity } from '../components/ExportModal'
import { ConfirmButton, ErrorBox, Modal, usePoll } from '../components/ui'

type Export = { name: string; title: string; created_at: number; experiments: string[]; sessions: string[]; warnings: string[]
  include_models: string; files: number; bytes: number; git?: { ok: boolean; commit?: string; error?: string }; remote: string | null }

export default function Exports() {
  const [sp, setSp] = useSearchParams()
  const [data, err, reload] = usePoll(() => api.get<{ exports: Export[]; identity: GitIdentity }>('/exports'), 0)
  const [readme, setReadme] = useState<string>()
  const [pushing, setPushing] = useState<Export>()
  const open = sp.get('open')
  useEffect(() => {
    setReadme(undefined)
    if (open) api.get<string>(`/exports/${open}/readme`).then(setReadme).catch(() => setReadme(''))
  }, [open])
  const cur = data?.exports.find((e) => e.name === open)

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>Portfolio exports</h1>
          <div className="sub">Standalone repositories built from sessions — anyone can clone one and re-run the experiments without this dashboard.
            Select sessions on the <Link to="/">Sessions</Link> page (or use Export on a session) to create one.</div>
        </div>
      </div>
      <ErrorBox error={err} />
      {data && !data.exports.length && <div className="card empty">No exports yet.</div>}
      <div className="col" style={{ gap: 12 }}>
        {data?.exports.map((e) => (
          <div key={e.name} className="card card-pad col" style={{ gap: 8, outline: e.name === open ? '2px solid var(--accent)' : undefined }}>
            <div className="row" style={{ justifyContent: 'space-between' }}>
              <div>
                <h3>{e.title}</h3>
                <div className="small text-2">{e.experiments.length} experiment{e.experiments.length > 1 ? 's' : ''} · {e.files} files · {fmtBytes(e.bytes)}
                  {' '}· weights: {e.include_models} · built {fmtTime(e.created_at)}
                  {e.git?.ok && <> · commit <span className="mono">{e.git.commit}</span></>}
                  {e.remote && <> · pushed to <a href={e.remote.replace(/^git@github\.com:/, 'https://github.com/').replace(/\.git$/, '')} target="_blank" rel="noreferrer">{e.remote}</a></>}</div>
                <div className="mono small muted">data/exports/{e.name}/</div>
              </div>
              <div className="row" style={{ gap: 6 }}>
                <button className="sm" onClick={() => setSp(e.name === open ? {} : { open: e.name })}>{e.name === open ? 'Hide README' : 'Preview README'}</button>
                <a className="btn sm" href={`/api/exports/${e.name}/zip`}>Download .zip</a>
                <button className="sm primary" disabled={!e.git?.ok} onClick={() => setPushing(e)}>Publish to GitHub…</button>
                <ConfirmButton className="sm ghost danger" onConfirm={() => api.del(`/exports/${e.name}`).then(() => { setSp({}); reload() })}>Delete</ConfirmButton>
              </div>
            </div>
            {e.warnings.length > 0 && <div className="notice small" style={{ whiteSpace: 'pre-wrap' }}>{e.warnings.map((w) => `• ${w}`).join('\n')}</div>}
            {e.git && !e.git.ok && <div className="error-box small">git commit failed: {e.git.error}</div>}
          </div>
        ))}
      </div>
      {cur && readme !== undefined && (
        <div className="card card-pad" style={{ marginTop: 16 }}>
          <div className="row small muted" style={{ justifyContent: 'space-between', marginBottom: 8 }}>
            <span>README.md preview (approximately how GitHub shows it)</span><span className="mono">{cur.name}/README.md</span>
          </div>
          <div className="markdown"><Markdown src={readme} base={`/api/exports/${cur.name}/file/`} /></div>
        </div>
      )}
      {pushing && data && <PushModal e={pushing} identity={data.identity} onClose={() => { setPushing(undefined); reload() }} />}
    </div>
  )
}

function PushModal({ e, identity, onClose }: { e: Export; identity: GitIdentity; onClose: () => void }) {
  const [how, setHow] = useState<'url' | 'gh'>(identity.gh ? 'gh' : 'url')
  const [remote, setRemote] = useState(e.remote ?? '')
  const [repoName, setRepoName] = useState(e.name)
  const [priv, setPriv] = useState(false)
  const [ack, setAck] = useState(false)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<unknown>()
  const [done, setDone] = useState<{ remote: string; output: string }>()
  const push = async () => {
    setBusy(true); setErr(undefined)
    try {
      setDone(await api.post(`/exports/${e.name}/push`, how === 'gh'
        ? { remote: repoName, create_with_gh: true, private: priv, confirm: true }
        : { remote, confirm: true }))
    } catch (x) { setErr(x) } finally { setBusy(false) }
  }
  const web = (done?.remote ?? '').replace(/^git@github\.com:/, 'https://github.com/').replace(/\.git$/, '')
  return (
    <Modal title={`Publish “${e.title}” to GitHub`} onClose={onClose}>
      {done ? (
        <div className="col" style={{ gap: 10 }}>
          <div className="notice">Pushed to <a href={web} target="_blank" rel="noreferrer">{web}</a>.</div>
          <pre className="log mono small">{done.output}</pre>
        </div>
      ) : (
        <div className="col" style={{ gap: 12 }}>
          <div className="segmented" style={{ alignSelf: 'flex-start' }}>
            <button className={how === 'gh' ? 'on' : ''} onClick={() => setHow('gh')} disabled={!identity.gh}>Create repo with GitHub CLI{identity.gh ? '' : ' (not installed)'}</button>
            <button className={how === 'url' ? 'on' : ''} onClick={() => setHow('url')}>Push to an existing empty repo</button>
          </div>
          {how === 'gh' ? (
            <div className="grid-2">
              <label className="field"><span className="lbl">Repository name</span><input value={repoName} onChange={(x) => setRepoName(x.target.value)} /></label>
              <label className="field"><span className="lbl">Visibility</span>
                <select value={priv ? 'private' : 'public'} onChange={(x) => setPriv(x.target.value === 'private')}><option value="public">public</option><option value="private">private</option></select></label>
            </div>
          ) : (
            <>
              <ol className="small text-2" style={{ margin: 0, paddingLeft: 18 }}>
                <li>On github.com, click <b>New repository</b>, give it a name, and leave it empty (no README, license or .gitignore).</li>
                <li>Paste its URL below. SSH URLs (<span className="mono">git@github.com:you/repo.git</span>) use your SSH key; HTTPS needs a git credential helper or <span className="mono">gh auth login</span>.</li>
              </ol>
              <label className="field"><span className="lbl">Repository URL</span>
                <input value={remote} onChange={(x) => setRemote(x.target.value)} placeholder="git@github.com:you/my-ml-experiments.git" /></label>
            </>
          )}
          {!identity.gh && how === 'url' && <div className="small muted">Tip: install the GitHub CLI (<span className="mono">sudo dnf install gh</span>, then <span className="mono">gh auth login</span>) to create repositories from here.</div>}
          <label className="row small" style={{ flexWrap: 'nowrap', alignItems: 'flex-start', gap: 8 }}>
            <input type="checkbox" checked={ack} onChange={(x) => setAck(x.target.checked)} style={{ width: 'auto', marginTop: 2 }} />
            <span>I reviewed the README and files. Pushing uploads everything in this export ({e.files} files, {fmtBytes(e.bytes)}, incl. data snapshots
              {e.include_models !== 'none' ? ' and model weights' : ''}) to GitHub{how === 'gh' && !priv ? ', where a public repository is visible to everyone' : ''}.</span>
          </label>
          <ErrorBox error={err} />
          <button className="primary" style={{ alignSelf: 'flex-start' }} disabled={!ack || busy || (how === 'url' ? !remote.trim() : !repoName.trim())} onClick={push}>
            {busy ? 'Pushing…' : 'Push to GitHub'}</button>
        </div>
      )}
    </Modal>
  )
}

// ---------------------------------------------------------------- tiny markdown renderer (what our README generator emits)
function inline(s: string, base: string, k = 0): ReactNode[] {
  const out: ReactNode[] = []
  const re = /(!\[([^\]]*)\]\(([^)]+)\))|(\[([^\]]+)\]\(([^)]+)\))|(`([^`]+)`)|(\*\*([^*]+)\*\*)/g
  let last = 0, m: RegExpExecArray | null
  while ((m = re.exec(s))) {
    if (m.index > last) out.push(s.slice(last, m.index))
    const key = `${k}-${m.index}`
    if (m[1]) out.push(<img key={key} src={url(m[3], base)} alt={m[2]} style={{ maxWidth: '100%' }} />)
    else if (m[4]) out.push(<a key={key} href={m[6].startsWith('#') ? undefined : url(m[6], base)} target="_blank" rel="noreferrer">{inline(m[5], base, k + 1)}</a>)
    else if (m[7]) out.push(<code key={key}>{m[8]}</code>)
    else if (m[9]) out.push(<b key={key}>{inline(m[10], base, k + 1)}</b>)
    last = re.lastIndex
  }
  if (last < s.length) out.push(s.slice(last))
  return out
}

const url = (u: string, base: string) => (/^(https?:|#|\/)/.test(u) ? u : base + u)

function Markdown({ src, base }: { src: string; base: string }) {
  const lines = src.split('\n')
  const blocks: ReactNode[] = []
  let i = 0
  while (i < lines.length) {
    const l = lines[i]
    if (l.startsWith('```')) {
      const body: string[] = []
      for (i++; i < lines.length && !lines[i].startsWith('```'); i++) body.push(lines[i])
      blocks.push(<pre key={i} className="log mono small">{body.join('\n')}</pre>); i++; continue
    }
    const h = /^(#{1,4}) (.*)/.exec(l)
    if (h) { const T = `h${Math.min(h[1].length + 1, 4)}` as 'h2'; blocks.push(<T key={i}>{inline(h[2], base)}</T>); i++; continue }
    if (l.startsWith('|')) {
      const rows: string[][] = []
      for (; i < lines.length && lines[i].startsWith('|'); i++) rows.push(lines[i].split('|').slice(1, -1).map((c) => c.trim()))
      const [head, , ...body] = rows
      blocks.push(<div key={i} className="table-wrap"><table><thead><tr>{head.map((c, j) => <th key={j}>{inline(c, base)}</th>)}</tr></thead>
        <tbody>{body.map((r, j) => <tr key={j}>{r.map((c, q) => <td key={q}>{inline(c, base)}</td>)}</tr>)}</tbody></table></div>)
      continue
    }
    if (/^- /.test(l)) {
      const items: string[] = []
      for (; i < lines.length && /^- /.test(lines[i]); i++) items.push(lines[i].slice(2))
      blocks.push(<ul key={i}>{items.map((t, j) => <li key={j}>{inline(t, base)}</li>)}</ul>); continue
    }
    const img = /^<img src="([^"]+)" alt="([^"]*)"(?: width="(\d+)")?>/.exec(l)
    if (img) { blocks.push(<img key={i} src={url(img[1], base)} alt={img[2]} style={{ width: Number(img[3]) || undefined, maxWidth: '100%' }} />); i++; continue }
    if (/^<\/?details>|^<summary>/.test(l)) {
      const sm = /<summary>(.*)<\/summary>/.exec(l)
      if (sm) {
        const inner: string[] = []
        for (i++; i < lines.length && !lines[i].startsWith('</details>'); i++) inner.push(lines[i])
        blocks.push(<details key={i}><summary>{sm[1]}</summary><Markdown src={inner.join('\n')} base={base} /></details>)
      }
      i++; continue
    }
    if (!l.trim()) { i++; continue }
    const para: string[] = []
    for (; i < lines.length && lines[i].trim() && !/^(#|\||- |```|<)/.test(lines[i]); i++) para.push(lines[i])
    if (!para.length) { i++; continue }
    blocks.push(<p key={i}>{inline(para.join(' '), base)}</p>)
  }
  return <>{blocks}</>
}
