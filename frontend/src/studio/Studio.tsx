import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { Link, NavLink, Route, Routes, useNavigate, useParams } from 'react-router-dom'
import { api, fmtBytes, fmtTime } from '../api'
import { ErrorBox, Modal, usePoll } from '../components/ui'
import { CompareView } from './Compare'
import { DatasetPage } from './DatasetPage'
import { SQLWorkspace } from './SQL'
import { fmtInt, studio, type Dataset } from './types'

/** Data Studio: catalog sidebar + dataset workspaces, SQL and subset comparison. */
export default function Studio() {
  const [list, err, reload] = usePoll(() => studio.list(), 3000)
  return (
    <div className="studio">
      <Catalog list={list ?? []} reload={reload} />
      <main className="studio-main">
        <ErrorBox error={err} />
        <Routes>
          <Route index element={<Home list={list} reload={reload} />} />
          <Route path="d/:id/*" element={<DatasetRoute list={list} reload={reload} />} />
          <Route path="sql" element={<SQLWorkspace list={list ?? []} reload={reload} />} />
          <Route path="compare" element={<CompareView list={list ?? []} />} />
        </Routes>
      </main>
    </div>
  )
}

function DatasetRoute({ list, reload }: { list?: Dataset[]; reload: () => void }) {
  const { id } = useParams()
  const ds = list?.find((d) => d.id === id)
  if (!list) return <div className="muted">Loading…</div>
  if (!ds) return <div className="card empty">Dataset not found. <Link to="/studio">Back to the catalog</Link></div>
  return <DatasetPage ds={ds} list={list} reload={reload} />
}

export function StatusDot({ ds }: { ds: Dataset }) {
  const cls = ds.building || ds.status === 'building' ? 'building' : ds.status === 'error' ? 'error' : ds.stale ? 'stale' : ds.status === 'ready' ? 'ready' : 'new'
  const label = { building: 'building', error: 'build failed', stale: 'stale — an upstream dataset changed', ready: 'ready', new: 'not built' }[cls]
  return <span className={`sdot ${cls}`} title={label} aria-label={label} />
}

const KIND_ICON: Record<string, string> = { source: '⬚', derived: '↳', query: '⌕' }

function Catalog({ list, reload }: { list: Dataset[]; reload: () => void }) {
  const [q, setQ] = useState('')
  const [adding, setAdding] = useState(false)
  const nav = useNavigate()
  // tree: sources and queries at the root, derived datasets under their parent
  const tree = useMemo(() => {
    const kids = new Map<string, Dataset[]>()
    list.forEach((d) => { if (d.parent_id) { if (!kids.has(d.parent_id)) kids.set(d.parent_id, []); kids.get(d.parent_id)!.push(d) } })
    const out: { d: Dataset; depth: number }[] = []
    const walk = (d: Dataset, depth: number) => { out.push({ d, depth }); (kids.get(d.id) ?? []).forEach((k) => walk(k, depth + 1)) }
    list.filter((d) => !d.parent_id || !list.some((p) => p.id === d.parent_id)).forEach((d) => walk(d, 0))
    return out
  }, [list])
  const shown = q ? tree.filter(({ d }) => `${d.name} ${d.slug}`.toLowerCase().includes(q.toLowerCase())) : tree
  return (
    <aside className="studio-side">
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <Link to="/studio" className="studio-title">Data Studio</Link>
        <button className="sm primary" onClick={() => setAdding(true)}>+ New</button>
      </div>
      <nav className="studio-tools">
        <NavLink to="/studio/sql">SQL editor</NavLink>
        <NavLink to="/studio/compare">Compare subsets</NavLink>
      </nav>
      <input className="sm-input" placeholder="Search datasets…" value={q} onChange={(e) => setQ(e.target.value)} />
      <div className="studio-tree">
        {shown.length === 0 && <div className="muted small" style={{ padding: 8 }}>{list.length ? 'No match.' : 'No datasets yet — import one with + New.'}</div>}
        {shown.map(({ d, depth }) => (
          <NavLink key={d.id} to={`/studio/d/${d.id}`} className="tree-item" style={{ paddingLeft: 8 + depth * 14 }} title={`${d.name} · ${d.slug}`}>
            <StatusDot ds={d} />
            <span className="muted" style={{ width: 12, flex: 'none' }}>{KIND_ICON[d.kind]}</span>
            <span className="tree-name">{d.name}</span>
            <span className="tree-meta num">{d.rows != null ? compact(d.rows) : ''}</span>
          </NavLink>
        ))}
      </div>
      {adding && <NewDataset onClose={() => setAdding(false)} onCreated={(d) => { setAdding(false); reload(); nav(`/studio/d/${d.id}`) }} />}
    </aside>
  )
}

export const compact = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(n >= 1e4 ? 0 : 1)}k` : String(n))

function Home({ list, reload }: { list?: Dataset[]; reload: () => void }) {
  const [adding, setAdding] = useState(false)
  const nav = useNavigate()
  const total = (list ?? []).reduce((a, d) => a + (d.bytes ?? 0), 0)
  return (
    <div className="col" style={{ gap: 16 }}>
      <div className="page-head">
        <div>
          <h1>Data Studio</h1>
          <div className="sub">Explore, clean and engineer tabular data; query it with SQL; carve out subsets and compare them; train models on any version.</div>
        </div>
        <div className="row"><Link className="btn" to="/studio/sql">Open SQL editor</Link><button className="primary" onClick={() => setAdding(true)}>+ New dataset</button></div>
      </div>
      <div className="grid-4">
        <Tile k="Datasets" v={fmtInt(list?.length ?? 0)} s={`${(list ?? []).filter((d) => d.kind === 'source').length} sources · ${(list ?? []).filter((d) => d.kind !== 'source').length} derived / queries`} />
        <Tile k="Rows (largest)" v={fmtInt(Math.max(0, ...(list ?? []).map((d) => d.rows ?? 0)))} s="DuckDB · Parquet, out-of-core" />
        <Tile k="Storage" v={fmtBytes(total)} s="data/studio (versioned, last 5 kept)" />
        <Tile k="Needs attention" v={String((list ?? []).filter((d) => d.stale || d.status === 'error').length)} s="stale or failed builds" />
      </div>
      <div className="card">
        <div className="card-head"><h3>Catalog</h3></div>
        {!list?.length ? <div className="empty">Start by importing a dataset: a dashboard dataset, a CSV / Parquet file (local or cloud), a live source, a database query, or your experiment results.</div> : (
          <div className="table-wrap"><table>
            <thead><tr><th>Dataset</th><th>Kind</th><th className="num">Rows</th><th className="num">Columns</th><th>Target</th><th className="num">Size</th><th>Built</th><th>Status</th></tr></thead>
            <tbody>{list.map((d) => (
              <tr key={d.id} className="clickable" onClick={() => nav(`/studio/d/${d.id}`)}>
                <td><b>{d.name}</b><div className="mono small muted">{d.slug}{d.parent_name ? ` ← ${d.parent_name}` : ''}</div></td>
                <td className="small">{d.kind}</td><td className="num">{fmtInt(d.rows)}</td><td className="num">{d.n_columns || '—'}</td>
                <td className="small mono">{d.target ?? <span className="muted">—</span>}</td><td className="num small">{d.bytes ? fmtBytes(d.bytes) : '—'}</td>
                <td className="small">{d.built_at ? fmtTime(d.built_at) : '—'} <span className="muted">v{d.version}</span></td>
                <td><StatusDot ds={d} /> <span className="small">{d.building ? 'building' : d.status === 'error' ? 'failed' : d.stale ? 'stale' : d.status}</span></td>
              </tr>))}</tbody>
          </table></div>
        )}
      </div>
      {adding && <NewDataset onClose={() => setAdding(false)} onCreated={(d) => { setAdding(false); reload(); nav(`/studio/d/${d.id}`) }} />}
    </div>
  )
}

export function Tile({ k, v, s }: { k: string; v: ReactNode; s?: ReactNode }) {
  return <div className="card tile"><div className="k">{k}</div><div className="v num">{v}</div>{s && <div className="s">{s}</div>}</div>
}

// ------------------------------------------------------------------ import
const SOURCES = [
  { key: 'dashboard_dataset', label: 'Dashboard dataset', help: 'Any tabular dataset the dashboard knows (built-in, uploaded CSV, Studio-independent).' },
  { key: 'file', label: 'File / cloud URL', help: 'CSV or Parquet: a file, a folder or a glob — local, or s3:// · gs:// · az:// (credentials: Settings → Cloud data).' },
  { key: 'upload', label: 'Upload', help: 'CSV, TSV, Parquet or JSON-lines from this computer.' },
  { key: 'live', label: 'Live source snapshot', help: 'Freeze the rows of a live source (optionally only the last N hours). Rebuild to refresh.' },
  { key: 'sql', label: 'Database query', help: 'Run a SQL query against SQLite, PostgreSQL, MySQL or a cloud warehouse and import the result.' },
  { key: 'dashboard_table', label: 'Experiment results', help: 'Your training sessions, runs and metrics as tables — analyze experiments like any other data.' },
] as const

function NewDataset({ onClose, onCreated }: { onClose: () => void; onCreated: (d: Dataset) => void }) {
  const [kind, setKind] = useState<(typeof SOURCES)[number]['key']>('dashboard_dataset')
  const [origins] = usePoll(() => studio.origins(), 0)
  const [name, setName] = useState('')
  const [plugin, setPlugin] = useState('')
  const [keepSplit, setKeepSplit] = useState(false)
  const [path, setPath] = useState('')
  const [format, setFormat] = useState('')
  const [file, setFile] = useState<File>()
  const [live, setLive] = useState('')
  const [hours, setHours] = useState('')
  const [connKind, setConnKind] = useState('sqlite')
  const [connLoc, setConnLoc] = useState('')
  const [sql, setSql] = useState('SELECT *\nFROM my_table\nLIMIT 100000')
  const [table, setTable] = useState('dashboard_runs')
  const [err, setErr] = useState<unknown>()
  const [busy, setBusy] = useState(false)
  useEffect(() => { if (origins && !plugin && origins.datasets.length) setPlugin(origins.datasets[0].id) }, [origins]) // eslint-disable-line react-hooks/exhaustive-deps
  const pdef = origins?.datasets.find((d: any) => d.id === plugin)
  const go = async () => {
    setBusy(true); setErr(undefined)
    try {
      let d: Dataset
      if (kind === 'upload') {
        if (!file) throw new Error('choose a file')
        const fd = new FormData(); fd.append('file', file); fd.append('name', name)
        d = await api.post<Dataset>('/studio/upload', fd)
      } else {
        const origin = kind === 'dashboard_dataset' ? { type: kind, plugin_id: plugin, keep_split: keepSplit }
          : kind === 'file' ? { type: kind, path, format: format || undefined }
          : kind === 'live' ? { type: kind, source_id: live, hours: hours ? Number(hours) : undefined }
          : kind === 'sql' ? { type: kind, connector: connKind === 'sql' ? { kind: 'sql', url: connLoc } : { kind: 'sqlite', path: connLoc }, query: sql }
          : { type: kind, table }
        const auto = kind === 'dashboard_dataset' ? pdef?.name : kind === 'file' ? path.split('/').filter(Boolean).pop()?.replace(/\.(csv|parquet|pq)$/i, '')
          : kind === 'live' ? origins?.live_sources.find((l: any) => l.id === live)?.name : kind === 'dashboard_table' ? table : 'query'
        d = await studio.create({ name: name || auto || 'dataset', kind: 'source', origin })
      }
      onCreated(d)
    } catch (e) { setErr(e); setBusy(false) }
  }
  return (
    <Modal title="New dataset" onClose={onClose}>
      <div className="col" style={{ gap: 12 }}>
        <div className="segmented wrap">{SOURCES.map((s) => <button key={s.key} className={kind === s.key ? 'on' : ''} onClick={() => setKind(s.key)}>{s.label}</button>)}</div>
        <div className="small text-2">{SOURCES.find((s) => s.key === kind)!.help}</div>
        {kind === 'dashboard_dataset' && origins && <>
          <label className="field"><span className="lbl">Dataset</span>
            <select value={plugin} onChange={(e) => setPlugin(e.target.value)}>{origins.datasets.map((d: any) => <option key={d.id} value={d.id}>{d.name} — {d.id}</option>)}</select>
            {pdef && <span className="help">{pdef.description}</span>}</label>
          <label className="check"><input type="checkbox" checked={keepSplit} onChange={(e) => setKeepSplit(e.target.checked)} />
            Keep its train / val / test assignment as a <span className="mono">__split</span> column</label>
        </>}
        {kind === 'file' && <div className="grid-2">
          <label className="field"><span className="lbl">Path or URL</span><input className="mono" value={path} onChange={(e) => setPath(e.target.value)} placeholder="/data/events/*.parquet · s3://bucket/prefix/" /></label>
          <label className="field"><span className="lbl">Format</span><select value={format} onChange={(e) => setFormat(e.target.value)}><option value="">auto</option><option>csv</option><option>parquet</option></select></label>
        </div>}
        {kind === 'upload' && <input type="file" accept=".csv,.tsv,.parquet,.pq,.json,.jsonl" onChange={(e) => setFile(e.target.files?.[0])} />}
        {kind === 'live' && origins && <div className="grid-2">
          <label className="field"><span className="lbl">Live source</span><select value={live} onChange={(e) => setLive(e.target.value)}><option value="">choose…</option>
            {origins.live_sources.map((l: any) => <option key={l.id} value={l.id}>{l.name}</option>)}</select></label>
          <label className="field"><span className="lbl">Last N hours (optional)</span><input value={hours} onChange={(e) => setHours(e.target.value)} placeholder="all rows" /></label>
        </div>}
        {kind === 'sql' && <>
          <div className="row" style={{ flexWrap: 'nowrap' }}>
            <select style={{ width: 200 }} value={connKind} onChange={(e) => setConnKind(e.target.value)}><option value="sqlite">SQLite file</option><option value="sql">SQL URL / warehouse</option></select>
            <input className="mono" value={connLoc} onChange={(e) => setConnLoc(e.target.value)} placeholder={connKind === 'sql' ? 'postgresql+psycopg://user:${DB_PASSWORD}@host/db' : '/path/to/db.sqlite'} />
          </div>
          <textarea className="mono" rows={5} value={sql} onChange={(e) => setSql(e.target.value)} />
          <span className="help small">The query runs on that database; the result is stored as a versioned Parquet table here. Passwords: use ${'{VAR}'} (data/cloud.env).</span>
        </>}
        {kind === 'dashboard_table' && origins && <label className="field"><span className="lbl">Table</span>
          <select value={table} onChange={(e) => setTable(e.target.value)}>{origins.dashboard_tables.map((t: any) => <option key={t.name} value={t.name}>{t.name} — {t.description}</option>)}</select>
          <span className="help">Also queryable directly in the SQL editor without importing.</span></label>}
        <label className="field"><span className="lbl">Name</span><input value={name} onChange={(e) => setName(e.target.value)} placeholder="auto" /></label>
        <ErrorBox error={err} />
        <button className="primary" style={{ alignSelf: 'flex-start' }} disabled={busy || (kind === 'file' && !path) || (kind === 'live' && !live) || (kind === 'sql' && !connLoc) || (kind === 'upload' && !file)} onClick={go}>
          {busy ? 'Importing…' : 'Import'}</button>
      </div>
    </Modal>
  )
}
