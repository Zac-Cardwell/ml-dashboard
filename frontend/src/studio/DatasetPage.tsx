import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { fmtBytes, fmtTime } from '../api'
import { ConfirmButton, ErrorBox, Modal } from '../components/ui'
import { ChartsTab } from './Charts'
import { DataGrid } from './DataGrid'
import { ExploreTab } from './Explore'
import { LineageTab } from './Lineage'
import { ProfileTab } from './Profile'
import { QueryEditorTab, RecipeTab } from './Recipe'
import { StatusDot, Tile } from './Studio'
import { fmtInt, studio, type Dataset } from './types'

const TABS = ['Profile', 'Data', 'Transform', 'Charts', 'Explore', 'Lineage'] as const

export function DatasetPage({ ds, list, reload }: { ds: Dataset; list: Dataset[]; reload: () => void }) {
  const [tab, setTab] = useState<(typeof TABS)[number]>('Profile')
  const [err, setErr] = useState<unknown>()
  const [deriving, setDeriving] = useState(false)
  const nav = useNavigate()
  const ready = ds.version > 0
  const act = async (fn: () => Promise<unknown>) => { setErr(undefined); try { await fn(); reload() } catch (e) { setErr(e) } }
  const transformLabel = ds.kind === 'derived' ? 'Recipe' : ds.kind === 'query' ? 'SQL' : 'Origin'
  return (
    <div className="col" style={{ gap: 14 }}>
      <div className="page-head" style={{ marginBottom: 0 }}>
        <div style={{ minWidth: 0 }}>
          <div className="small muted">
            <Link to="/studio">Data Studio</Link>
            {ds.parent_id && <> / <Link to={`/studio/d/${ds.parent_id}`}>{ds.parent_name ?? 'parent'}</Link></>} / {ds.kind}
          </div>
          <div className="row" style={{ gap: 10 }}>
            <h1 style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{ds.name}</h1>
            <StatusDot ds={ds} />
            <span className="tag mono">{ds.slug}</span>
            <span className="tag">v{ds.version}</span>
          </div>
          <div className="sub">{fmtInt(ds.rows)} rows · {ds.n_columns} columns{ds.bytes ? ` · ${fmtBytes(ds.bytes)}` : ''}
            {ds.built_at ? ` · built ${fmtTime(ds.built_at)}${ds.build_seconds != null ? ` in ${ds.build_seconds}s` : ''}` : ''}</div>
        </div>
        <div className="row" style={{ gap: 6 }}>
          <TargetPicker ds={ds} onChange={reload} />
          <button disabled={!ready} onClick={() => setDeriving(true)} title="New dataset whose recipe starts from this one">Derive</button>
          <button className="primary" disabled={!ready} onClick={() => nav(`/new?problem=studio.${ds.slug}`)} title="Train a model on this dataset">Train model</button>
          <div className="dropdown">
            <button disabled={!ready}>Download ▾</button>
            <div className="dropdown-menu">
              <a href={`/api/studio/datasets/${ds.id}/download?format=parquet`}>Parquet</a>
              <a href={`/api/studio/datasets/${ds.id}/download?format=csv`}>CSV</a>
            </div>
          </div>
          <button onClick={() => act(() => studio.build(ds.id))} disabled={ds.building}>{ds.building ? 'Building…' : 'Rebuild'}</button>
          <ConfirmButton className="danger" onConfirm={() => act(async () => { await studio.remove(ds.id); nav('/studio') })}>Delete</ConfirmButton>
        </div>
      </div>
      <ErrorBox error={err} />
      {ds.status === 'error' && <div className="error-box small">Last build failed: {ds.error}</div>}
      {ds.stale && !ds.building && <div className="notice small row" style={{ justifyContent: 'space-between' }}>
        An upstream dataset changed since this version was built. <button className="sm" onClick={() => act(() => studio.build(ds.id))}>Rebuild now</button></div>}
      {ds.warnings.length > 0 && (
        <div className="callout warning small">{ds.warnings.map((w, i) => <div key={i}><b>{w.kind === 'leakage' ? 'Leakage risk' : 'Note'}:</b> {w.message}</div>)}</div>
      )}
      {!ready ? (
        <div className="card empty">{ds.building || ds.status === 'building' ? 'Building the first version…' : ds.status === 'error' ? 'Fix the problem above, then Rebuild.' : 'Not built yet.'}</div>
      ) : (
        <>
          <div className="tabs" role="tablist">{TABS.map((t) => <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? 'on' : ''} onClick={() => setTab(t)}>
            {t === 'Transform' ? transformLabel : t}</button>)}</div>
          {tab === 'Profile' && <ProfileTab ds={ds} />}
          {tab === 'Data' && <DataGrid ds={ds} />}
          {tab === 'Transform' && (ds.kind === 'derived' ? <RecipeTab ds={ds} list={list} onSaved={reload} />
            : ds.kind === 'query' ? <QueryEditorTab ds={ds} onSaved={reload} /> : <OriginTab ds={ds} />)}
          {tab === 'Charts' && <ChartsTab ds={ds} list={list} />}
          {tab === 'Explore' && <ExploreTab ds={ds} onCreated={(id) => { reload(); nav(`/studio/d/${id}`) }} />}
          {tab === 'Lineage' && <LineageTab ds={ds} />}
        </>
      )}
      {deriving && <DeriveModal ds={ds} onClose={() => setDeriving(false)} onCreated={(id) => { setDeriving(false); reload(); nav(`/studio/d/${id}`) }} />}
    </div>
  )
}

function TargetPicker({ ds, onChange }: { ds: Dataset; onChange: () => void }) {
  const [open, setOpen] = useState(false)
  const [target, setTarget] = useState(ds.target ?? '')
  const [task, setTask] = useState(ds.task ?? 'classification')
  const [err, setErr] = useState<unknown>()
  const save = async () => {
    try { await studio.patch(ds.id, { target, task: target ? task : '' }); setOpen(false); onChange() } catch (e) { setErr(e) }
  }
  return (
    <>
      <button onClick={() => setOpen(true)} title="Which column models predict (none = unsupervised)">
        Target: <span className="mono">{ds.target ?? 'none'}</span></button>
      {open && <Modal title="Modeling target" onClose={() => setOpen(false)}>
        <div className="col" style={{ gap: 12 }}>
          <div className="small text-2">The target is what supervised models predict and what the profile analyzes features against.
            Without a target the dataset is offered to unsupervised models (clustering, anomaly detection, PCA). Derived datasets inherit it.</div>
          <div className="grid-2">
            <label className="field"><span className="lbl">Target column</span>
              <select value={target} onChange={(e) => setTarget(e.target.value)}><option value="">none (unsupervised)</option>
                {ds.schema.filter((c) => c.name !== '__split').map((c) => <option key={c.name}>{c.name}</option>)}</select></label>
            <label className="field"><span className="lbl">Task</span>
              <select value={task} disabled={!target} onChange={(e) => setTask(e.target.value)}><option>classification</option><option>regression</option></select></label>
          </div>
          <ErrorBox error={err} />
          <button className="primary" style={{ alignSelf: 'flex-start' }} onClick={save}>Save</button>
        </div>
      </Modal>}
    </>
  )
}

function DeriveModal({ ds, onClose, onCreated }: { ds: Dataset; onClose: () => void; onCreated: (id: string) => void }) {
  const [name, setName] = useState(`${ds.name} (clean)`)
  const [err, setErr] = useState<unknown>()
  const go = async () => {
    try { const d = await studio.create({ name, kind: 'derived', parent_id: ds.id, recipe: [] }); onCreated(d.id) } catch (e) { setErr(e) }
  }
  return (
    <Modal title={`Derive from ${ds.name}`} onClose={onClose}>
      <div className="col" style={{ gap: 12 }}>
        <div className="small text-2">A derived dataset is a recipe of steps applied to <b>{ds.name}</b>. It is rebuilt when you change the recipe or when its
          parent changes, and every version is kept for reproducibility. Use it for cleaning, feature engineering, or carving out a subset.</div>
        <label className="field"><span className="lbl">Name</span><input value={name} onChange={(e) => setName(e.target.value)} /></label>
        <ErrorBox error={err} />
        <button className="primary" style={{ alignSelf: 'flex-start' }} disabled={!name.trim()} onClick={go}>Create & open recipe</button>
      </div>
    </Modal>
  )
}

function OriginTab({ ds }: { ds: Dataset }) {
  const o = ds.origin
  const rows: [string, string][] = Object.entries(o).filter(([k]) => k !== 'type').map(([k, v]) => [k, typeof v === 'string' ? v : JSON.stringify(v)])
  return (
    <div className="grid-2">
      <div className="card card-pad col" style={{ gap: 8 }}>
        <h3>Origin</h3>
        <div className="small text-2">Where the rows come from. <b>Rebuild</b> re-imports them as a new version (for files, live sources and database
          queries this picks up new data); datasets derived from it then show as stale until they're rebuilt too.</div>
        <div className="kv small"><div>Type</div><div className="mono">{o.type}</div>{rows.map(([k, v]) => [<div key={k}>{k}</div>, <div key={k + 'v'} className="mono" style={{ wordBreak: 'break-all' }}>{v}</div>])}</div>
      </div>
      <div className="card card-pad col" style={{ gap: 8 }}>
        <h3>Schema</h3>
        <div className="table-wrap" style={{ maxHeight: 420 }}><table><tbody>{ds.schema.map((c) => (
          <tr key={c.name}><td className="mono small">{c.name}</td><td className="small muted">{c.type}</td></tr>))}</tbody></table></div>
      </div>
      <div className="row" style={{ gap: 18 }}>
        <Tile k="Rows" v={fmtInt(ds.rows)} />
        <Tile k="Versions" v={`v${ds.version}`} s="last 5 kept" />
      </div>
    </div>
  )
}
