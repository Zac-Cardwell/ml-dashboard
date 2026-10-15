import { createContext, useCallback, useContext, useEffect, useState, type FormEvent } from 'react'
import { BrowserRouter, NavLink, Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { api, setUnauthorizedHandler, type Plugins } from './api'
import { usePoll } from './components/ui'
import Compare from './pages/Compare'
import Data from './pages/Data'
import Exports from './pages/Exports'
import SourceDetail from './pages/SourceDetail'
import DeploymentDetail from './pages/DeploymentDetail'
import Deployments from './pages/Deployments'
import Registry from './pages/Registry'
import Library from './pages/Library'
import NewSession from './pages/NewSession'
import SessionDetail from './pages/SessionDetail'
import Sessions from './pages/Sessions'
import Settings from './pages/Settings'

type PluginCtx = { plugins: Plugins | undefined; reload: () => Promise<void> }
const PluginsContext = createContext<PluginCtx>({ plugins: undefined, reload: async () => {} })
export const usePlugins = () => useContext(PluginsContext)

export default function App() {
  const [authed, setAuthed] = useState<boolean | null>(null)
  useEffect(() => {
    setUnauthorizedHandler(() => setAuthed(false))
    api.get<{ authenticated: boolean }>('/auth/status').then((r) => setAuthed(r.authenticated)).catch(() => setAuthed(false))
  }, [])
  if (authed === null) return null
  if (!authed) return <Login onLogin={() => setAuthed(true)} />
  return <Shell onLogout={() => setAuthed(false)} />
}

function Shell({ onLogout }: { onLogout: () => void }) {
  const [plugins, setPlugins] = useState<Plugins>()
  const reload = useCallback(async () => setPlugins(await api.get<Plugins>('/plugins')), [])
  useEffect(() => { reload() }, [reload])
  return (
    <PluginsContext.Provider value={{ plugins, reload }}>
      <BrowserRouter>
        <div className="app">
          <header className="topbar">
            <NavLink to="/" className="brand">
              <img src="/favicon.svg" width={22} height={22} alt="" /> ML Dashboard
            </NavLink>
            <nav className="nav">
              <NavLink to="/" end>Sessions</NavLink>
              <NavLink to="/new" end className={({ isActive }) => (isActive && !location.search.includes('mode=evaluate') ? 'active' : '')}>New training</NavLink>
              <NavLink to="/new?mode=evaluate" className={() => (location.search.includes('mode=evaluate') ? 'active' : '')}>Evaluate</NavLink>
              <NavLink to="/compare">Compare</NavLink>
              <NavLink to="/data">Data</NavLink>
              <NavLink to="/registry">Registry</NavLink>
              <NavLink to="/deployments">Deployments</NavLink>
              <NavLink to="/library">Library</NavLink>
              <NavLink to="/settings">Settings</NavLink>
            </nav>
            <div className="spacer" />
            <SystemStatus />
            <button className="ghost sm" onClick={() => api.post('/auth/logout').then(onLogout)} title="Log out">⎋</button>
          </header>
          <main className="main">
            <Routes>
              <Route path="/" element={<Sessions />} />
              <Route path="/new" element={<KeyedNewSession />} />
              <Route path="/sessions/:id" element={<SessionDetail />} />
              <Route path="/compare" element={<Compare />} />
              <Route path="/data" element={<Data />} />
              <Route path="/data/sources/:id" element={<SourceDetail />} />
              <Route path="/registry" element={<Registry />} />
              <Route path="/deployments" element={<Deployments />} />
              <Route path="/deployments/:id" element={<DeploymentDetail />} />
              <Route path="/library" element={<Library />} />
              <Route path="/exports" element={<Exports />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="*" element={<Navigate to="/" />} />
            </Routes>
          </main>
        </div>
      </BrowserRouter>
    </PluginsContext.Provider>
  )
}

/** Remount the form when its query (train again / evaluate / fine-tune) changes. */
function KeyedNewSession() {
  const loc = useLocation()
  return <NewSession key={loc.search} />
}

type System = {
  gpus: { name: string; util: number; mem_used: number; mem_total: number; temp: number }[]
  cpu_percent: number
  queue: { running: { gpu: number; cpu: number }; limits: { gpu: number; cpu: number }; queued: number; scheduled: number
    power: { enabled: boolean; active: boolean; reason: string } }
}

function SystemStatus() {
  const [sys] = usePoll(() => api.get<System>('/system'), 4000)
  if (!sys) return null
  const g = sys.gpus[0]
  return (
    <div className="row small text-2 num sys" style={{ gap: 14 }}>
      {g && <span className="opt" title={`${g.name} · ${g.temp}°C`}>GPU {g.util.toFixed(0)}% · {(g.mem_used / 1024).toFixed(1)}/{(g.mem_total / 1024).toFixed(0)}G</span>}
      <span className="opt">CPU {sys.cpu_percent.toFixed(0)}%</span>
      <span title={`GPU jobs ${sys.queue.running.gpu}/${sys.queue.limits.gpu} · CPU jobs ${sys.queue.running.cpu}/${sys.queue.limits.cpu}`}>
        {sys.queue.running.gpu + sys.queue.running.cpu} running{sys.queue.queued > 0 && ` · ${sys.queue.queued} queued`}{sys.queue.scheduled > 0 && ` · ${sys.queue.scheduled} scheduled`}</span>
      {sys.queue.power.active && <span title={`Sleep is blocked: ${sys.queue.power.reason}. The screen can still turn off.`}>☾ awake</span>}
    </div>
  )
}

function Login({ onLogin }: { onLogin: () => void }) {
  const [pw, setPw] = useState('')
  const [err, setErr] = useState('')
  const submit = async (e: FormEvent) => {
    e.preventDefault()
    try {
      await api.post('/auth/login', { password: pw })
      onLogin()
    } catch (e) {
      setErr((e as Error).message)
    }
  }
  return (
    <div style={{ minHeight: '100vh', display: 'grid', placeItems: 'center', padding: 16 }}>
      <form className="card card-pad col" style={{ width: 'min(360px, 100%)', gap: 14 }} onSubmit={submit}>
        <div className="row"><img src="/favicon.svg" width={26} height={26} alt="" /><h1>ML Dashboard</h1></div>
        <label className="field">
          <span className="lbl">Password</span>
          <input type="password" autoFocus value={pw} onChange={(e) => setPw(e.target.value)} autoComplete="current-password" />
          <span className="help">Printed in the server console on first start; change it in Settings.</span>
        </label>
        {err && <div className="err-text small">{err}</div>}
        <button className="primary" type="submit">Sign in</button>
      </form>
    </div>
  )
}
