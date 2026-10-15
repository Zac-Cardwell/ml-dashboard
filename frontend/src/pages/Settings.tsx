import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, fmtBytes, fmtTime } from '../api'
import { CloudSettings } from '../components/Cloud'
import { ConfirmButton, ErrorBox, usePoll } from '../components/ui'

type S = {
  max_gpu_jobs: number; max_cpu_jobs: number; api_token: string; gpu_available: boolean
  lan_only: boolean; allowed_networks: string[]; keep_awake: boolean
  power: { enabled: boolean; available: boolean; active: boolean; reason: string; since: number | null }
}
type Usage = { id: string; name: string; created_at: number; model_bytes: number; checkpoint_bytes: number; media_bytes: number; total_bytes: number }

export default function Settings() {
  const [s, setS] = useState<S>()
  const [gpu, setGpu] = useState('1')
  const [cpu, setCpu] = useState('2')
  const [nets, setNets] = useState('')
  const [pw, setPw] = useState('')
  const [msg, setMsg] = useState('')
  const [err, setErr] = useState<unknown>()
  const [showToken, setShowToken] = useState(false)
  const [usage, , reloadUsage] = usePoll(() => api.get<Usage[]>('/storage'), 0)
  const [sys] = usePoll(() => api.get<{ disk_free: number; disk_total: number }>('/system'), 0)

  const apply = (r: S) => { setS(r); setGpu(String(r.max_gpu_jobs)); setCpu(String(r.max_cpu_jobs)); setNets(r.allowed_networks.join(', ')) }
  useEffect(() => { api.get<S>('/settings').then(apply).catch(setErr) }, [])
  const flash = (m: string) => { setMsg(m); setTimeout(() => setMsg(''), 2500) }
  const save = async (body: Partial<S>, m: string) => {
    setErr(undefined)
    try { apply(await api.put<S>('/settings', body)); flash(m) } catch (e) { setErr(e) }
  }
  const savePw = async () => {
    try { await api.put('/settings/password', { new_password: pw }); setPw(''); flash('Password changed') } catch (e) { setErr(e) }
  }
  const origin = location.origin
  const total = (usage ?? []).reduce((a, u) => a + u.total_bytes, 0)
  const heavy = (usage ?? []).filter((u) => u.model_bytes + u.checkpoint_bytes > 0).sort((a, b) => b.total_bytes - a.total_bytes)

  return (
    <div className="col" style={{ gap: 16, maxWidth: 980 }}>
      <div className="page-head"><div><h1>Settings</h1></div>{msg && <div className="notice">{msg}</div>}</div>
      <ErrorBox error={err} />

      <div className="card card-pad col">
        <h2>Network access</h2>
        <div className="small text-2">
          The dashboard listens on your home network so phones and laptops can open <span className="mono">{origin}</span>.
          It is not reachable from the internet unless your router forwards a port to this machine. With <b>LAN-only</b> on, the server
          additionally refuses every request that doesn't come from a private address (192.168.x.x, 10.x.x.x, 172.16-31.x.x, this machine).
        </div>
        <label className="check"><input type="checkbox" checked={!!s?.lan_only} onChange={(e) => save({ lan_only: e.target.checked }, e.target.checked ? 'LAN-only on' : 'LAN-only off')} />
          <b>LAN-only</b> <span className="small text-2">(recommended)</span></label>
        <label className="field" style={{ maxWidth: 520 }}>
          <span className="lbl">Extra allowed networks</span>
          <div className="row" style={{ flexWrap: 'nowrap' }}>
            <input placeholder="e.g. 100.64.0.0/10 for Tailscale" value={nets} onChange={(e) => setNets(e.target.value)} />
            <button onClick={() => save({ allowed_networks: nets.split(',').map((x) => x.trim()).filter(Boolean) }, 'Networks saved')}>Save</button>
          </div>
          <span className="help">CIDR ranges, comma separated. Use this if you add a VPN like Tailscale to reach the dashboard away from home.</span>
        </label>
      </div>

      <CloudSettings />

      <div className="card card-pad col">
        <h2>Power</h2>
        <div className="small text-2">While training runs — or runs are queued or scheduled — the dashboard can block automatic <b>sleep</b>
          so jobs keep going overnight. It does <b>not</b> block the screen from dimming, locking or turning off.</div>
        <label className="check"><input type="checkbox" checked={!!s?.keep_awake} onChange={(e) => save({ keep_awake: e.target.checked } as Partial<S>, e.target.checked ? 'Keep-awake on' : 'Keep-awake off')} />
          <b>Keep the computer awake while training</b></label>
        <div className="small">{!s?.power.available ? <span className="err-text">systemd-inhibit not available on this system</span>
          : s?.power.active ? <>☾ Sleep is blocked now — {s.power.reason}</> : 'Not blocking sleep right now (nothing running or queued).'}</div>
        <div className="row"><button onClick={() => api.post<{ message: string }>('/system/display-off').then((r) => flash(r.message)).catch(setErr)}>Turn the display off now</button>
          <span className="small muted">Monitors wake on mouse/keyboard; training continues.</span></div>
      </div>

      <Notifications />

      <div className="card card-pad col">
        <h2>Job queue</h2>
        <div className="small text-2">How many runs (seeds) may train at the same time. Extra runs wait in the queue.
          GPU runs share {s?.gpu_available ? 'your GPU' : 'the GPU (none detected)'}; keep GPU jobs at 1 unless your models are small.</div>
        <div className="row">
          <label className="field" style={{ width: 160 }}><span className="lbl">Max GPU jobs</span><input value={gpu} onChange={(e) => setGpu(e.target.value)} inputMode="numeric" /></label>
          <label className="field" style={{ width: 160 }}><span className="lbl">Max CPU jobs</span><input value={cpu} onChange={(e) => setCpu(e.target.value)} inputMode="numeric" /></label>
        </div>
        <button style={{ alignSelf: 'flex-start' }} onClick={() => save({ max_gpu_jobs: Number(gpu), max_cpu_jobs: Number(cpu) }, 'Queue limits saved')}>Save limits</button>
      </div>

      <div className="card">
        <div className="card-head">
          <h2>Storage</h2>
          <span className="small text-2">runs use {fmtBytes(total)}{sys ? ` · ${fmtBytes(sys.disk_free)} free on disk` : ''}</span>
        </div>
        {!heavy.length ? <div className="empty small">No saved models or checkpoints.</div> : (
          <div className="table-wrap" style={{ maxHeight: 420 }}>
            <table>
              <thead><tr><th>Session</th><th className="num">Models</th><th className="num">Checkpoints</th><th className="num">Snapshots</th><th></th></tr></thead>
              <tbody>
                {heavy.map((u) => (
                  <tr key={u.id}>
                    <td><Link to={`/sessions/${u.id}`}>{u.name}</Link><div className="muted small">{fmtTime(u.created_at)}</div></td>
                    <td className="num">{fmtBytes(u.model_bytes)}</td>
                    <td className="num">{fmtBytes(u.checkpoint_bytes)}</td>
                    <td className="num">{fmtBytes(u.media_bytes)}</td>
                    <td style={{ textAlign: 'right' }}>
                      <ConfirmButton className="sm ghost danger" onConfirm={() => api.del(`/sessions/${u.id}/models`).then(reloadUsage).catch(setErr)}>Delete models</ConfirmButton>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="card card-pad col">
        <h2>Password</h2>
        <div className="row" style={{ flexWrap: 'nowrap', maxWidth: 480 }}>
          <input type="password" placeholder="New password (min 4 chars)" value={pw} onChange={(e) => setPw(e.target.value)} autoComplete="new-password" />
          <button disabled={pw.length < 4} onClick={savePw}>Change</button>
        </div>
        <div className="small muted">Changing it signs out every other device.</div>
      </div>

      <div className="card card-pad col">
        <h2>API & LLM access</h2>
        <div className="small text-2">Everything in this dashboard is available as JSON. Send the token as a bearer header.
          Interactive docs: <a href={`${origin}/docs`} target="_blank" rel="noreferrer">{origin}/docs</a> ·
          LLM tool definitions: <a href={`${origin}/api/llm/tools`} target="_blank" rel="noreferrer">/api/llm/tools</a></div>
        <div className="row" style={{ flexWrap: 'nowrap' }}>
          <input readOnly className="mono" value={showToken ? s?.api_token ?? '' : '•'.repeat(24)} />
          <button onClick={() => setShowToken((x) => !x)}>{showToken ? 'Hide' : 'Show'}</button>
        </div>
        <pre className="log mono small">{`# Connect Claude Code (or any MCP client) to this dashboard:
claude mcp add mltrain -e MLTRAIN_URL=${origin} -e MLTRAIN_TOKEN=$TOKEN -- \\
  env PYTHONPATH=<project>/backend <conda>/envs/mltrain/bin/python -m mltrain.mcp_server

# Compact digest of all sessions (good LLM context)
curl -H "Authorization: Bearer $TOKEN" ${origin}/api/llm/overview

# Everything about one session: config, curves, final metrics, snapshots, lineage
curl -H "Authorization: Bearer $TOKEN" ${origin}/api/sessions/<id>/export

# Start a session programmatically
curl -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \\
  -d '{"problem_kind":"dataset","problem_id":"sklearn.iris","model_id":"torch.mlp","model_params":{"lr":0.001},"seeds":[0,1,2]}' \\
  ${origin}/api/sessions`}</pre>
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ notifications
type Channel = { id?: string; type: string; name: string; enabled: boolean; events: string[]; url?: string; server?: string; topic?: string; token?: string
  host?: string; port?: number; username?: string; password?: string; to?: string; from?: string }
type NState = { channels: Channel[]; events: Record<string, string>; types: string[]; dashboard_url: string; dashboard_url_set: boolean
  history: { channel: string; title: string; ts: number; error: string | null }[] }

const HELP: Record<string, string> = {
  ntfy: 'Free phone push: install the ntfy app, subscribe to a hard-to-guess topic name, and enter it here. Messages go through ntfy.sh unless you set your own server.',
  discord: 'Server Settings → Integrations → Webhooks → New webhook → Copy URL.',
  slack: 'Create an Incoming Webhook in a Slack app and paste its URL.',
  email: 'SMTP account. For Gmail use smtp.gmail.com, port 587, and an App Password (not your normal password).',
  webhook: 'Any URL; receives a JSON POST {event, title, message, url, severity, data}.',
}

function Notifications() {
  const [st, , reload] = usePoll(() => api.get<NState>('/notifications'), 0)
  const [edit, setEdit] = useState<Channel>()
  const [err, setErr] = useState<unknown>()
  const [msg, setMsg] = useState('')
  const [url, setUrl] = useState<string>()
  if (!st) return null
  const blank = (type: string): Channel => ({ type, name: '', enabled: true, events: ['session_completed', 'session_failed', 'group_completed', 'alert_critical', 'automation'],
    ...(type === 'ntfy' ? { server: 'https://ntfy.sh', topic: '' } : type === 'email' ? { port: 587 } : {}) })
  const save = () => api.post('/notifications/channels', edit).then(() => { setEdit(undefined); reload() }).catch(setErr)
  const test = (c: Channel) => { setMsg(''); setErr(undefined); api.post(`/notifications/channels/${c.id}/test`).then(() => setMsg(`Test sent to ${c.name}`)).catch(setErr) }
  const f = (k: keyof Channel, label: string, type = 'text', ph = '') => (
    <label className="field"><span className="lbl">{label}</span>
      <input type={type} placeholder={ph} value={(edit?.[k] as string | number | undefined) ?? ''} onChange={(e) => setEdit({ ...edit!, [k]: type === 'number' ? Number(e.target.value) : e.target.value })} /></label>
  )
  return (
    <div className="card card-pad col">
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <h2>Notifications</h2>
        <div className="row">{st.types.map((t) => <button key={t} className="sm" onClick={() => setEdit(blank(t))}>+ {t}</button>)}</div>
      </div>
      <ErrorBox error={err} />
      {msg && <div className="notice small">{msg}</div>}
      {st.channels.length === 0 && !edit && <div className="small text-2">No channels yet. Add one to get pinged when training finishes, a search completes or a deployment needs attention.</div>}
      {st.channels.map((c) => (
        <div key={c.id} className="row" style={{ justifyContent: 'space-between', borderTop: '1px solid var(--border)', paddingTop: 8 }}>
          <div><b>{c.name}</b> <span className="tag">{c.type}</span>{!c.enabled && <span className="tag">disabled</span>}
            <div className="small text-2">{c.events.map((e) => st.events[e] ?? e).join(' · ')}</div></div>
          <div className="row" style={{ gap: 4 }}>
            <button className="sm" onClick={() => test(c)}>Send test</button>
            <button className="sm ghost" onClick={() => setEdit({ ...c })}>Edit</button>
            <ConfirmButton className="sm ghost danger" onConfirm={() => api.del(`/notifications/channels/${c.id}`).then(reload)}>Delete</ConfirmButton>
          </div>
        </div>
      ))}
      {edit && (
        <div className="card card-pad col" style={{ gap: 10, background: 'var(--surface-2)' }}>
          <h3>{edit.id ? 'Edit' : 'New'} {edit.type} channel</h3>
          <div className="small text-2">{HELP[edit.type]}</div>
          <div className="grid-3">
            {f('name', 'Name', 'text', `My ${edit.type}`)}
            {['discord', 'slack', 'webhook'].includes(edit.type) && f('url', 'Webhook URL', 'url', 'https://…')}
            {edit.type === 'ntfy' && <>{f('topic', 'Topic', 'text', 'ml-dash-7f3k2')}{f('server', 'Server')}{f('token', 'Access token (optional)', 'password')}</>}
            {edit.type === 'email' && <>{f('host', 'SMTP host', 'text', 'smtp.gmail.com')}{f('port', 'Port', 'number')}{f('username', 'Username')}
              {f('password', 'Password / app password', 'password')}{f('to', 'Send to', 'email')}{f('from', 'From (optional)', 'email')}</>}
          </div>
          <div className="col" style={{ gap: 4 }}>
            <span className="small text-2">Notify me about</span>
            {Object.entries(st.events).map(([k, l]) => (
              <label key={k} className="check small"><input type="checkbox" checked={edit.events.includes(k)}
                onChange={(e) => setEdit({ ...edit, events: e.target.checked ? [...edit.events, k] : edit.events.filter((x) => x !== k) })} />{l}</label>))}
            <label className="check small"><input type="checkbox" checked={edit.enabled} onChange={(e) => setEdit({ ...edit, enabled: e.target.checked })} />Enabled</label>
          </div>
          <div className="row"><button className="primary" onClick={save}>Save channel</button><button onClick={() => setEdit(undefined)}>Cancel</button></div>
        </div>
      )}
      <label className="field" style={{ maxWidth: 520 }}>
        <span className="lbl">Dashboard address used in links</span>
        <div className="row" style={{ flexWrap: 'nowrap' }}>
          <input value={url ?? st.dashboard_url} onChange={(e) => setUrl(e.target.value)} />
          <button onClick={() => api.put('/notifications/dashboard-url', { dashboard_url: url ?? '' }).then(() => { setUrl(undefined); reload() })}>Save</button>
        </div>
        <span className="help">{st.dashboard_url_set ? 'Custom address' : 'Detected automatically'} — links only open on your home network.</span>
      </label>
      {st.history.length > 0 && (
        <details><summary>Recent deliveries</summary>
          <table><tbody>{st.history.map((h, i) => <tr key={i}><td className="small">{new Date(h.ts * 1000).toLocaleString()}</td><td className="small">{h.channel}</td>
            <td className="small">{h.title}</td><td className="small">{h.error ? <span className="err-text">{h.error}</span> : '✓'}</td></tr>)}</tbody></table>
        </details>
      )}
    </div>
  )
}
