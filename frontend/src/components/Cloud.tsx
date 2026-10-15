import { useState } from 'react'
import { api } from '../api'
import { ErrorBox, usePoll } from './ui'

type Prov = { library: boolean; credentials: string; region?: string | null; settings: Record<string, unknown> }
type Status = { env_file: string; env_file_vars: string[]; providers: Record<'aws' | 'gcp' | 'azure', Prov>
  warehouses: Record<string, { label: string; template: string; help: string; provider: string; driver: boolean }> }

const FIELDS: Record<'aws' | 'gcp' | 'azure', { key: string; label: string; placeholder: string; help?: string }[]> = {
  aws: [{ key: 'profile', label: 'Profile', placeholder: 'default', help: 'From ~/.aws/config (aws configure / aws sso login)' },
    { key: 'region', label: 'Region', placeholder: 'us-east-1' },
    { key: 'endpoint_url', label: 'Endpoint URL', placeholder: 'only for S3-compatible storage (MinIO, R2…)' }],
  gcp: [{ key: 'project', label: 'Project', placeholder: 'my-project' },
    { key: 'credentials_file', label: 'Service-account key file', placeholder: '/path/to/key.json',
      help: 'Empty = application default credentials (gcloud auth application-default login)' }],
  azure: [{ key: 'account_name', label: 'Storage account', placeholder: 'mystorageacct', help: 'With no key: az login / managed identity' },
    { key: 'connection_string_env', label: 'Connection string env var', placeholder: 'AZURE_STORAGE_CONNECTION_STRING' },
    { key: 'account_key_env', label: 'Account key env var', placeholder: 'AZURE_STORAGE_KEY' }],
}
const NAMES = { aws: 'AWS (S3)', gcp: 'Google Cloud (GCS)', azure: 'Azure (Blob / ADLS Gen2)' }

/** Settings → Cloud data: credential status per provider, non-secret settings, connection test. */
export function CloudSettings() {
  const [st, err, reload] = usePoll(() => api.get<Status>('/cloud/status'), 0)
  const [vals, setVals] = useState<Record<string, Record<string, string>>>()
  const [msg, setMsg] = useState('')
  const [e2, setE2] = useState<unknown>()
  const [url, setUrl] = useState('')
  const [test, setTest] = useState<{ ok: boolean; files?: string[]; count?: number; listing?: string[]; message?: string }>()
  const v = vals ?? (st ? Object.fromEntries((['aws', 'gcp', 'azure'] as const).map((p) => [p, Object.fromEntries(Object.entries(st.providers[p].settings).map(([k, x]) => [k, String(x)]))])) : {})
  const set = (p: string, k: string, x: string) => setVals({ ...v, [p]: { ...(v[p] ?? {}), [k]: x } })
  const save = async () => {
    setE2(undefined)
    try { await api.put('/cloud/settings', v); setMsg('Saved'); setVals(undefined); reload(); setTimeout(() => setMsg(''), 2000) } catch (x) { setE2(x) }
  }
  const run = async () => { setTest(undefined); setE2(undefined); try { setTest(await api.post('/cloud/test', { path: url })) } catch (x) { setE2(x) } }
  return (
    <div className="card card-pad col">
      <div className="row" style={{ justifyContent: 'space-between' }}><h2>Cloud data</h2>{msg && <span className="notice small">{msg}</span>}</div>
      <div className="small text-2">
        Live sources and CSV/Parquet datasets can read <span className="mono">s3://</span>, <span className="mono">gs://</span> and <span className="mono">az://</span> URLs,
        and SQL sources can be Redshift, Athena, BigQuery or Azure SQL. The dashboard <b>never stores secrets</b>: it uses each cloud's normal login
        (aws configure, gcloud auth application-default login, az login, key files, roles). Secrets such as passwords can go in{' '}
        <span className="mono">{st?.env_file ?? 'data/cloud.env'}</span> (KEY=value lines; git-ignored) and be used in SQL URLs as <span className="mono">{'${KEY}'}</span>.
        {st && st.env_file_vars.length > 0 && <> Variables loaded from it: {st.env_file_vars.map((n) => <span key={n} className="tag mono" style={{ marginLeft: 4 }}>{n}</span>)}</>}
      </div>
      <ErrorBox error={err ?? e2} />
      {st && (['aws', 'gcp', 'azure'] as const).map((p) => {
        const pr = st.providers[p]
        const okCred = !/not found|NOT SET|missing/.test(pr.credentials)
        return (
          <div key={p} style={{ borderTop: '1px solid var(--border)', paddingTop: 10 }}>
            <div className="row" style={{ gap: 8 }}>
              <b>{NAMES[p]}</b>
              <span className={`badge ${pr.library ? 'completed' : 'failed'}`}><span className="dot" />{pr.library ? 'library installed' : 'library missing'}</span>
              <span className={`badge ${okCred ? 'completed' : 'queued'}`}><span className="dot" />{pr.credentials}</span>
              {pr.region && <span className="small muted">region {pr.region}</span>}
            </div>
            <div className="grid-3" style={{ marginTop: 6 }}>
              {FIELDS[p].map((f) => (
                <label key={f.key} className="field"><span className="lbl">{f.label}</span>
                  <input value={v[p]?.[f.key] ?? ''} placeholder={f.placeholder} onChange={(e) => set(p, f.key, e.target.value)} />
                  {f.help && <span className="help">{f.help}</span>}</label>))}
            </div>
          </div>
        )
      })}
      <button style={{ alignSelf: 'flex-start' }} onClick={save} disabled={!vals}>Save cloud settings</button>
      <div className="row" style={{ flexWrap: 'nowrap', borderTop: '1px solid var(--border)', paddingTop: 10 }}>
        <input className="mono" placeholder="Test access: s3://bucket/prefix/ · gs://bucket/data/*.parquet · az://container/folder" value={url} onChange={(e) => setUrl(e.target.value)} />
        <button onClick={run} disabled={!url.trim()}>Test</button>
      </div>
      {test && <div className={test.ok ? 'notice small' : 'error-box small'}>
        {test.files?.length ? <>Found {test.count} data file{test.count === 1 ? '' : 's'}: <span className="mono">{test.files.slice(0, 5).join(', ')}{(test.count ?? 0) > 5 ? ' …' : ''}</span></>
          : `${test.message}${test.listing?.length ? `: ${test.listing.slice(0, 5).join(', ')}` : ''}`}</div>}
      {st && <div className="small text-2">Warehouse drivers: {Object.values(st.warehouses).map((w) => <span key={w.label} className="tag" style={{ marginRight: 4 }}>{w.label} {w.driver ? '✓' : '✗'}</span>)}</div>}
    </div>
  )
}

export type Warehouses = Status['warehouses']
