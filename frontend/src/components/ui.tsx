import { useEffect, useRef, useState, type ReactNode } from 'react'
import { fmtMetric, type Headline } from '../api'

const STATUS_LABEL: Record<string, string> = {
  scheduled: 'Scheduled', running: 'Running', queued: 'Queued', completed: 'Completed', failed: 'Failed', stopped: 'Stopped', stopping: 'Stopping',
  partial: 'Partly failed', empty: 'Empty',
}

export function StatusBadge({ status }: { status: string }) {
  return <span className={`badge ${status}`}><span className="dot" />{STATUS_LABEL[status] ?? status}</span>
}

export function Progress({ value }: { value: number }) {
  return <div className="progress" role="progressbar" aria-valuenow={Math.round(value * 100)}><div style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} /></div>
}

export function HeadlineValue({ h }: { h: Headline | null }) {
  if (!h) return <span className="muted">—</span>
  return (
    <span className="num" title={h.key}>
      {fmtMetric(h.key, h.mean)}
      {h.n > 1 && <span className="muted small"> ± {fmtMetric(h.key, h.std)}</span>}
      <span className="muted small"> {h.key}</span>
    </span>
  )
}

export function ErrorBox({ error }: { error: unknown }) {
  if (!error) return null
  return <div className="error-box">{error instanceof Error ? error.message : String(error)}</div>
}

/** Poll an async loader on an interval; returns [data, error, reload]. */
export function usePoll<T>(loader: () => Promise<T>, ms: number, deps: unknown[] = []): [T | undefined, unknown, () => void] {
  const [data, setData] = useState<T>()
  const [err, setErr] = useState<unknown>()
  const ref = useRef(loader)
  ref.current = loader
  const [tick, setTick] = useState(0)
  useEffect(() => {
    let alive = true
    const run = () => ref.current().then((d) => { if (alive) { setData(d); setErr(undefined) } }).catch((e) => alive && setErr(e))
    run()
    const t = ms > 0 ? setInterval(run, ms) : undefined
    return () => { alive = false; if (t) clearInterval(t) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ms, tick, ...deps])
  return [data, err, () => setTick((x) => x + 1)]
}

export function Modal({ title, children, onClose }: { title: string; children: ReactNode; onClose: () => void }) {
  useEffect(() => {
    const k = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', k)
    return () => window.removeEventListener('keydown', k)
  }, [onClose])
  return (
    <div onClick={onClose} style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.45)', zIndex: 50, display: 'grid', placeItems: 'center', padding: 16 }}>
      <div className="card" onClick={(e) => e.stopPropagation()} style={{ width: 'min(900px, 100%)', maxHeight: '88vh', overflow: 'auto' }}>
        <div className="card-head"><h2>{title}</h2><button className="ghost" onClick={onClose} aria-label="Close">✕</button></div>
        <div className="card-pad">{children}</div>
      </div>
    </div>
  )
}

/** Two-click confirm button (browser confirm() dialogs are avoided). */
export function ConfirmButton({ children, onConfirm, className = '', disabled }: { children: ReactNode; onConfirm: () => void; className?: string; disabled?: boolean }) {
  const [armed, setArmed] = useState(false)
  useEffect(() => {
    if (!armed) return
    const t = setTimeout(() => setArmed(false), 3500)
    return () => clearTimeout(t)
  }, [armed])
  return (
    <button className={className} disabled={disabled} onClick={() => (armed ? (setArmed(false), onConfirm()) : setArmed(true))}>
      {armed ? 'Click again to confirm' : children}
    </button>
  )
}

const HEALTH: Record<string, [string, string]> = {
  healthy: ['completed', 'Healthy'], warning: ['queued', 'Warning'], critical: ['failed', 'Critical'], no_data: ['', 'No data'],
}

/** Deployment health: color dot + label (never color alone). */
export function HealthBadge({ health }: { health: string }) {
  const [cls, label] = HEALTH[health] ?? ['', health]
  const icon = health === 'critical' ? '✕' : health === 'warning' ? '!' : health === 'healthy' ? '✓' : '–'
  return <span className={`badge ${cls}`}><span className="dot" />{icon} {label}</span>
}

export function StageBadge({ stage }: { stage: string }) {
  const cls = stage === 'production' ? 'completed' : stage === 'staging' ? 'queued' : stage === 'archived' ? 'stopped' : ''
  return <span className={`badge ${cls}`}><span className="dot" />{stage === 'none' ? 'registered' : stage}</span>
}
