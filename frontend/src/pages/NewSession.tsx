import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { api, fmtBytes, fmtTime, type Param, type PluginMeta, type SavedModel, type SessionDetail } from '../api'
import { usePlugins } from '../App'
import { initialValues, ParamForm } from '../components/ParamForm'
import { ErrorBox, HeadlineValue } from '../components/ui'

type Kind = 'dataset' | 'env'
type Mode = 'train' | 'evaluate'
type EvalPick = { key: string; session_id?: string; run_id?: string | null; model_id: string; pretrained?: boolean; variant?: string }
type SweepRow = { key: string; values: string }
type SpaceRow = { key: string; type: 'float' | 'int' | 'categorical'; low: string; high: string; log: boolean; choices: string }
const PRIORITIES: [number, string][] = [[-1, 'Low'], [0, 'Normal'], [1, 'High'], [2, 'Urgent']]

function defaultSpaceRow(p: Param & { key: string }): SpaceRow {
  if (p.type === 'float') {
    const d = Number(p.default) || 0.001
    return { key: p.key, type: 'float', low: String(p.log ? d / 10 : p.min ?? 0), high: String(p.log ? d * 10 : p.max ?? (d * 2 || 1)), log: !!p.log, choices: '' }
  }
  if (p.type === 'int') {
    const d = Number(p.default) || 1
    return { key: p.key, type: 'int', low: String(Math.max(p.min ?? 1, Math.round(d / 2))), high: String(Math.max(d * 2, d + 1)), log: false, choices: '' }
  }
  const choices = p.type === 'choice' ? p.choices!.join(', ') : p.type === 'bool' ? 'true, false' : p.type === 'int_list' ? '64; 128,64; 256,256' : String(p.default ?? '')
  return { key: p.key, type: 'categorical', low: '', high: '', log: false, choices }
}

function parseSeeds(mode: 'count' | 'list', count: string, start: string, list: string): number[] | string {
  if (mode === 'list') {
    const xs = list.split(/[,\s]+/).filter(Boolean).map(Number)
    if (!xs.length || xs.some((x) => !Number.isInteger(x))) return 'Seeds must be integers, e.g. 0, 1, 2'
    return xs
  }
  const n = Number(count), s = Number(start)
  if (!Number.isInteger(n) || n < 1 || n > 100) return 'Number of seeds must be 1-100'
  if (!Number.isInteger(s)) return 'Start seed must be an integer'
  return Array.from({ length: n }, (_, i) => s + i)
}

function parseSweepValues(p: Param | undefined, raw: string): unknown[] {
  const parts = raw.split(p?.type === 'int_list' ? ';' : ',').map((x) => x.trim()).filter(Boolean)
  if (!p) return parts
  if (p.type === 'float' || p.type === 'int') return parts.map(Number)
  if (p.type === 'bool') return parts.map((x) => ['true', '1', 'yes', 'on'].includes(x.toLowerCase()))
  return parts
}

const fits = (m: PluginMeta, p: PluginMeta) => (m.any_task || m.tasks!.includes(p.task!)) && m.modalities!.includes(p.modality!)

export default function NewSession() {
  const { plugins } = usePlugins()
  const [sp] = useSearchParams()
  const nav = useNavigate()
  const fromId = sp.get('from'), initId = sp.get('init'), evalId = sp.get('eval'), problemParam = sp.get('problem')
  const prefillId = fromId ?? initId ?? evalId

  const [mode, setMode] = useState<Mode>(evalId || sp.get('mode') === 'evaluate' ? 'evaluate' : 'train')
  const [prefill, setPrefill] = useState<SessionDetail>()
  const [kind, setKind] = useState<Kind>('dataset')
  const [problemId, setProblemId] = useState<string>()
  const [problemVals, setProblemVals] = useState<Record<string, unknown>>({})
  const [problemOpen, setProblemOpen] = useState(true)
  const [search, setSearch] = useState('')
  // training
  const [multi, setMulti] = useState(false)
  const [modelIds, setModelIds] = useState<string[]>([])
  const [modelVals, setModelVals] = useState<Record<string, Record<string, unknown>>>({})
  const [modelOpen, setModelOpen] = useState(true)
  const [start, setStart] = useState<'scratch' | 'saved'>('scratch')
  const [initSession, setInitSession] = useState<string>()
  const [initStrategy, setInitStrategy] = useState('match_seed')
  const [initVariant, setInitVariant] = useState('best')
  const [sweep, setSweep] = useState<SweepRow[]>([])
  const [searchMode, setSearchMode] = useState<'grid' | 'smart'>('grid')
  const [space, setSpace] = useState<SpaceRow[]>([])
  const [nTrials, setNTrials] = useState('20')
  const [parallel, setParallel] = useState('2')
  const [pruning, setPruning] = useState(true)
  const [priority, setPriority] = useState(0)
  const [repeat, setRepeat] = useState(false)
  const [cadKind, setCadKind] = useState<'interval' | 'daily' | 'weekly'>('daily')
  const [everyHours, setEveryHours] = useState('24')
  const [atTime, setAtTime] = useState('02:00')
  const [weekday, setWeekday] = useState('mon')
  const [registerAs, setRegisterAs] = useState('')
  const [runNow, setRunNow] = useState(true)
  const [startAt, setStartAt] = useState('')
  // evaluation
  const [picks, setPicks] = useState<EvalPick[]>([])
  const [saved, setSaved] = useState<SavedModel[]>()
  // run settings
  const [optionSpec, setOptionSpec] = useState<Param[]>([])
  const [options, setOptions] = useState<Record<string, unknown>>({})
  const [seedMode, setSeedMode] = useState<'count' | 'list'>('count')
  const [seedCount, setSeedCount] = useState('1')
  const [seedStart, setSeedStart] = useState('0')
  const [seedList, setSeedList] = useState('0, 1, 2')
  const [device, setDevice] = useState('auto')
  const [name, setName] = useState('')
  const [tags, setTags] = useState('')
  const [notes, setNotes] = useState('')
  const [err, setErr] = useState<unknown>()
  const [busy, setBusy] = useState(false)

  // ---- prefill: train again (?from), fine-tune (?init), evaluate (?eval[&run])
  useEffect(() => {
    if (!prefillId) return
    api.get<SessionDetail>(`/sessions/${prefillId}`).then((s) => {
      setPrefill(s)
      setKind(s.problem_kind)
      setProblemId(s.problem_id)
      setProblemVals(s.problem_params)
      setProblemOpen(false)
      setDevice(s.device)
      setTags(s.tags.join(', '))
      if (evalId) {
        const run = sp.get('run')
        const variant = sp.get('variant') ?? 'best'
        setPicks([{ key: `${s.id}:${run ?? 'best'}:${variant}`, session_id: s.id, run_id: run, model_id: s.model_id, variant }])
        return
      }
      setModelIds([s.model_id])
      setModelVals({ [s.model_id]: s.model_params })
      setModelOpen(false)
      setSeedMode('list')
      setSeedList(s.seeds.join(', '))
      setOptions(s.options ?? {})
      if (initId) {
        setStart('saved'); setInitSession(s.id)
      } else if (s.init_from && s.mode === 'train') {
        setStart('saved'); setInitSession(s.init_from.session_id); setInitStrategy(s.init_from.strategy); setInitVariant(s.init_from.variant ?? 'best')
      }
    }).catch(setErr)
  }, [prefillId]) // eslint-disable-line react-hooks/exhaustive-deps

  const problems = useMemo(() => (kind === 'dataset' ? plugins?.datasets : plugins?.envs) ?? [], [plugins, kind])
  // ?problem=<dataset id> (e.g. "Train on this source" from a live source page)
  useEffect(() => {
    if (!problemParam || !plugins || problemId) return
    const p = plugins.datasets.find((x) => x.id === problemParam)
    if (p) { setKind('dataset'); setProblemId(p.id); setProblemVals(initialValues(p.params)); setProblemOpen(false) }
  }, [problemParam, plugins]) // eslint-disable-line react-hooks/exhaustive-deps
  const problem = problems.find((p) => p.id === problemId)
  const compatible = useMemo(() => (plugins?.models ?? []).filter((m) => problem && fits(m, problem)), [plugins, problem])
  const models = modelIds.map((id) => compatible.find((m) => m.id === id)).filter(Boolean) as PluginMeta[]
  const model = models[0]
  // generative models train on any dataset of their modality; their sessions have task "generation"
  const effTask = model?.session_task ?? problem?.task

  // saved models usable on this problem
  useEffect(() => {
    if (!problem) return
    api.get<SavedModel[]>(`/models/saved?task=${effTask}&modality=${problem.modality}`)
      .then((xs) => setSaved([...xs].sort((a, b) => Number(b.problem_id === problem.id) - Number(a.problem_id === problem.id))))
      .catch(setErr)
  }, [problem?.id, effTask]) // eslint-disable-line react-hooks/exhaustive-deps

  // session options depend on the task and mode
  useEffect(() => {
    if (!problem) return
    api.get<Param[]>(`/options?task=${effTask}&mode=${mode}`).then((spec) => {
      setOptionSpec(spec)
      setOptions((o) => ({ ...initialValues(spec), ...Object.fromEntries(Object.entries(o).filter(([k]) => spec.some((p) => p.name === k))) }))
    }).catch(setErr)
  }, [problem?.id, mode, effTask]) // eslint-disable-line react-hooks/exhaustive-deps

  // fill in defaults for newly picked plugins (keeps prefilled values)
  useEffect(() => {
    if (problem) setProblemVals((v) => ({ ...initialValues(problem.params), ...v }))
  }, [problem?.id]) // eslint-disable-line react-hooks/exhaustive-deps

  const pickProblem = (p: PluginMeta) => {
    setProblemId(p.id)
    setProblemOpen(false)
    setProblemVals(initialValues(p.params, prefill?.problem_id === p.id ? prefill.problem_params : undefined))
    const ok = (m: PluginMeta) => fits(m, p)
    setModelIds((ids) => ids.filter((id) => plugins?.models.find((m) => m.id === id && ok(m))))
    setPicks((ps) => ps.filter((x) => plugins?.models.find((m) => m.id === x.model_id && ok(m))))
  }
  const toggleModel = (m: PluginMeta) => {
    const vals = initialValues(m.params, prefill?.model_id === m.id ? prefill.model_params : undefined)
    setModelVals((v) => ({ ...v, [m.id]: v[m.id] ?? vals }))
    if (!multi) {
      setModelIds([m.id]); setModelOpen(false); setInitSession(undefined); setStart('scratch'); setSweep([]); setSpace([])
      return
    }
    setModelIds((ids) => (ids.includes(m.id) ? ids.filter((x) => x !== m.id) : [...ids, m.id]))
  }

  const savedForModel = (saved ?? []).filter((s) => model && s.model_id === model.id)
  const initSaved = savedForModel.find((s) => s.session_id === initSession)
  const pretrainable = compatible.filter((m) => m.eval_without_init)
  const seeds = parseSeeds(seedMode, seedCount, seedStart, seedList)
  const sweepParams: (Param & { key: string })[] = [
    ...(model?.params ?? []).filter((p) => p.type !== 'json').map((p) => ({ ...p, key: `model.${p.name}` })),
    ...(problem?.params ?? []).filter((p) => p.type !== 'json').map((p) => ({ ...p, key: `problem.${p.name}` })),
  ]
  const grid = Object.fromEntries(sweep.filter((r) => r.values.trim()).map((r) => [r.key, parseSweepValues(sweepParams.find((p) => p.key === r.key), r.values)]))
  const nCombos = Object.values(grid).reduce((a, v) => a * Math.max(1, v.length), 1)
  const nSeeds = typeof seeds === 'string' ? 0 : seeds.length
  const smart = searchMode === 'smart' && space.length > 0
  const nSessions = mode === 'evaluate' ? picks.length : multi ? models.length : smart ? Number(nTrials) || 0 : sweep.length && searchMode === 'grid' ? nCombos : model ? 1 : 0

  const submit = async () => {
    if (!problem) return
    if (typeof seeds === 'string') return setErr(seeds)
    setBusy(true)
    setErr(undefined)
    const scheduled_at = startAt ? new Date(startAt).getTime() / 1000 : null
    const base = {
      problem_kind: kind, problem_id: problem.id, problem_params: problemVals, seeds, device, mode, options, notes, priority, scheduled_at,
      tags: tags.split(',').map((t) => t.trim()).filter(Boolean),
    }
    try {
      if (mode === 'evaluate') {
        const bodies = picks.map((p) => ({
          ...base, model_id: p.model_id, model_params: p.pretrained ? modelVals[p.model_id] ?? {} : {},
          init_from: p.pretrained ? null : { session_id: p.session_id, strategy: p.run_id ? 'run' : 'best', run_id: p.run_id ?? undefined, variant: p.variant ?? 'best' },
          name: picks.length === 1 ? name.trim() || undefined : undefined,
        }))
        if (bodies.length === 1) {
          const r = await api.post<{ id: string }>('/sessions', bodies[0])
          return nav(`/sessions/${r.id}`)
        }
        const g = await api.post<{ group_id: string }>('/groups', { kind: 'benchmark', sessions: bodies })
        return nav(`/compare?group=${g.group_id}`)
      }
      const trainBody = (m: PluginMeta) => ({
        ...base, model_id: m.id, model_params: modelVals[m.id] ?? {}, parent_id: fromId ?? null,
        init_from: !multi && start === 'saved' && initSession ? { session_id: initSession, strategy: initStrategy, variant: initVariant } : null,
        name: name.trim() || undefined,
      })
      if (multi && models.length > 1) {
        const g = await api.post<{ group_id: string }>('/groups', { kind: 'compare', sessions: models.map((m) => ({ ...trainBody(m), name: undefined })) })
        return nav(`/compare?group=${g.group_id}`)
      }
      if (repeat && !smart && !multi) {
        const tpl = { ...trainBody(model), name: undefined, scheduled_at: null }
        const cadence = cadKind === 'interval' ? { kind: 'interval', every_hours: Number(everyHours) } : cadKind === 'daily' ? { kind: 'daily', at: atTime } : { kind: 'weekly', at: atTime, weekday }
        await api.post('/schedules', { name: name.trim() || `${model.name} on ${problem.name}`, template: tpl, cadence, register_as: registerAs.trim() || null, start_now: runNow })
        return nav('/data?tab=schedules')
      }
      if (smart) {
        const spaceObj = Object.fromEntries(space.map((r) => [r.key, r.type === 'categorical'
          ? { type: 'categorical', choices: r.choices.split(sweepParams.find((x) => x.key === r.key)?.type === 'int_list' ? ';' : ',').map((c) => c.trim()).filter(Boolean)
            .map((c) => { const p = sweepParams.find((x) => x.key === r.key); return p?.type === 'bool' ? c === 'true' : p?.type === 'int' || p?.type === 'float' ? Number(c) : c }) }
          : { type: r.type, low: Number(r.low), high: Number(r.high), log: r.log }]))
        const s = await api.post<{ id: string }>('/searches', { name: name.trim() || undefined, base: { ...trainBody(model), name: undefined }, space: spaceObj,
          n_trials: Number(nTrials), parallel: Number(parallel), pruning })
        return nav(`/compare?group=${s.id}`)
      }
      if (searchMode === 'grid' && sweep.length && Object.keys(grid).length) {
        const g = await api.post<{ group_id: string }>('/sweeps', { base: { ...trainBody(model), name: undefined }, grid })
        return nav(`/compare?group=${g.group_id}`)
      }
      const r = await api.post<{ id: string }>('/sessions', trainBody(model))
      nav(`/sessions/${r.id}`)
    } catch (e) {
      setErr(e)
      setBusy(false)
    }
  }

  const grouped = useMemo(() => {
    const g = new Map<string, PluginMeta[]>()
    const ql = search.toLowerCase()
    for (const p of problems) {
      if (ql && !`${p.name} ${p.description} ${p.id}`.toLowerCase().includes(ql)) continue
      const key = kind === 'env' ? 'Environments' : p.source === 'live' ? 'Live sources' : `${p.modality} · ${p.task}`
      if (!g.has(key)) g.set(key, [])
      g.get(key)!.push(p)
    }
    return [...g.entries()].sort((a, b) => Number(b[0] === 'Live sources') - Number(a[0] === 'Live sources'))
  }, [problems, search, kind])

  if (!plugins) return <div className="muted">Loading plugins…</div>
  const sameProblem = prefill && prefill.problem_id === problemId && !!fromId
  const ready = problem && (mode === 'evaluate' ? picks.length > 0 : models.length > 0 && (start === 'scratch' || multi || !!initSession))
  const title = fromId ? 'Train again' : initId ? 'Fine-tune a saved model' : mode === 'evaluate' ? 'Evaluate models' : 'New training session'

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>{title}</h1>
          <div className="sub">
            {fromId && prefill
              ? <>Based on <Link to={`/sessions/${prefill.id}`}>{prefill.name}</Link>. Fields you change are highlighted; the new session is linked to it.</>
              : initId && prefill
                ? <>Continue training the saved model(s) of <Link to={`/sessions/${prefill.id}`}>{prefill.name}</Link> — on the same problem or a new one. Adjust training parameters below.</>
              : mode === 'evaluate'
                ? 'Run one or more trained models on a dataset or environment without training them, and compare the results.'
                : 'Pick a problem, a compatible model (or several), set parameters, and launch.'}
          </div>
        </div>
        {!fromId && !initId && (
          <div className="segmented">
            <button className={mode === 'train' ? 'on' : ''} onClick={() => setMode('train')}>Train</button>
            <button className={mode === 'evaluate' ? 'on' : ''} onClick={() => setMode('evaluate')}>Evaluate only</button>
          </div>
        )}
      </div>

      {/* ---- 1. problem */}
      <section className="card card-pad">
        <div className="row" style={{ justifyContent: 'space-between', marginBottom: 12 }}>
          <h2>1 · {mode === 'evaluate' ? 'Evaluate on' : 'Problem'}</h2>
          {problem && !problemOpen ? <button onClick={() => setProblemOpen(true)}>Change problem</button> : (
            <div className="row">
              <input style={{ width: 220 }} placeholder="Filter…" value={search} onChange={(e) => setSearch(e.target.value)} />
              <div className="segmented">
                <button className={kind === 'dataset' ? 'on' : ''} onClick={() => { setKind('dataset'); setProblemId(undefined) }}>Dataset</button>
                <button className={kind === 'env' ? 'on' : ''} onClick={() => { setKind('env'); setProblemId(undefined) }}>RL environment</button>
              </div>
            </div>
          )}
        </div>
        {problem && !problemOpen ? <Chosen p={problem} /> : (
          <div className="col" style={{ gap: 14 }}>
            {grouped.map(([g, ps]) => (
              <div key={g}>
                <div className="small text-2" style={{ marginBottom: 6, textTransform: 'capitalize', fontWeight: 550 }}>{g}</div>
                <div className="pick-list">
                  {ps.map((p) => (
                    <button key={p.id} className={`pick ${p.id === problemId ? 'on' : ''}`} onClick={() => pickProblem(p)}>
                      <span className="t">{p.name}</span>
                      <span className="d">{p.description}</span>
                      {p.source !== 'builtin' && <span className="tag">{p.source === 'upload' ? 'uploaded' : p.source === 'live' ? 'live' : 'custom'}</span>}
                    </button>
                  ))}
                </div>
              </div>
            ))}
            {!grouped.length && <div className="muted">Nothing matches. Add datasets or environments on the <Link to="/library">Library</Link> page.</div>}
          </div>
        )}
        {problem && problem.params.length > 0 && (
          <details style={{ marginTop: 14, paddingTop: 10, borderTop: '1px solid var(--border)' }} open={!!fromId || !prefill}>
            <summary>{problem.name} parameters</summary>
            <div style={{ marginTop: 10 }}>
              <ParamForm params={problem.params} values={problemVals} onChange={setProblemVals} baseline={sameProblem ? prefill!.problem_params : undefined} />
            </div>
          </details>
        )}
      </section>

      {/* ---- 2a. model(s) to train */}
      {problem && mode === 'train' && (
        <section className="card card-pad section">
          <div className="row" style={{ justifyContent: 'space-between', marginBottom: 12 }}>
            <h2>2 · Model{multi ? 's' : ''} <span className="muted small">({compatible.length} compatible with {problem.task} / {problem.modality})</span></h2>
            <div className="row">
              {!fromId && !initId && (
                <label className="check small" title="Train several model types on the same problem and compare them">
                  <input type="checkbox" checked={multi} onChange={(e) => { setMulti(e.target.checked); setModelOpen(true); if (!e.target.checked) setModelIds((ids) => ids.slice(0, 1)) }} />
                  Compare several models
                </label>
              )}
              {model && !modelOpen && <button onClick={() => setModelOpen(true)}>Change model</button>}
            </div>
          </div>
          {model && !modelOpen && !multi ? <Chosen p={model} /> : (
            <div className="pick-list">
              {compatible.map((m) => (
                <button key={m.id} className={`pick ${modelIds.includes(m.id) ? 'on' : ''}`} onClick={() => toggleModel(m)}>
                  <span className="t">{multi && (modelIds.includes(m.id) ? '☑ ' : '☐ ')}{m.name}</span>
                  <span className="d">{m.description}</span>
                  <span className="chips"><span className="tag">{m.framework}</span>{m.source !== 'builtin' && <span className="tag">custom</span>}</span>
                </button>
              ))}
            </div>
          )}

          {/* starting point */}
          {model && !multi && model.supports_init && (
            <div style={{ marginTop: 16, paddingTop: 14, borderTop: '1px solid var(--border)' }} className="col">
              <h3 className="text-2">Starting weights</h3>
              <div className="segmented" style={{ alignSelf: 'flex-start' }}>
                <button className={start === 'scratch' ? 'on' : ''} onClick={() => setStart('scratch')}>{model.framework === 'hf' || model.id === 'torch.torchvision' ? 'Pretrained / scratch (from params)' : 'From scratch'}</button>
                <button className={start === 'saved' ? 'on' : ''} onClick={() => setStart('saved')}>Fine-tune a saved model ({savedForModel.length})</button>
              </div>
              {start === 'saved' && (
                savedForModel.length === 0 ? <div className="muted small">No saved {model.name} models yet for {problem.task}/{problem.modality} problems.</div> : (
                  <>
                    <SavedList items={savedForModel} problemId={problem.id} selected={(s) => s.session_id === initSession} onPick={(s) => {
                      setInitSession(s.session_id)
                      setModelVals((v) => ({ ...v, [model.id]: { ...initialValues(model.params), ...s.model_params } }))
                    }} />
                    {initSaved && (
                      <label className="field" style={{ maxWidth: 360 }}>
                        <span className="lbl">Which seed's weights each new seed starts from</span>
                        <select value={initStrategy} onChange={(e) => setInitStrategy(e.target.value)}>
                          <option value="match_seed">same seed (falls back to the best)</option>
                          <option value="best">the best seed</option>
                        </select>
                        <span className="help">Architecture parameters were copied from the saved model; change training parameters freely.</span>
                      </label>
                    )}
                    {initSaved?.has_final && (
                      <label className="field" style={{ maxWidth: 360 }}>
                        <span className="lbl">Which weights</span>
                        <select value={initVariant} onChange={(e) => setInitVariant(e.target.value)}>
                          <option value="best">best (by validation)</option><option value="final">final (last epoch)</option>
                        </select>
                      </label>
                    )}
                  </>
                )
              )}
            </div>
          )}

          {models.map((m) => (
            <details key={m.id} open style={{ marginTop: 16, paddingTop: 14, borderTop: '1px solid var(--border)' }}>
              <summary>{m.name} parameters</summary>
              <div style={{ marginTop: 10 }}>
                <ParamForm params={m.params} values={modelVals[m.id] ?? {}} onChange={(v) => setModelVals((all) => ({ ...all, [m.id]: v }))}
                  baseline={fromId && prefill?.model_id === m.id ? prefill.model_params : undefined} />
              </div>
            </details>
          ))}

          {/* hyperparameter search */}
          {model && !multi && (
            <div style={{ marginTop: 16, paddingTop: 14, borderTop: '1px solid var(--border)' }} className="col">
              <div className="row" style={{ justifyContent: 'space-between' }}>
                <h3 className="text-2">Hyperparameter search <span className="muted small">(optional)</span></h3>
                <div className="segmented">
                  <button className={searchMode === 'grid' ? 'on' : ''} onClick={() => setSearchMode('grid')}>Grid</button>
                  <button className={searchMode === 'smart' ? 'on' : ''} onClick={() => setSearchMode('smart')}>Smart search (Optuna)</button>
                </div>
              </div>
              {searchMode === 'grid' ? (
                <>
                  <div className="small text-2">Every combination of the listed values.</div>
                  {sweep.map((row, i) => {
                    const p = sweepParams.find((x) => x.key === row.key)
                    return (
                      <div key={i} className="row" style={{ flexWrap: 'nowrap' }}>
                        <select style={{ maxWidth: 260 }} value={row.key} onChange={(e) => setSweep((s) => s.map((r, j) => (j === i ? { ...r, key: e.target.value } : r)))}>
                          {sweepParams.map((x) => <option key={x.key} value={x.key}>{x.key}</option>)}
                        </select>
                        <input placeholder={p?.type === 'choice' ? p.choices!.join(', ') : p?.type === 'int_list' ? '64,64; 128,128' : 'e.g. 0.001, 0.003, 0.01'}
                          value={row.values} onChange={(e) => setSweep((s) => s.map((r, j) => (j === i ? { ...r, values: e.target.value } : r)))} />
                        <button className="ghost sm" onClick={() => setSweep((s) => s.filter((_, j) => j !== i))} aria-label="remove">✕</button>
                      </div>
                    )
                  })}
                  <button className="sm" style={{ alignSelf: 'flex-start' }} onClick={() => setSweep((s) => [...s, { key: sweepParams.find((p) => p.name === 'lr' || p.name === 'learning_rate')?.key ?? sweepParams[0]?.key, values: '' }])}>+ Add parameter</button>
                  {sweep.length > 0 && <div className="small text-2">{nCombos} combination{nCombos === 1 ? '' : 's'} × {nSeeds} seed{nSeeds === 1 ? '' : 's'} = <b>{nCombos * nSeeds} runs</b>. Values comma-separated{sweepParams.some((x) => x.type === 'int_list') ? ' (lists separated by ;)' : ''}.</div>}
                </>
              ) : (
                <>
                  <div className="small text-2">Bayesian optimization (TPE): each trial picks parameters based on how earlier trials scored.
                    Trials are ranked on the <b>validation</b> metric (the test set stays untouched), and with pruning, trials that are clearly worse than the median stop early.</div>
                  {space.map((r, i) => {
                    const set = (patch: Partial<SpaceRow>) => setSpace((s) => s.map((x, j) => (j === i ? { ...x, ...patch } : x)))
                    return (
                      <div key={i} className="row" style={{ flexWrap: 'wrap' }}>
                        <select style={{ maxWidth: 240 }} value={r.key} onChange={(e) => { const p = sweepParams.find((x) => x.key === e.target.value)!; setSpace((s) => s.map((x, j) => (j === i ? defaultSpaceRow(p) : x))) }}>
                          {sweepParams.map((x) => <option key={x.key} value={x.key}>{x.key}</option>)}
                        </select>
                        {r.type === 'categorical' ? (
                          <input style={{ flex: 1, minWidth: 200 }} placeholder="choices, comma separated" value={r.choices} onChange={(e) => set({ choices: e.target.value })} />
                        ) : (
                          <>
                            <span className="small muted">{r.type} from</span><input style={{ width: 110 }} value={r.low} onChange={(e) => set({ low: e.target.value })} />
                            <span className="small muted">to</span><input style={{ width: 110 }} value={r.high} onChange={(e) => set({ high: e.target.value })} />
                            <label className="check small"><input type="checkbox" checked={r.log} onChange={(e) => set({ log: e.target.checked })} />log scale</label>
                          </>
                        )}
                        <button className="ghost sm" onClick={() => setSpace((s) => s.filter((_, j) => j !== i))} aria-label="remove">✕</button>
                      </div>
                    )
                  })}
                  <button className="sm" style={{ alignSelf: 'flex-start' }} onClick={() => {
                    const used = new Set(space.map((r) => r.key))
                    const p = sweepParams.find((x) => !used.has(x.key) && (x.name === 'lr' || x.name === 'learning_rate')) ?? sweepParams.find((x) => !used.has(x.key))
                    if (p) setSpace((s) => [...s, defaultSpaceRow(p)])
                  }}>+ Add parameter</button>
                  {space.length > 0 && (
                    <div className="row">
                      <label className="row small">trials <input style={{ width: 70 }} value={nTrials} onChange={(e) => setNTrials(e.target.value)} /></label>
                      <label className="row small">in parallel <input style={{ width: 60 }} value={parallel} onChange={(e) => setParallel(e.target.value)} /></label>
                      <label className="check small"><input type="checkbox" checked={pruning} onChange={(e) => setPruning(e.target.checked)} />prune bad trials early</label>
                      <span className="small text-2">{nTrials} trials × {nSeeds} seed{nSeeds === 1 ? '' : 's'}</span>
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </section>
      )}

      {/* ---- 2b. models to evaluate */}
      {problem && mode === 'evaluate' && (
        <section className="card card-pad section">
          <h2 style={{ marginBottom: 6 }}>2 · Models to evaluate <span className="muted small">({picks.length} selected)</span></h2>
          <div className="small text-2" style={{ marginBottom: 12 }}>Saved models trained on any {problem.task}/{problem.modality} problem. Inputs must match this problem's shape (features / observation space).</div>
          {(saved ?? []).length === 0 && <div className="muted small">No saved models for {problem.task}/{problem.modality} yet. Train one with “Save model” on.</div>}
          <SavedList items={saved ?? []} multi problemId={problem.id}
            selected={(s) => picks.some((p) => p.session_id === s.session_id)}
            onPick={(s) => setPicks((ps) => ps.some((p) => p.session_id === s.session_id)
              ? ps.filter((p) => p.session_id !== s.session_id)
              : [...ps, { key: s.session_id, session_id: s.session_id, run_id: null, model_id: s.model_id }])}
            extra={(s) => {
              const p = picks.find((x) => x.session_id === s.session_id)
              if (!p || (s.runs.length < 2 && !s.has_final)) return null
              return (
                <div className="row" style={{ gap: 4, marginTop: 4 }}>
                  {s.runs.length > 1 && <select className="sm" style={{ width: 'auto' }} value={p.run_id ?? ''} onClick={(e) => e.stopPropagation()}
                    onChange={(e) => setPicks((ps) => ps.map((x) => (x === p ? { ...x, run_id: e.target.value || null } : x)))}>
                    <option value="">best seed</option>
                    {s.runs.map((r) => <option key={r.run_id} value={r.run_id}>seed {r.seed}</option>)}
                  </select>}
                  {s.has_final && <select className="sm" style={{ width: 'auto' }} value={p.variant ?? 'best'} onClick={(e) => e.stopPropagation()}
                    onChange={(e) => setPicks((ps) => ps.map((x) => (x === p ? { ...x, variant: e.target.value } : x)))}>
                    <option value="best">best weights</option><option value="final">final weights</option>
                  </select>}
                </div>
              )
            }} />
          {pretrainable.length > 0 && (
            <div style={{ marginTop: 16, paddingTop: 14, borderTop: '1px solid var(--border)' }} className="col">
              <h3 className="text-2">Pretrained checkpoints (zero-shot, no saved model needed)</h3>
              <div className="pick-list">
                {pretrainable.map((m) => {
                  const on = picks.some((p) => p.pretrained && p.model_id === m.id)
                  return (
                    <button key={m.id} className={`pick ${on ? 'on' : ''}`} onClick={() => {
                      setModelVals((v) => ({ ...v, [m.id]: v[m.id] ?? initialValues(m.params) }))
                      setPicks((ps) => (on ? ps.filter((p) => !(p.pretrained && p.model_id === m.id)) : [...ps, { key: `pre:${m.id}`, model_id: m.id, pretrained: true }]))
                    }}>
                      <span className="t">{on ? '☑' : '☐'} {m.name}</span>
                      <span className="d">Evaluate the checkpoint named in its parameters (e.g. a Hugging Face model id)</span>
                    </button>
                  )
                })}
              </div>
              {picks.filter((p) => p.pretrained).map((p) => {
                const m = pretrainable.find((x) => x.id === p.model_id)!
                return (
                  <details key={p.key} open>
                    <summary>{m.name} parameters</summary>
                    <div style={{ marginTop: 10 }}>
                      <ParamForm params={m.params.filter((x) => x.group !== 'Training' && x.group !== 'Early stopping')} values={modelVals[m.id] ?? {}}
                        onChange={(v) => setModelVals((all) => ({ ...all, [m.id]: v }))} />
                    </div>
                  </details>
                )
              })}
            </div>
          )}
        </section>
      )}

      {/* ---- 3. run settings */}
      {ready && (
        <section className="card card-pad section">
          <h2 style={{ marginBottom: 12 }}>3 · Runs & options</h2>
          <div className="grid-3">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <span style={{ fontWeight: 550, fontSize: '0.88rem' }}>Seeds</span>
              <div className="segmented" style={{ alignSelf: 'flex-start' }}>
                <button className={seedMode === 'count' ? 'on' : ''} onClick={() => setSeedMode('count')}>Count</button>
                <button className={seedMode === 'list' ? 'on' : ''} onClick={() => setSeedMode('list')}>Explicit list</button>
              </div>
              {seedMode === 'count' ? (
                <div className="row" style={{ flexWrap: 'nowrap' }}>
                  <input aria-label="number of seeds" value={seedCount} onChange={(e) => setSeedCount(e.target.value)} inputMode="numeric" />
                  <span className="muted small" style={{ whiteSpace: 'nowrap' }}>starting at</span>
                  <input aria-label="first seed" value={seedStart} onChange={(e) => setSeedStart(e.target.value)} inputMode="numeric" />
                </div>
              ) : <input aria-label="seed list" value={seedList} onChange={(e) => setSeedList(e.target.value)} />}
              <span className="small muted">{typeof seeds === 'string' ? <span className="err-text">{seeds}</span> : `${seeds.length} run${seeds.length > 1 ? 's' : ''} per session: ${seeds.slice(0, 12).join(', ')}${seeds.length > 12 ? '…' : ''}`}</span>
            </div>
            <label className="field">
              <span className="lbl">Device</span>
              <select value={device} onChange={(e) => setDevice(e.target.value)}>
                <option value="auto">auto (GPU for neural nets, CPU for RL/trees)</option>
                <option value="cuda">GPU (cuda)</option>
                <option value="cpu">CPU</option>
              </select>
              <span className="help">GPU runs share the GPU queue limit set in Settings.</span>
            </label>
            {nSessions <= 1 && (
              <label className="field">
                <span className="lbl">Session name</span>
                <input placeholder="auto" value={name} onChange={(e) => setName(e.target.value)} />
              </label>
            )}
            <label className="field">
              <span className="lbl">Tags</span>
              <input placeholder="baseline, sweep-1" value={tags} onChange={(e) => setTags(e.target.value)} />
            </label>
            <label className="field">
              <span className="lbl">Queue priority</span>
              <select value={priority} onChange={(e) => setPriority(Number(e.target.value))}>{PRIORITIES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select>
              <span className="help">Higher priority runs start first when the queue is busy.</span>
            </label>
            <label className="field">
              <span className="lbl">Start at (optional)</span>
              <input type="datetime-local" value={startAt} onChange={(e) => setStartAt(e.target.value)} />
              <span className="help">{startAt ? `Waits in the queue until ${new Date(startAt).toLocaleString()}` : 'Empty = as soon as a slot is free'}</span>
            </label>
          </div>
          {optionSpec.length > 0 && (
            <div style={{ marginTop: 18, paddingTop: 14, borderTop: '1px solid var(--border)' }}>
              <ParamForm params={optionSpec} values={options} onChange={setOptions}
                baseline={fromId && prefill ? prefill.options : undefined} />
            </div>
          )}
          {mode === 'train' && !multi && !smart && (
            <div style={{ marginTop: 14, paddingTop: 12, borderTop: '1px solid var(--border)' }} className="col">
              <label className="check"><input type="checkbox" checked={repeat} onChange={(e) => setRepeat(e.target.checked)} /><b>Repeat on a schedule</b>
                <span className="small text-2">{problem?.source === 'live' ? '— each run retrains on a fresh snapshot of the live data' : '— retrain this configuration regularly'}</span></label>
              {repeat && (
                <div className="row">
                  <select style={{ width: 'auto' }} value={cadKind} onChange={(e) => setCadKind(e.target.value as typeof cadKind)}>
                    <option value="interval">every N hours</option><option value="daily">daily at</option><option value="weekly">weekly on</option>
                  </select>
                  {cadKind === 'interval' && <input style={{ width: 80 }} value={everyHours} onChange={(e) => setEveryHours(e.target.value)} />}
                  {cadKind === 'weekly' && <select style={{ width: 'auto' }} value={weekday} onChange={(e) => setWeekday(e.target.value)}>{['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'].map((d) => <option key={d}>{d}</option>)}</select>}
                  {cadKind !== 'interval' && <input type="time" style={{ width: 120 }} value={atTime} onChange={(e) => setAtTime(e.target.value)} />}
                  <input style={{ maxWidth: 260 }} placeholder="register each result as… (optional)" value={registerAs} onChange={(e) => setRegisterAs(e.target.value)} />
                  <label className="check small"><input type="checkbox" checked={runNow} onChange={(e) => setRunNow(e.target.checked)} />also run once now</label>
                </div>
              )}
            </div>
          )}
          <label className="field" style={{ marginTop: 14 }}>
            <span className="lbl">Notes</span>
            <textarea placeholder={fromId ? 'What are you trying this time?' : 'Hypothesis, context…'} value={notes} onChange={(e) => setNotes(e.target.value)} />
          </label>
        </section>
      )}

      <div className="sticky-actions row">
        <ErrorBox error={err} />
        <div className="grow small text-2">
          {problem ? <b>{problem.name}</b> : 'No problem selected'}
          {mode === 'train' && <> · {models.length ? models.map((m) => m.name).join(', ') : 'no model'}</>}
          {mode === 'evaluate' && <> · {picks.length} model{picks.length === 1 ? '' : 's'} to evaluate</>}
          {ready && typeof seeds !== 'string' && <> · {nSessions} session{nSessions === 1 ? '' : 's'} × {seeds.length} seed{seeds.length > 1 ? 's' : ''}</>}
        </div>
        <button className="primary" disabled={!ready || busy} onClick={submit}>
          {busy ? 'Starting…' : mode === 'evaluate' ? (picks.length > 1 ? `Evaluate ${picks.length} models` : 'Evaluate') : repeat && !smart && !multi ? 'Create schedule' : smart ? `Start search (${nTrials} trials)` : nSessions > 1 ? `Start ${nSessions} sessions` : startAt ? 'Schedule training' : 'Start training'}
        </button>
      </div>
    </div>
  )
}

function Chosen({ p }: { p: PluginMeta }) {
  return (
    <div className="pick on card" style={{ maxWidth: 520, padding: '10px 12px' }}>
      <span className="t">{p.name}</span>
      <span className="d">{p.description}</span>
      <span className="mono small muted">{p.id}</span>
    </div>
  )
}

function SavedList({ items, selected, onPick, multi, extra, problemId }: {
  items: SavedModel[]; selected: (s: SavedModel) => boolean; onPick: (s: SavedModel) => void; multi?: boolean
  extra?: (s: SavedModel) => ReactNode; problemId?: string
}) {
  return (
    <div className="pick-list" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))' }}>
      {items.map((s) => (
        <div key={s.session_id} role="button" tabIndex={0} className={`btn pick ${selected(s) ? 'on' : ''}`}
          onClick={() => onPick(s)} onKeyDown={(e) => e.key === 'Enter' && onPick(s)}>
          <span className="t">{multi && (selected(s) ? '☑ ' : '☐ ')}{s.session_name}</span>
          <span className="d">{s.model_name} · trained on {s.problem_id} · {fmtTime(s.created_at)}</span>
          {problemId && s.problem_id !== problemId && <span className="tag" title="Its input shape must match this problem">different problem</span>}
          <span className="small"><HeadlineValue h={s.headline} /></span>
          <span className="muted small">{s.runs.length} seed{s.runs.length > 1 ? 's' : ''} saved · {fmtBytes(s.bytes)}</span>
          {extra?.(s)}
        </div>
      ))}
    </div>
  )
}
