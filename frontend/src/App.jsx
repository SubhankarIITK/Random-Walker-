import { useState, useCallback, useMemo, useEffect, useRef } from 'react'
import JSZip from 'jszip'
import WalkerCanvas from './components/WalkerCanvas'
import Controls    from './components/Controls'
import StatsPanel  from './components/StatsPanel'
import PanelDivider from './components/PanelDivider'
import EnsembleInspector from './components/EnsembleInspector'
import { prefixMSD } from './lib/ensemble'
import { simulate, getEnsemble, exportEnsemble, inspectWalker } from './api'

export default function App() {
  // ── Config state ──────────────────────────────────────────────────────
  const layoutRef = useRef(null)
  const [panelWidth, setPanelWidth] = useState(320)
  const [maxPanelWidth, setMaxPanelWidth] = useState(720)
  const [resizingPanel, setResizingPanel] = useState(false)
  const [mobileSection, setMobileSection] = useState('controls')
  const [viewerMode, setViewerMode] = useState('sample')
  const [walkerId, setWalkerId] = useState(1)
  const [inspected, setInspected] = useState(null)
  const [inspecting, setInspecting] = useState(false)
  const [prefixCount, setPrefixCount] = useState(1)
  const inspectionCache = useRef(new Map())
  const [expanded, setExpanded] = useState(false)
  const [dimension, setDimension] = useState(3)
  const [projection, setProjection] = useState([0, 1, 2])
  const [nSteps,    setNSteps]    = useState(2000)
  const [nWalkers,  setNWalkers]  = useState(500)
  const [nVisualWalkers, setNVisualWalkers] = useState(5)
  const [seed,      setSeed]      = useState('')

  // ── Simulation data ───────────────────────────────────────────────────
  const [trajectory, setTrajectory] = useState(null)
  const [simData,    setSimData]    = useState(null)
  const [ensData,    setEnsData]    = useState(null)
  const viewDimension = simData?.config.dimension ?? dimension
  const viewProjection = useMemo(() => projection.slice(0, Math.min(3, viewDimension)).every(i => i < viewDimension) ? projection : [0, 1, 2], [projection, viewDimension])
  const [isLoading,  setIsLoading]  = useState(false)
  const [isExporting, setIsExporting] = useState(false)
  const [error,      setError]      = useState(null)

  useEffect(() => {
    const element = layoutRef.current
    if (!element) return
    const measure = () => {
      const sidebarWidth = element.querySelector('.sidebar').getBoundingClientRect().width
      const maximum = Math.max(260, Math.min(720, Math.floor(element.clientWidth - sidebarWidth - 336)))
      setMaxPanelWidth(maximum)
    }
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(element)
    window.addEventListener('resize', measure)
    return () => { observer.disconnect(); window.removeEventListener('resize', measure) }
  }, [expanded])
  const actualPanelWidth = Math.min(panelWidth, maxPanelWidth)

  useEffect(() => {
    if (!ensData || viewerMode === 'sample') { setInspecting(false); return }
    const key = `${ensData.config.seed}:${walkerId}`
    const cached = inspectionCache.current.get(key)
    if (cached) { setInspected(cached); setInspecting(false); return }
    const controller = new AbortController()
    setInspecting(true)
    const timer = setTimeout(async () => {
      try {
        const result = await inspectWalker({ ...ensData.config, walker_id: walkerId }, controller.signal)
        if (controller.signal.aborted) return
        const expected = ensData.walkers.final_positions[walkerId - 1]
        if (!expected.every((x, i) => x === result.trajectory.at(-1)[i])) throw new Error('Replayed endpoint does not match this ensemble run.')
        if (inspectionCache.current.size >= 12) inspectionCache.current.delete(inspectionCache.current.keys().next().value)
        inspectionCache.current.set(key, result)
        setInspected(result)
      } catch (err) {
        if (err.name !== 'AbortError') setError(err.message)
      } finally {
        if (!controller.signal.aborted) setInspecting(false)
      }
    }, 180)
    return () => { clearTimeout(timer); controller.abort() }
  }, [ensData, viewerMode, walkerId])

  const viewingEnsemble = viewerMode !== 'sample' && inspected !== null
  const displayedData = viewingEnsemble ? inspected : simData
  const displayedTrajectory = displayedData?.trajectory
  const displayedTrajectories = viewingEnsemble ? undefined : simData?.trajectories
  const endpoints = viewerMode !== 'sample' ? ensData?.walkers.final_positions : undefined

  const handleRun = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    setViewerMode('sample')
    setInspected(null)
    setWalkerId(1)
    inspectionCache.current.clear()
    setTrajectory(null)
    setSimData(null)
    setEnsData(null)

    const s = seed ? parseInt(seed, 10) : null

    try {
      // Run both requests in parallel
      const [sim, ens] = await Promise.all([
        simulate({ dimension, n_steps: nSteps, n_visual_walkers: nVisualWalkers, seed: s }),
        getEnsemble({ dimension, n_steps: nSteps, n_walkers: nWalkers, seed: s }),
      ])

      setSimData(sim)
      setEnsData(ens)
      setPrefixCount(ens.config.n_walkers)
      setTrajectory(sim.trajectory)
      setMobileSection('viewer')
      
    } catch (err) {
      setError(
        err.message.includes('fetch')
          ? 'Cannot reach backend. Run: cd backend && uvicorn main:app --port 8000'
          : err.message
      )
    } finally {
      setIsLoading(false)
    }
  }, [dimension, nSteps, nWalkers, nVisualWalkers, seed])

  const handleExportData = useCallback(async () => {
    if (!simData || !ensData) return
    const zip = new JSZip()

    const config = ensData.config
    const labels = Array.from({ length: config.dimension }, (_, i) => `x${i + 1}`)
    for (const [walker, path] of (simData.trajectories || [trajectory]).entries()) {
      const rows = path.map((pt, i) => `${simData.steps[i]},${pt.join(',')}`)
      zip.file(`visual_walker_${walker + 1}.csv`, `Step,${labels.join(',')}\n${rows.join('\n')}\n`)
    }
    const endpoints = ensData.walkers.final_positions
    const squared = endpoints.map(p => p.reduce((sum, x) => sum + x * x, 0))
    const recomputed = squared.reduce((sum, x) => sum + x, 0) / endpoints.length
    const rows = endpoints.map((p, i) => `${i + 1},${p.join(',')},${squared[i]},${ensData.walkers.n_returns[i]}`)
    zip.file('ensemble_endpoints.csv', `Walker,${labels.join(',')},R_squared,Returns\n${rows.join('\n')}\n`)
    zip.file('ensemble_verification.json', JSON.stringify({
      config, exported_walkers: endpoints.length,
      walker_step_updates: config.n_walkers * config.n_steps,
      msd_recomputed_from_endpoints: recomputed,
      msd_reported: ensData.summary.msd_sim,
      absolute_difference: Math.abs(recomputed - ensData.summary.msd_sim),
      displayed_paths: simData.trajectories.length,
      displayed_paths_source: 'Separate visual simulation; HUD follows visual walker 1',
    }, null, 2))
    zip.file('HOW_TO_VERIFY.txt', `This run used ${config.n_walkers} ensemble walkers, each taking ${config.n_steps} steps.\nSeed: ${config.seed}\nOpen ensemble_endpoints.csv. It has one row per ensemble walker.\nFor each row, R_squared = sum of squared coordinates. Average the R_squared column to reproduce MSD = ${ensData.summary.msd_sim}.\nVisual walkers are separately generated and do not identify ensemble members.\nUse Export ALL ensemble paths to obtain every walker at every step by exact deterministic replay.\n`)

    const prefixes = prefixMSD(endpoints, config.n_steps)
    zip.file('ensemble_convergence.csv', 'Walkers,MSD,Standard_error,Approx_95CI_low,Approx_95CI_high,Error_pct\n' + prefixes.map(r => `${r.count},${r.msd},${r.se ?? ''},${r.lo ?? ''},${r.hi ?? ''},${r.errorPct}`).join('\n'))
    if (inspected) {
      zip.file(`ensemble_walker_${inspected.config.walker_id}.csv`, `Step,${labels.join(',')}\n` + inspected.trajectory.map((p, t) => `${t},${p.join(',')}`).join('\n'))
    }

    // 2. MSD Series CSV
    if (ensData.series) {
      const s = ensData.series
      let msdCsv = `Step,MSD_Sim,MSD_Theory,95CI_Low,95CI_High\n`
      s.t.forEach((t, i) => {
        msdCsv += `${t},${s.msd[i]},${s.msd_theory[i]},${s.msd_lo[i]},${s.msd_hi[i]}\n`
      })
      zip.file('msd_series.csv', msdCsv)
    }

    // 3. Coordinate Distributions CSVs
    if (ensData.coords) {
      ensData.coords.forEach((c) => {
        let distCsv = `Bin_Center,Density_Sim,Density_Theory\n`
        c.hist.x.forEach((x, i) => {
          const t_y = c.theory ? c.theory.y[i] || '' : ''
          distCsv += `${x},${c.hist.y[i]},${t_y}\n`
        })
        zip.file(`distribution_x${c.index}.csv`, distCsv)
      })
    }
    if (ensData.radial) {
      const r = ensData.radial
      let distCsv = `Radius,Density_Sim,Density_Theory\n`
      r.hist.x.forEach((x, i) => {
        const t_y = r.theory ? r.theory.y[i] || '' : ''
        distCsv += `${x},${r.hist.y[i]},${t_y}\n`
      })
      zip.file(`distribution_radial.csv`, distCsv)
    }

    // 4. Statistics JSON
    zip.file('statistics.json', JSON.stringify({
      parameters: ensData.config,
      singleWalker: simData,
      ensemble: ensData.summary,
      recurrence: ensData.returns,
      scaling: ensData.scaling
    }, null, 2))

    const blob = await zip.generateAsync({ type: 'blob' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `random_walk_${config.dimension}D_${config.n_steps}steps.zip`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }, [dimension, nSteps, nWalkers, seed, trajectory, simData, ensData, inspected])

  // ── Overlay hint text ─────────────────────────────────────────────────
  const handleExportAll = async () => {
    if (!ensData) return
    setIsExporting(true)
    setError(null)
    try {
      const blob = await exportEnsemble(ensData.config)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `ensemble_${ensData.config.n_walkers}_walkers_all_steps.zip`
      a.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsExporting(false)
    }
  }

  const overlayHint = trajectory
    ? `${viewDimension}D walk · ${(simData?.config.n_steps ?? nSteps).toLocaleString()} steps${viewDimension >= 4 ? ' · 3D projection shown' : ''}\nDrag to orbit · Scroll to zoom`
    : null

  return (
    <div ref={layoutRef} className={`layout ${expanded ? 'viewer-expanded' : ''} ${resizingPanel ? 'panel-resizing' : ''} mobile-${mobileSection}`}
      style={{ '--stats-user-width': `${actualPanelWidth}px` }}>
      <nav className="mobile-navigation" aria-label="Workspace sections">
        {[['controls', 'Controls'], ['viewer', 'Walk viewer'], ['statistics', 'Statistics']].map(([key, label]) =>
          <button key={key} type="button" className={mobileSection === key ? 'active' : ''} aria-pressed={mobileSection === key}
            onClick={() => { setExpanded(false); setMobileSection(key) }}>{label}</button>)}
      </nav>
      {/* Left sidebar */}
      <div className="sidebar">
        <div className="sidebar-header">
          <h1>Random Walk Laboratory</h1>
          <p>IDC612 — IIT Kanpur</p>
        </div>

        <Controls
          dimension={dimension}   setDimension={setDimension}
          viewDimension={viewDimension}
          projection={viewProjection} setProjection={setProjection}
          nSteps={nSteps}         setNSteps={setNSteps}
          nWalkers={nWalkers}     setNWalkers={setNWalkers}
          nVisualWalkers={nVisualWalkers} setNVisualWalkers={setNVisualWalkers}
          seed={seed}             setSeed={setSeed}
          onRun={handleRun}
          onExport={handleExportData}
          onExportAll={handleExportAll}
          isExporting={isExporting}
          ensembleConfig={ensData?.config}
          inspector={<EnsembleInspector data={ensData} mode={viewerMode} setMode={setViewerMode}
            walkerId={walkerId} setWalkerId={setWalkerId} inspected={inspected} loading={inspecting}
            prefixCount={prefixCount} />}
          isLoading={isLoading}
          hasData={!!trajectory}
        />
      </div>

      {/* WebGL canvas */}
      <div className="canvas-area">
        {error && <div className="error-banner">{error}</div>}

        <WalkerCanvas
          expanded={expanded}
          onToggleExpand={() => setExpanded(value => !value)}
          trajectory={displayedTrajectory}
          trajectories={displayedTrajectories}
          endpoints={endpoints}
          endpointCount={prefixCount}
          selectedWalkerId={inspected?.config.walker_id}
          onSelectWalker={setWalkerId}
          cloudOnly={viewerMode === 'cloud'}
          dimension={viewDimension}
          proj={viewProjection}
          steps={displayedData?.steps}
          nSteps={simData?.config.n_steps ?? nSteps}
          caption={overlayHint && (
          <div className="viewer-caption">
            <div className="canvas-hint viewer-source">{viewerMode === 'sample' ? 'Separate sample paths' : viewingEnsemble ? `Ensemble walker #${inspected.config.walker_id} | ${prefixCount.toLocaleString()} endpoint dots` : 'Loading ensemble walker...'}</div>
            {overlayHint.split('\n').map((line, i) => (
              <div key={i} className="canvas-hint">{line}</div>
            ))}
            <div className="canvas-hint" style={{ marginTop: 6, opacity: 0.8 }}>
              Axes: {viewDimension === 1 ? 'step t, x1' : viewProjection.slice(0, Math.min(3, viewDimension)).map((axis, slot) => (
                <span key={slot} style={{ color: ['#ef4444', '#10b981', '#3b82f6'][slot] }}>{slot > 0 ? ', ' : ''}x{axis + 1}</span>
              ))}
            </div>
          </div>
          )}
        />

        {(isLoading || inspecting) && (
          <div className="loading-overlay">
            <span className="loading-text">{inspecting ? `Replaying ensemble walker #${walkerId}...` : 'Computing simulation...'}</span>
          </div>
        )}

        {!trajectory && !isLoading && (
          <div className="canvas-empty">
            <span className="canvas-empty-title">No simulation running</span>
            <span className="canvas-empty-sub">Configure parameters and press Run Simulation</span>
          </div>
        )}


      </div>

      <PanelDivider width={actualPanelWidth} maxWidth={maxPanelWidth} onChange={setPanelWidth} onActiveChange={setResizingPanel} />

      {/* Right stats panel */}
      <StatsPanel panelSizing={{ width: actualPanelWidth, maximum: maxPanelWidth, onChange: setPanelWidth }} simData={viewingEnsemble ? inspected : simData} ensData={ensData}
        selectedWalkerId={viewingEnsemble ? inspected.config.walker_id : null}
        prefixCount={prefixCount} setPrefixCount={setPrefixCount} onShowCloud={() => { setViewerMode('cloud'); setMobileSection('viewer') }} />
    </div>
  )
}
