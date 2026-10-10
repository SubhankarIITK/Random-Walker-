export default function Controls({
  dimension, setDimension,
  viewDimension, projection, setProjection,
  nSteps, setNSteps,
  nWalkers, setNWalkers,
  nVisualWalkers, setNVisualWalkers,
  seed, setSeed,
  onRun, onExport, onExportAll,
  isExporting, ensembleConfig, inspector,
  isLoading, hasData,
}) {
  const fmt = (n) => n.toLocaleString()

  return (
    <>
      {/* Dimension */}
      <div className="section">
        <div className="section-label">Dimension</div>
        <div className="dim-row">
          {[1, 2, 3, 4, 5].map((d) => (
            <button
              key={d}
              className={`dim-btn ${dimension === d ? 'active' : ''}`}
              onClick={() => setDimension(d)}
              disabled={isLoading}
            >
              {d}D
            </button>
          ))}
        </div>
        {dimension >= 4 && (
          <p style={{ fontSize: 10.5, color: 'var(--text-3)', marginTop: 8, lineHeight: 1.6 }}>
            The walk uses all coordinates. Select any three below for the 3D projection.
          </p>
        )}
      </div>

      {viewDimension >= 4 && (
        <div className="section">
          <div className="section-label">Projection coordinates</div>
          {['Red axis', 'Green axis', 'Blue axis'].map((label, slot) => (
            <label className="projection-field" key={slot}>
              <span>{label}</span>
              <select aria-label={label + ' coordinate'} value={projection[slot]}
                onChange={(e) => {
                  const next = [...projection]
                  const selected = Number(e.target.value)
                  const other = next.indexOf(selected)
                  if (other !== -1) next[other] = next[slot]
                  next[slot] = selected
                  setProjection(next)
                }}>
                {Array.from({ length: viewDimension }, (_, i) => (
                  <option value={i} key={i}>x{i + 1}</option>
                ))}
              </select>
            </label>
          ))}
          <p className="parameter-help">Changes the current view instantly. Choosing an occupied coordinate swaps axes. Hidden coordinates still count in |X| and statistics.</p>
        </div>
      )}

      {/* Parameters */}
      <div className="section">
        <div className="section-label">Parameters</div>

        <div className="field">
          <div className="field-row">
            <span className="field-label">Steps</span>
            <input className="exact-count" type="number" aria-label="Steps exact value" min={1} max={20000} step={1} value={nSteps}
              onFocus={(e) => e.target.select()}
              onChange={(e) => {
                const value = Number(e.target.value)
                if (Number.isInteger(value) && value >= 1 && value <= 20000) setNSteps(value)
              }} disabled={isLoading} />
          </div>
          <input
            type="range"
            min={1}
            max={20000}
            step={1}
            value={nSteps}
            onChange={(e) => setNSteps(Number(e.target.value))}
            disabled={isLoading}
          />
        </div>

        <div className="field">
          <div className="field-row">
            <span className="field-label">Ensemble walkers</span>
            <input className="exact-count" type="number" aria-label="Ensemble walkers exact value" min={1} max={5000} step={1} value={nWalkers}
              onFocus={(e) => e.target.select()}
              onChange={(e) => {
                const value = Number(e.target.value)
                if (Number.isInteger(value) && value >= 1 && value <= 5000) setNWalkers(value)
              }} disabled={isLoading} />
          </div>
          <input
            type="range"
            min={1}
            max={5000}
            step={1}
            value={nWalkers}
            onChange={(e) => setNWalkers(Number(e.target.value))}
            disabled={isLoading}
          />
        </div>

        <div className="field">
          <div className="field-row">
            <span className="field-label">Displayed walkers</span>
            <input className="exact-count" type="number" aria-label="Visual walkers exact value" min={1} max={20} step={1} value={nVisualWalkers}
              onFocus={(e) => e.target.select()}
              onChange={(e) => {
                const value = Number(e.target.value)
                if (Number.isInteger(value) && value >= 1 && value <= 20) setNVisualWalkers(value)
              }} disabled={isLoading} />
          </div>
          <input
            type="range"
            min={1}
            max={20}
            step={1}
            value={nVisualWalkers}
            onChange={(e) => setNVisualWalkers(Number(e.target.value))}
            disabled={isLoading}
          />
        </div>

        <p className="parameter-help">Statistics use the ensemble. The viewer shows separate sample paths.</p>

        <div className="field">
          <div className="field-row">
            <span className="field-label">Seed</span>
            <span className="field-val" style={{ color: 'var(--text-3)' }}>
              {seed || 'random'}
            </span>
          </div>
          <input
            type="number"
            placeholder="leave empty for random"
            value={seed}
            onChange={(e) => setSeed(e.target.value)}
            disabled={isLoading}
          />
        </div>
      </div>

      {inspector}

      {/* Run */}
      <div className="section">
        <button className="btn-primary" onClick={onRun} disabled={isLoading} style={{ marginBottom: hasData ? 8 : 0 }}>
          {isLoading ? 'Computing...' : 'Run Simulation'}
        </button>
        {hasData && (
          <button className="btn" style={{ width: '100%' }} onClick={onExport} disabled={isLoading}>
            Export run evidence (ZIP)
          </button>
        )}
      </div>

      {hasData && ensembleConfig && (
        <div className="section">
          <div className="section-label">Ensemble evidence</div>
          <p className="parameter-help run-evidence">Completed run: {ensembleConfig.n_walkers.toLocaleString()} walkers x {ensembleConfig.n_steps.toLocaleString()} steps = {(ensembleConfig.n_walkers * ensembleConfig.n_steps).toLocaleString()} walker updates. Seed: {ensembleConfig.seed}.</p>
          <button className="btn" style={{ width: '100%' }} onClick={onExportAll} disabled={isLoading || isExporting}>
            {isExporting ? 'Preparing all paths...' : 'Export all ensemble paths'}
          </button>
          <p className="parameter-help">Every step of every ensemble walker, replayed exactly from this run's seed. Large runs can take time. The evidence ZIP includes one endpoint per walker and an MSD cross-check.</p>
        </div>
      )}

      {/* Theory reference */}
      <div className="section" style={{ marginTop: 'auto' }}>
        <div className="section-label">Theory</div>
        <div className="theory-ref">
          <div>E[R²] = n for all d</div>
          <div>Var(xᵢ) = n / d</div>
          <div>RMS ~ √n, α = 0.5</div>
        </div>
      </div>
    </>
  )
}
