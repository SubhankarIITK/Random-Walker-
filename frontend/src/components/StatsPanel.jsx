import { useState } from 'react'
import Plot from './Plot'
import PanelSizeControl from './PanelSizeControl'
import ConvergencePanel from './ConvergencePanel'
import LiveDistributionPanel from './LiveDistributionPanel'

function MSDPlot({ series }) {
  if (!series || !series.t || !series.t.length) return null

  const plotSeries = [
    {
      type: 'band',
      name: '95% CI',
      x: series.t,
      y0: series.msd_lo,
      y1: series.msd_hi,
      color: '#3b82f6',
      opacity: 0.2,
      legend: false
    },
    {
      type: 'line',
      name: 'Theory E[R²] = t',
      x: series.t,
      y: series.msd_theory,
      color: '#10b981',
      dash: true,
      strokeWidth: 2
    },
    {
      type: 'line',
      name: 'Simulation MSD',
      x: series.t,
      y: series.msd,
      color: '#3b82f6',
      strokeWidth: 2
    }
  ]

  return (
    <div style={{ padding: '14px 18px', borderTop: '1px solid var(--border)', marginTop: 'auto' }}>
      <div className="stat-label" style={{ marginBottom: 12 }}>Mean Squared Displacement</div>
      <Plot
        series={plotSeries}
        height={180}
        xLabel="Steps (t)"
        yLabel="MSD R²"
        xScale="log"
        yScale="log"
        legend={true}
      />
    </div>
  )
}

export default function StatsPanel({ simData, ensData, selectedWalkerId, prefixCount, setPrefixCount, onShowCloud, panelSizing }) {
  const [tab, setTab] = useState('summary')

  if (!simData && !ensData) {
    return (
      <div className="stats-panel" id="statistics-panel">
      <PanelSizeControl sizing={panelSizing} />
        <div className="stats-idle">
          <p>Run a simulation to view statistics.</p>
        </div>
      </div>
    )
  }

  const f  = (n, d = 3) => (typeof n === 'number' ? n.toFixed(d) : '—')
  const fi = (n)         => (typeof n === 'number' ? Math.round(n).toLocaleString() : '—')

  const summary = ensData?.summary
  const rec     = ensData?.returns
  const scaling = ensData?.scaling
  const series  = ensData?.series

  const walk    = simData?.walk
  const returns = simData?.returns
  const sites   = simData?.sites

  return (
    <div className="stats-panel" id="statistics-panel">
      <PanelSizeControl sizing={panelSizing} />
      {/* Panel header & tabs */}
      <div style={{ padding: '16px 20px 0', borderBottom: '1px solid var(--border)', flexShrink: 0 }}>
        <div className="section-label" style={{ marginBottom: 12 }}>Statistical Laboratory</div>
        <div className="stats-tabs" role="tablist" aria-label="Statistics view">
          {[['summary', 'Summary'], ['distributions', 'Distributions'], ['convergence', 'Convergence']].map(([key, label]) =>
            <button key={key} role="tab" aria-selected={tab === key} className={tab === key ? 'active' : ''} onClick={() => setTab(key)}>{label}</button>)}
        </div>
      </div>
      {tab === 'convergence' && ensData && <ConvergencePanel key={ensData.config.seed} data={ensData}
        count={prefixCount} setCount={setPrefixCount} onShowCloud={onShowCloud} />}

      {tab === 'summary' && (
        <>
          <div className="summary-source">{selectedWalkerId ? `Inspecting ensemble walker #${selectedWalkerId}` : 'First visual sample walker'}<br />Ensemble statistics use all {ensData?.config.n_walkers.toLocaleString()} walkers.</div>
          {/* Single-walk stats */}
          {simData && walk && (
            <>
              <div className="stat-item">
                <div className="stat-label">Distance from origin |X|</div>
                <div className="stat-value">{f(walk.final_r, 2)}</div>
                <div className="stat-sub">Theory RMS ≈ {f(walk.rms_theory, 2)}</div>
              </div>

              <div className="stat-item">
                <div className="stat-label">Squared displacement |X|²</div>
                <div className="stat-value">{fi(walk.final_r2)}</div>
                <div className="stat-sub">Theory E[R²] = {fi(walk.msd_theory)}</div>
              </div>

              {sites && (
                <div className="stat-item">
                  <div className="stat-label">Unique Sites Visited</div>
                  <div className="stat-value">{fi(sites.n_distinct_sites)}</div>
                  <div className="stat-sub">Coverage: {f(sites.coverage_fraction, 4)}</div>
                </div>
              )}

              {returns && (
                <div className="stat-item">
                  <div className="stat-label">Origin Returns (Single Walk)</div>
                  <div className="stat-value">{fi(returns.n_returns)} returns</div>
                  <div className="stat-sub">
                    {returns.n_returns > 0 ? `First return at step ${fi(returns.first_return_time)}` : "Did not return"}
                  </div>
                </div>
              )}
            </>
          )}

          {/* Ensemble stats */}
          {summary && (
            <>
              <div className="stat-item" style={{ background: 'rgba(59, 130, 246, 0.05)' }}>
                <div className="stat-label">Ensemble MSD ({ensData.config.n_walkers} walkers)</div>
                <div className="stat-value" style={{ color: '#60a5fa' }}>{f(summary.msd_sim, 2)}</div>
                <div className="stat-sub">Theory = {fi(summary.msd_theory)} (Error: {f(summary.pct_error_msd, 2)}%)</div>
              </div>

              {scaling && (
                <div className="stat-item">
                  <div className="stat-label">Scaling Exponent (Fitted α)</div>
                  <div className="stat-value">{f(scaling.alpha_rms, 4)}</div>
                  <div className="stat-sub">Theoretical α = 0.5000 (RMS ~ t^α)</div>
                </div>
              )}
            </>
          )}

          {/* Recurrence */}
          {rec && (
            <>
              <div className="stat-item">
                <div className="stat-label">Recurrence class</div>
                <div>
                  <span className={`tag ${rec.is_recurrent ? 'recurrent' : 'transient'}`}>
                    {rec.classification}
                  </span>
                </div>
                <div className="stat-sub" style={{ marginTop: 8 }}>
                  {rec.is_recurrent
                    ? 'Returns to origin with probability 1'
                    : `Return prob ≈ ${typeof rec.theory_prob === 'number' ? rec.theory_prob.toFixed(4) : rec.theory_prob}`
                  }
                </div>
              </div>

              <div className="stat-item">
                <div className="stat-label">Fraction returned</div>
                <div className="stat-value">{f(rec.fraction_returned, 4)}</div>
                <div className="stat-sub">
                  Observed across {ensData.config.n_walkers?.toLocaleString()} walkers
                </div>
              </div>
            </>
          )}

          {/* MSD chart at bottom */}
          {series && <MSDPlot series={series} />}
        </>
      )}

      {tab === 'distributions' && ensData && <LiveDistributionPanel key={ensData.config.seed} data={ensData}
        count={prefixCount} setCount={setPrefixCount} onShowCloud={onShowCloud} />}

    </div>
  )
}
