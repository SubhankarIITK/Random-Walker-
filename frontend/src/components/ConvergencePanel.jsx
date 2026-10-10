import { useMemo } from 'react'
import Plot from './Plot'
import EnsembleCountControls from './EnsembleCountControls'
import { prefixMSD } from '../lib/ensemble'

export default function ConvergencePanel({ data, count, setCount, onShowCloud }) {
  const positions = data.walkers.final_positions
  const N = positions.length
  const theory = data.config.n_steps
  const rows = useMemo(() => prefixMSD(positions, theory), [positions, theory])
  const shown = rows.slice(0, count)
  const current = rows[count - 1]
  if (!current) return null
  const x = shown.map(r => r.count)
  const band = shown.filter(r => r.se !== null)
  const series = [
    { type: 'band', name: 'Approx. 95% CI', x: band.map(r => r.count), y0: band.map(r => r.lo), y1: band.map(r => r.hi), color: '#3b82f6', opacity: .2 },
    { type: 'line', name: 'Cumulative MSD', x, y: shown.map(r => r.msd), color: '#60a5fa' },
    { type: 'points', name: 'Current estimate', x: [count], y: [current.msd], color: '#f59e0b', legend: false },
  ]
  return <div className="convergence-panel">
    <p className="parameter-help">Add members of the same completed ensemble in walker-ID order. No new simulation is generated.</p>
    <EnsembleCountControls total={N} count={count} setCount={setCount} runKey={positions} onShowCloud={onShowCloud} />
    <div className="convergence-readout">
      <div className="stat-label">MSD from first {count.toLocaleString()} walkers</div>
      <div className="stat-value">{current.msd.toFixed(2)}</div>
      <div className="stat-sub">Theory: {theory.toLocaleString()} | Error: {current.errorPct.toFixed(2)}%</div>
      <div className="stat-sub">Standard error: {current.se === null ? 'Requires at least 2 walkers' : current.se.toFixed(2)}</div>
    </div>
    <Plot series={series} height={230} xLabel="Walkers included (N)" yLabel="MSD" xDomain={[1, Math.max(2, N)]}
      hlines={[{y: theory, color:'#10b981', dash:true, label:'Theory'}]} legend />
    <Plot series={[{type:'line',name:'Absolute relative error',x,y:shown.map(r=>r.errorPct),color:'#f59e0b'}]}
      height={180} xLabel="Walkers included (N)" yLabel="Error (%)" xDomain={[1, Math.max(2, N)]} yZero legend />
    <p className="parameter-help">The band is mean +/- 1.96 x sample standard error: an approximate pointwise interval, less reliable for small ensembles. Error can rise between additions; more walkers improve precision on average.</p>
  </div>
}
