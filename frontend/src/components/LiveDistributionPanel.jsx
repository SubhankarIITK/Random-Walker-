import { useMemo, useState } from 'react'
import Plot from './Plot'
import EnsembleCountControls from './EnsembleCountControls'
import { prefixHistogram } from '../lib/ensemble'
import { coordName } from '../lib/format'

export default function LiveDistributionPanel({ data, count, setCount, onShowCloud }) {
  const [choice, setChoice] = useState('radial')
  const positions = data.walkers.final_positions
  const distributions = useMemo(() => [
    { key: 'radial', label: 'Radial distance |X|', axis: '|X|', template: data.radial.hist,
      theory: data.radial.theory, values: positions.map(p => Math.sqrt(p.reduce((sum, x) => sum + x*x, 0))) },
    ...data.coords.map(c => ({ key: String(c.index), label: `Coordinate ${coordName(c.index)}`,
      axis: coordName(c.index), template: c.hist, theory: c.theory, values: positions.map(p => p[c.index]) })),
  ], [data, positions])
  const effectiveChoice = choice === 'all' || distributions.some(d => d.key === choice) ? choice : 'radial'
  const selected = effectiveChoice === 'all' ? distributions : distributions.filter(d => d.key === effectiveChoice)
  const shown = selected.map(d => ({ ...d, hist: prefixHistogram(d.values, count, d.template) }))
  const exportBins = () => {
    const rows = ['Distribution,Walkers_included,Bin_left,Bin_right,Bin_center,Count,Density']
    for (const d of shown) d.hist.x.forEach((center, i) => rows.push(`${d.key},${count},${d.hist.edges[i]},${d.hist.edges[i+1]},${center},${d.hist.counts[i]},${d.hist.y[i]}`))
    const url = URL.createObjectURL(new Blob([rows.join('\n') + '\n'], {type: 'text/csv;charset=utf-8'}))
    const a = document.createElement('a')
    a.href = url
    a.download = `distribution_${effectiveChoice}_${count}_of_${positions.length}_walkers.csv`
    a.click()
    URL.revokeObjectURL(url)
  }
  return <div className="live-distributions">
    <p className="parameter-help">Watch the distributions build from the first 1, 2, ... N walkers of this completed ensemble, at fixed final time t = {data.config.n_steps.toLocaleString()}.</p>
    <EnsembleCountControls total={positions.length} count={count} setCount={setCount} runKey={positions} onShowCloud={onShowCloud} />
    <label className="distribution-choice"><span>Distribution</span><select aria-label="Distribution to animate" value={effectiveChoice} onChange={e => setChoice(e.target.value)}>
      {distributions.map(d => <option key={d.key} value={d.key}>{d.label}</option>)}
      <option value="all">All distributions</option>
    </select></label>
    <div className="distribution-progress" role="status">{count.toLocaleString()} / {positions.length.toLocaleString()} walkers included</div>
    <button className="btn" style={{width:'100%',marginBottom:12}} onClick={exportBins}>Export these bin counts (CSV)</button>
    {shown.map(d => {
      const h = d.hist
      const series = [
        {type:'bars', name:'Included walkers', x:h.x, y:h.y, width:h.width, color:'#3b82f6', opacity:.65},
        {type:'line', name:'Theory (approx.)', x:d.theory.x, y:d.theory.y, color:'#10b981', strokeWidth:2},
        {type:'bars', name:'Last walker bin', x:[h.x[h.lastBin]], y:[h.y[h.lastBin]], width:h.width, color:'#f59e0b', opacity:.85},
      ]
      return <div className="live-distribution-chart" key={d.key}>
        <div className="stat-label">{d.label}</div>
        <div className="stat-sub">Last included: walker #{count}, value {Number(h.lastValue.toPrecision(5))}. Mean: {Number(h.mean.toPrecision(5))}.</div>
        <Plot series={series} height={235} xLabel={d.axis} yLabel="Density" yZero xDomain={[h.edges[0],h.edges.at(-1)]} legend />
        <details className="distribution-bins"><summary>Check the bin counts ({h.counts.reduce((a,b) => a+b,0)} walkers)</summary>
          <div className="distribution-bin-scroll"><table><thead><tr><th>Bin center</th><th>Count</th><th>Density</th></tr></thead>
            <tbody>{h.x.map((center, i) => <tr key={i}><td>{Number(center.toPrecision(5))}</td><td>{h.counts[i]}</td><td>{Number(h.y[i].toPrecision(5))}</td></tr>)}</tbody>
          </table></div>
        </details>
      </div>
    })}
    <p className="parameter-help">Bins stay fixed throughout playback. Density = bin count / (included walkers x bin width), so the histogram area stays 1. Amber marks the bin of the last included walker. The green curve is the theoretical approximation at this final time.</p>
  </div>
}
