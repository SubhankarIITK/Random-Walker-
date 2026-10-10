export default function EnsembleInspector({ data, mode, setMode, walkerId, setWalkerId, inspected, loading, prefixCount }) {
  if (!data) return null
  const N = data.config.n_walkers
  const id = inspected?.config.walker_id
  const endpoint = id ? data.walkers.final_positions[id - 1] : null
  const r2 = endpoint?.reduce((sum, x) => sum + x * x, 0)
  const verified = endpoint && endpoint.every((x, i) => x === inspected.trajectory.at(-1)[i])
  return <div className="section ensemble-inspector">
    <div className="section-label">Inspect the ensemble</div>
    <div className="mode-switch" role="group" aria-label="Viewer data source">
      {[['sample', 'Samples'], ['ensemble', 'Walker'], ['cloud', 'Endpoints']].map(([key, label]) =>
        <button key={key} className={`btn ${mode === key ? 'active' : ''}`} onClick={() => setMode(key)}>{label}</button>)}
    </div>
    <label className="field-row inspector-id"><span className="field-label">Ensemble walker #</span>
      <input className="exact-count" type="number" min={1} max={N} value={walkerId} aria-label="Ensemble walker ID"
        onFocus={e => e.target.select()} onChange={e => {
          const id = Number(e.target.value)
          if (Number.isInteger(id) && id >= 1 && id <= N) { setWalkerId(id); if (mode === 'sample') setMode('ensemble') }
        }} />
    </label>
    <input type="range" min={1} max={N} step={1} value={walkerId} aria-label="Select ensemble walker"
      onChange={e => { setWalkerId(Number(e.target.value)); if (mode === 'sample') setMode('ensemble') }} />
    {mode === 'sample' ? <p className="parameter-help">Separate sample paths. Switch to Walker to inspect a real member of this completed ensemble.</p>
      : loading ? <p className="parameter-help" role="status">Replaying walker #{walkerId}...</p>
      : endpoint && <>
        <p className="parameter-help">Walker #{id}: ({endpoint.join(', ')}). {verified ? 'Endpoint matches the ensemble record.' : 'Endpoint mismatch.'}</p>
        <div className="inspector-metric"><span>Squared distance</span><strong>{r2.toLocaleString()}</strong></div>
        <div className="inspector-metric"><span>Contribution to full MSD</span><strong>{(r2 / N).toFixed(4)}</strong></div>
        <p className="parameter-help">MSD = sum of all squared distances / {N.toLocaleString()}. {id > prefixCount ? 'This walker is outside the current convergence prefix.' : 'Included in the current convergence prefix.'}</p>
      </>}
    {mode !== 'sample' && <p className="parameter-help">Blue dots: endpoints of the first {prefixCount.toLocaleString()} walkers. Amber: selected endpoint. Click a dot to inspect it. Overlapping endpoints may share a dot.</p>}
  </div>
}
