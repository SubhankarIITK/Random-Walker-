import { useEffect, useId, useState } from 'react'

export default function EnsembleCountControls({ total, count, setCount, runKey, onShowCloud }) {
  const [playing, setPlaying] = useState(false)
  const [pace, setPace] = useState('fast')
  const inputId = useId()
  useEffect(() => { setPlaying(false) }, [runKey])
  useEffect(() => {
    if (!playing) return
    const timer = setInterval(() => setCount(previous => Math.min(total,
      previous + (pace === 'single' ? 1 : Math.max(1, Math.ceil(total / 150))))), 80)
    return () => clearInterval(timer)
  }, [playing, total, setCount, pace])
  useEffect(() => { if (count >= total) setPlaying(false) }, [count, total])
  return <>
    <div className="field-row"><label htmlFor={inputId} className="field-label">Walkers included</label>
      <input id={inputId} className="exact-count" type="number" min={1} max={total} value={count}
        onFocus={e => e.target.select()} onChange={e => {
          const n = Number(e.target.value)
          if (Number.isInteger(n) && n >= 1 && n <= total) { setPlaying(false); setCount(n) }
        }} />
    </div>
    <input type="range" min={1} max={total} step={1} value={count} aria-label="Included ensemble size"
      onChange={e => { setPlaying(false); setCount(Number(e.target.value)) }} />
    <div className="mode-switch convergence-actions">
      <button className="btn" disabled={total < 2} onClick={() => { if (count >= total) setCount(1); setPlaying(value => !value) }}>{playing ? 'Pause' : 'Play'}</button>
      <button className="btn" onClick={() => { setPlaying(false); setCount(1) }}>Reset</button>
      <button className="btn" onClick={() => { setPlaying(false); setCount(total) }}>All {total}</button>
    </div>
    <label className="distribution-choice"><span>Playback</span><select value={pace} onChange={e => setPace(e.target.value)}>
      <option value="fast">Fast</option><option value="single">One walker at a time</option>
    </select></label>
    <button className="btn" style={{width:'100%',marginTop:8}} onClick={onShowCloud}>Show these endpoints in viewer</button>
  </>
}
