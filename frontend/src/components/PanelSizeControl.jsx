export default function PanelSizeControl({ sizing }) {
  if (!sizing) return null
  const { width, maximum, onChange } = sizing
  return <div className="panel-size-control">
    <label htmlFor="statistics-width">Panel width <span>{width}px</span></label>
    <div className="panel-size-row">
      <input id="statistics-width" type="range" min={260} max={maximum} step={1} value={width}
        aria-label="Statistics panel width" onChange={event => onChange(Number(event.target.value))} />
      <button type="button" className="chip" onClick={() => onChange(Math.min(320, maximum))}>Reset</button>
    </div>
  </div>
}
