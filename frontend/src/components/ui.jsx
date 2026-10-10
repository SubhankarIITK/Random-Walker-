/** Small presentational building blocks shared by every view. */

export function Card({ title, subtitle, actions, children, className = '', id }) {
  return (
    <section className={`card ${className}`} id={id}>
      {(title || actions) && (
        <header className="card-head">
          <div>
            {title && <h3 className="card-title">{title}</h3>}
            {subtitle && <p className="card-sub">{subtitle}</p>}
          </div>
          {actions && <div className="card-actions">{actions}</div>}
        </header>
      )}
      <div className="card-body">{children}</div>
    </section>
  )
}

export function Metric({ label, value, unit, theory, note, tone, compact }) {
  return (
    <div className={`metric ${tone ? `tone-${tone}` : ''} ${compact ? 'compact' : ''}`}>
      <div className="metric-label">{label}</div>
      <div className="metric-value">
        {value}
        {unit && <span className="metric-unit">{unit}</span>}
      </div>
      {theory && <div className="metric-theory">Theory {theory}</div>}
      {note && <div className="metric-note">{note}</div>}
    </div>
  )
}

export function DataTable({ columns, rows, dense }) {
  return (
    <div className="table-wrap">
      <table className={`table ${dense ? 'dense' : ''}`}>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key} style={{ textAlign: c.align || 'right' }}>{c.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {columns.map((c) => (
                <td key={c.key} style={{ textAlign: c.align || 'right' }} className={c.mono === false ? '' : 'mono'}>
                  {c.render ? c.render(r) : r[c.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function Segmented({ options, value, onChange, disabled, size }) {
  return (
    <div className={`seg ${size === 'sm' ? 'seg-sm' : ''}`} role="tablist">
      {options.map((o) => {
        const v = typeof o === 'object' ? o.value : o
        const l = typeof o === 'object' ? o.label : o
        return (
          <button
            key={v}
            type="button"
            role="tab"
            aria-selected={value === v}
            className={`seg-btn ${value === v ? 'active' : ''}`}
            onClick={() => onChange(v)}
            disabled={disabled}
          >
            {l}
          </button>
        )
      })}
    </div>
  )
}

export function Slider({ id, label, value, min, max, step, onChange, format, disabled }) {
  return (
    <div className="field">
      <div className="field-row">
        <label className="field-label" htmlFor={id}>{label}</label>
        <span className="field-val">{format ? format(value) : value.toLocaleString()}</span>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ '--fill': `${((value - min) / (max - min)) * 100}%` }}
      />
    </div>
  )
}

export function Caveat({ children, tone = 'info' }) {
  return <div className={`caveat caveat-${tone}`}>{children}</div>
}

export function Badge({ children, tone = 'neutral' }) {
  return <span className={`badge badge-${tone}`}>{children}</span>
}

export function Empty({ title, children }) {
  return (
    <div className="empty">
      <div className="empty-title">{title}</div>
      {children && <div className="empty-sub">{children}</div>}
    </div>
  )
}

export function RunButton({ onClick, loading, children, id }) {
  return (
    <button id={id} type="button" className="btn-primary btn-inline" onClick={onClick} disabled={loading}>
      {loading ? <span className="spinner" /> : null}
      {loading ? 'Running' : children}
    </button>
  )
}

export function ViewHeader({ title, children, right }) {
  return (
    <div className="view-header">
      <div>
        <h2 className="view-title">{title}</h2>
        {children && <p className="view-desc">{children}</p>}
      </div>
      {right && <div className="view-header-right">{right}</div>}
    </div>
  )
}

export function Check({ ok, children }) {
  return (
    <span className={`check ${ok ? 'check-ok' : 'check-warn'}`}>
      <span className="check-dot" />
      {children}
    </span>
  )
}
