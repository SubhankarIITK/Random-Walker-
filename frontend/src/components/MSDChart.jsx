/**
 * Minimal SVG line chart for MSD vs time.
 * No external library — builds path strings directly.
 */
export default function MSDChart({ data }) {
  if (!data?.time_steps?.length) return null

  const { time_steps, msd_sim, msd_theory } = data

  const W = 212, H = 120
  const pad = { t: 8, r: 8, b: 26, l: 36 }
  const pw  = W - pad.l - pad.r
  const ph  = H - pad.t - pad.b

  const xMin = time_steps[0] ?? 0
  const xMax = time_steps.at(-1) ?? 1
  const yMax = Math.max(...msd_sim, ...msd_theory) * 1.06

  const sx = (v) => ((v - xMin) / (xMax - xMin || 1)) * pw
  const sy = (v) => ph - (v / (yMax || 1)) * ph

  const makePath = (xs, ys) => {
    const pts = xs.map((x, i) => `${sx(x).toFixed(1)},${sy(ys[i]).toFixed(1)}`)
    return `M ${pts.join(' L ')}`
  }

  const yTicks = [0, 0.5, 1].map((t) => ({ y: sy(yMax * t), val: Math.round(yMax * t) }))

  return (
    <div className="chart-wrap">
      <div className="chart-label">MSD vs Time</div>

      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ overflow: 'visible', display: 'block' }}>
        <g transform={`translate(${pad.l},${pad.t})`}>
          {/* Horizontal grid lines */}
          {yTicks.map((tk, i) => (
            <g key={i}>
              <line
                x1={0} y1={tk.y} x2={pw} y2={tk.y}
                stroke="rgba(255,255,255,0.04)" strokeWidth={0.5}
              />
              {i > 0 && (
                <text
                  x={-4} y={tk.y + 3}
                  textAnchor="end"
                  fill="rgba(255,255,255,0.18)"
                  fontSize={7}
                  fontFamily="var(--mono)"
                >
                  {tk.val.toLocaleString()}
                </text>
              )}
            </g>
          ))}

          {/* Theory */}
          <path
            d={makePath(time_steps, msd_theory)}
            stroke="rgba(255,255,255,0.2)"
            strokeWidth={1}
            strokeDasharray="4 3"
            fill="none"
          />

          {/* Simulation */}
          <path
            d={makePath(time_steps, msd_sim)}
            stroke="#3b82f6"
            strokeWidth={1.5}
            fill="none"
          />

          {/* Axes */}
          <line x1={0} y1={ph} x2={pw} y2={ph} stroke="rgba(255,255,255,0.07)" strokeWidth={0.5} />
          <line x1={0} y1={0}  x2={0}  y2={ph} stroke="rgba(255,255,255,0.07)" strokeWidth={0.5} />

          {/* X labels */}
          <text x={0} y={ph + 14} fill="rgba(255,255,255,0.18)" fontSize={7} fontFamily="var(--mono)">
            0
          </text>
          <text x={pw} y={ph + 14} textAnchor="end" fill="rgba(255,255,255,0.18)" fontSize={7} fontFamily="var(--mono)">
            {xMax.toLocaleString()}
          </text>
          <text x={pw / 2} y={ph + 22} textAnchor="middle" fill="rgba(255,255,255,0.12)" fontSize={7}>
            steps
          </text>
        </g>
      </svg>

      <div className="chart-legend">
        <span className="legend-item">
          <span style={{ display: 'inline-block', width: 14, height: 1.5, background: '#3b82f6', verticalAlign: 'middle' }} />
          Simulation
        </span>
        <span className="legend-item">
          <span style={{ display: 'inline-block', width: 14, borderTop: '1px dashed rgba(255,255,255,0.22)', verticalAlign: 'middle' }} />
          Theory E[R²]=t
        </span>
      </div>
    </div>
  )
}
