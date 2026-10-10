import { useEffect, useId, useMemo, useRef, useState } from 'react'
import { tick as fmtTick, val as fmtVal } from '../lib/format'

/**
 * Plot
 *
 * Lightweight SVG chart engine with always-visible, labelled axes.
 * Supports linear and log scales, and four series types:
 *   line   { x, y, color, dash?, strokeWidth? }
 *   band   { x, y0, y1, color }            shaded interval, e.g. 95% CI
 *   bars   { x, y, width, color }          histogram bars centred on x
 *   points { x, y, color, r? }
 * Hovering shows a crosshair and a tooltip with every series value.
 */

function useWidth(ref) {
  const [w, setW] = useState(0)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    setW(Math.floor(el.clientWidth))
    const ro = new ResizeObserver(([e]) => setW(Math.floor(e.contentRect.width)))
    ro.observe(el)
    return () => ro.disconnect()
  }, [ref])
  return w
}

function niceStep(range, target) {
  const raw = (range || 1) / Math.max(target, 1)
  const mag = 10 ** Math.floor(Math.log10(raw))
  const n = raw / mag
  return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * mag
}

function linTicks(lo, hi, target) {
  const step = niceStep(hi - lo, target)
  const out = []
  const start = Math.ceil(lo / step - 1e-9) * step
  for (let i = 0; i < 60; i++) {
    const v = start + i * step
    if (v > hi + step * 1e-6) break
    out.push(Math.abs(v) < step * 1e-9 ? 0 : +v.toPrecision(12))
  }
  return { ticks: out, step }
}

function logTicks(lo, hi) {
  const e0 = Math.floor(Math.log10(lo))
  const e1 = Math.ceil(Math.log10(hi))
  const dec = e1 - e0
  const mults = dec <= 1 ? [1, 2, 3, 5] : dec <= 3 ? [1, 2, 5] : [1]
  const out = []
  for (let e = e0; e <= e1; e++) {
    for (const m of mults) {
      const v = m * 10 ** e
      if (v >= lo * (1 - 1e-9) && v <= hi * (1 + 1e-9)) out.push(+v.toPrecision(12))
    }
  }
  return out
}

const ok = (v) => typeof v === 'number' && Number.isFinite(v)

function computeDomain(values, scale, forced, opts = {}) {
  let lo = Infinity
  let hi = -Infinity
  for (const v of values) {
    if (!ok(v) || (scale === 'log' && v <= 0)) continue
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  if (opts.includeZero && scale !== 'log') {
    lo = Math.min(lo, 0)
    hi = Math.max(hi, 0)
  }
  if (!Number.isFinite(lo)) {
    lo = scale === 'log' ? 1 : 0
    hi = scale === 'log' ? 10 : 1
  }
  if (forced?.[0] != null) lo = forced[0]
  if (forced?.[1] != null) hi = forced[1]

  if (scale === 'log') {
    if (lo === hi) { lo /= 2; hi *= 2 }
    return [forced?.[0] ?? lo / 1.12, forced?.[1] ?? hi * 1.12]
  }
  if (lo === hi) {
    const d = Math.abs(lo) || 1
    lo -= d * 0.5
    hi += d * 0.5
  }
  if (opts.nice === false) return [lo, hi]
  const step = niceStep(hi - lo, opts.target || 5)
  return [
    forced?.[0] ?? Math.floor(lo / step + 1e-9) * step,
    forced?.[1] ?? Math.ceil(hi / step - 1e-9) * step,
  ]
}

function nearestIndex(xs, x, log) {
  if (!xs?.length) return -1
  const tx = log ? Math.log(Math.max(x, 1e-300)) : x
  const g = (i) => (log ? Math.log(Math.max(xs[i], 1e-300)) : xs[i])
  let lo = 0
  let hi = xs.length - 1
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (g(mid) < tx) lo = mid
    else hi = mid
  }
  return Math.abs(g(lo) - tx) <= Math.abs(g(hi) - tx) ? lo : hi
}

export default function Plot({
  series = [],
  height = 240,
  xLabel,
  yLabel,
  xScale = 'linear',
  yScale = 'linear',
  xDomain,
  yDomain,
  yZero,
  xNice = true,
  xTicks: customXTicks,
  hlines = [],
  vlines = [],
  xFormat,
  yFormat,
  legend = true,
}) {
  const wrapRef = useRef(null)
  const width = useWidth(wrapRef)
  const clipId = useId().replace(/:/g, '')
  const [hover, setHover] = useState(null)

  const hasBars = series.some((s) => s.type === 'bars')

  const layout = useMemo(() => {
    const xs = []
    const ys = []
    for (const s of series) {
      const n = s.x?.length || 0
      for (let i = 0; i < n; i++) {
        const x = s.x[i]
        if (s.type === 'bars') {
          const w = (s.width || 0) / 2
          xs.push(x - w, x + w)
        } else xs.push(x)
        if (s.type === 'band') ys.push(s.y0[i], s.y1[i])
        else ys.push(s.y[i])
      }
    }
    for (const h of hlines) ys.push(h.y)
    for (const v of vlines) xs.push(v.x)

    const xd = computeDomain(xs, xScale, xDomain, { nice: xNice && !hasBars, target: 6 })
    const yd = computeDomain(ys, yScale, yDomain, { includeZero: yZero ?? hasBars, target: 5 })

    const yt = yScale === 'log' ? { ticks: logTicks(yd[0], yd[1]) } : linTicks(yd[0], yd[1], 5)
    const yFmt = yFormat || ((v) => fmtTick(v, yt.step))
    const yLabels = yt.ticks.map(yFmt)
    const maxLen = Math.max(1, ...yLabels.map((l) => String(l).length))

    const ml = (yLabel ? 20 : 4) + maxLen * 6.6 + 10
    const mr = 14
    const mt = 10
    const mb = xLabel ? 44 : 26
    return { xd, yd, yt, yLabels, ml, mr, mt, mb }
  }, [series, hlines, vlines, xScale, yScale, xDomain, yDomain, yZero, xNice, hasBars, yFormat, xLabel, yLabel])

  const { xd, yd, yt, yLabels, ml, mr, mt, mb } = layout
  const pw = Math.max(10, width - ml - mr)
  const ph = Math.max(10, height - mt - mb)

  const tx = (v) => (xScale === 'log' ? Math.log10(v) : v)
  const ty = (v) => (yScale === 'log' ? Math.log10(v) : v)
  const sx = (v) => ml + ((tx(v) - tx(xd[0])) / (tx(xd[1]) - tx(xd[0]) || 1)) * pw
  const sy = (v) => mt + ph - ((ty(v) - ty(yd[0])) / (ty(yd[1]) - ty(yd[0]) || 1)) * ph
  const invX = (px) => {
    const t = tx(xd[0]) + ((px - ml) / pw) * (tx(xd[1]) - tx(xd[0]))
    return xScale === 'log' ? 10 ** t : t
  }
  const valid = (x, y) => ok(x) && ok(y) && (xScale !== 'log' || x > 0) && (yScale !== 'log' || y > 0)

  // X ticks
  let xTickList
  if (customXTicks) {
    xTickList = customXTicks.map((t) => ({ v: t.v, label: t.label }))
  } else if (xScale === 'log') {
    xTickList = logTicks(xd[0], xd[1]).map((v) => ({ v, label: (xFormat || fmtTick)(v) }))
  } else {
    const target = Math.max(2, Math.min(8, Math.floor(pw / 70)))
    const lt = linTicks(xd[0], xd[1], target)
    xTickList = lt.ticks.map((v) => ({ v, label: xFormat ? xFormat(v) : fmtTick(v, lt.step) }))
  }

  const linePath = (s) => {
    let d = ''
    let pen = false
    for (let i = 0; i < s.x.length; i++) {
      const x = s.x[i]
      const y = s.y[i]
      if (!valid(x, y)) { pen = false; continue }
      d += `${pen ? 'L' : 'M'}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`
      pen = true
    }
    return d
  }

  const bandPath = (s) => {
    const top = []
    const bot = []
    for (let i = 0; i < s.x.length; i++) {
      if (!valid(s.x[i], s.y1[i]) || !valid(s.x[i], s.y0[i])) continue
      top.push(`${sx(s.x[i]).toFixed(1)},${sy(s.y1[i]).toFixed(1)}`)
      bot.push(`${sx(s.x[i]).toFixed(1)},${sy(s.y0[i]).toFixed(1)}`)
    }
    if (!top.length) return ''
    return `M${top.join('L')}L${bot.reverse().join('L')}Z`
  }

  const baseY = yScale === 'log' ? yd[0] : Math.max(yd[0], Math.min(0, yd[1]))

  const onMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const px = e.clientX - rect.left
    const py = e.clientY - rect.top
    if (px < ml || px > ml + pw || py < mt || py > mt + ph) { setHover(null); return }
    setHover({ px, py, x: invX(px) })
  }

  let tip = null
  if (hover) {
    const rows = []
    let anchorX = null
    for (const s of series) {
      if (s.tooltip === false || !s.x?.length) continue
      const i = nearestIndex(s.x, hover.x, xScale === 'log')
      if (i < 0) continue
      if (anchorX === null) anchorX = s.x[i]
      if (s.type === 'band') {
        rows.push({ name: s.name, color: s.color, value: `${fmtVal(s.y0[i])} – ${fmtVal(s.y1[i])}`, band: true })
      } else {
        rows.push({ name: s.name, color: s.color, value: fmtVal(s.y[i]), x: s.x[i], y: s.y[i], type: s.type })
      }
    }
    if (anchorX !== null) {
      const customLabel = customXTicks?.find((t) => t.v === anchorX)?.label
      tip = { anchorX, rows, header: `${xLabel || 'x'}: ${customLabel ?? fmtVal(anchorX)}` }
    }
  }

  const legendItems = series.filter((s) => s.name && s.legend !== false)

  return (
    <div className="plot">
      <div className="plot-canvas" ref={wrapRef} style={{ height }}>
        {width > 0 && (
          <svg
            width={width}
            height={height}
            onMouseMove={onMove}
            onMouseLeave={() => setHover(null)}
            role="img"
            aria-label={`${yLabel || 'y'} versus ${xLabel || 'x'}`}
          >
            <defs>
              <clipPath id={clipId}>
                <rect x={ml} y={mt} width={pw} height={ph} />
              </clipPath>
            </defs>

            {/* Plot background */}
            <rect x={ml} y={mt} width={pw} height={ph} className="plot-bg" />

            {/* Grid */}
            {yt.ticks.map((v, i) => (
              <line key={`gy${i}`} x1={ml} x2={ml + pw} y1={sy(v)} y2={sy(v)} className="plot-grid" />
            ))}
            {xTickList.map((t, i) => (
              <line key={`gx${i}`} y1={mt} y2={mt + ph} x1={sx(t.v)} x2={sx(t.v)} className="plot-grid" />
            ))}

            {/* Zero reference lines */}
            {yScale !== 'log' && yd[0] < 0 && yd[1] > 0 && (
              <line x1={ml} x2={ml + pw} y1={sy(0)} y2={sy(0)} className="plot-zero" />
            )}
            {xScale !== 'log' && xd[0] < 0 && xd[1] > 0 && (
              <line y1={mt} y2={mt + ph} x1={sx(0)} x2={sx(0)} className="plot-zero" />
            )}

            {/* Series */}
            <g clipPath={`url(#${clipId})`}>
              {series.map((s, si) => {
                if (!s.x?.length) return null
                if (s.type === 'band') {
                  return <path key={si} d={bandPath(s)} fill={s.color} opacity={s.opacity ?? 0.18} />
                }
                if (s.type === 'bars') {
                  const w = s.width || 1
                  return (
                    <g key={si}>
                      {s.x.map((x, i) => {
                        const y = s.y[i]
                        if (!ok(x) || !ok(y)) return null
                        const x0 = sx(x - w / 2)
                        const x1 = sx(x + w / 2)
                        const yTop = sy(Math.max(y, yScale === 'log' ? yd[0] : y))
                        const yB = sy(baseY)
                        const h = Math.max(0, yB - yTop)
                        return (
                          <rect
                            key={i}
                            x={x0 + 0.5}
                            y={yTop}
                            width={Math.max(0.5, x1 - x0 - 1)}
                            height={h}
                            fill={s.color}
                            fillOpacity={s.opacity ?? 0.5}
                            stroke={s.color}
                            strokeOpacity={0.9}
                            strokeWidth={0.75}
                          />
                        )
                      })}
                    </g>
                  )
                }
                if (s.type === 'points') {
                  return (
                    <g key={si}>
                      {s.x.map((x, i) =>
                        valid(x, s.y[i]) ? (
                          <circle key={i} cx={sx(x)} cy={sy(s.y[i])} r={s.r ?? 4} fill={s.color} stroke="#0b0d12" strokeWidth={1.5} />
                        ) : null
                      )}
                    </g>
                  )
                }
                return (
                  <path
                    key={si}
                    d={linePath(s)}
                    fill="none"
                    stroke={s.color}
                    strokeWidth={s.strokeWidth ?? 1.8}
                    strokeDasharray={s.dash ? '6 4' : undefined}
                    strokeLinejoin="round"
                    strokeLinecap="round"
                    opacity={s.opacity ?? 1}
                  />
                )
              })}

              {hlines.map((h, i) => (
                <line key={`h${i}`} x1={ml} x2={ml + pw} y1={sy(h.y)} y2={sy(h.y)} stroke={h.color || '#cbd5e1'} strokeDasharray={h.dash === false ? undefined : '3 4'} strokeWidth={1.2} />
              ))}
              {vlines.map((v, i) => (
                <line key={`v${i}`} y1={mt} y2={mt + ph} x1={sx(v.x)} x2={sx(v.x)} stroke={v.color || '#cbd5e1'} strokeDasharray={v.dash === false ? undefined : '3 4'} strokeWidth={1.2} />
              ))}
            </g>

            {hlines.map((h, i) => h.label && (
              <text key={`hl${i}`} x={ml + pw - 6} y={sy(h.y) - 5} textAnchor="end" className="plot-ref-label" fill={h.color || '#cbd5e1'}>{h.label}</text>
            ))}
            {vlines.map((v, i) => v.label && (
              <text key={`vl${i}`} x={sx(v.x) + 5} y={mt + 12} className="plot-ref-label" fill={v.color || '#cbd5e1'}>{v.label}</text>
            ))}

            {/* Axes */}
            <line x1={ml} x2={ml + pw} y1={mt + ph} y2={mt + ph} className="plot-axis" />
            <line x1={ml} x2={ml} y1={mt} y2={mt + ph} className="plot-axis" />

            {yt.ticks.map((v, i) => (
              <g key={`ty${i}`}>
                <line x1={ml - 4} x2={ml} y1={sy(v)} y2={sy(v)} className="plot-axis" />
                <text x={ml - 7} y={sy(v) + 3.5} textAnchor="end" className="plot-tick">{yLabels[i]}</text>
              </g>
            ))}
            {xTickList.map((t, i) => (
              <g key={`tx${i}`}>
                <line y1={mt + ph} y2={mt + ph + 4} x1={sx(t.v)} x2={sx(t.v)} className="plot-axis" />
                <text x={sx(t.v)} y={mt + ph + 16} textAnchor="middle" className="plot-tick">{t.label}</text>
              </g>
            ))}

            {xLabel && (
              <text x={ml + pw / 2} y={height - 6} textAnchor="middle" className="plot-axis-label">{xLabel}</text>
            )}
            {yLabel && (
              <text transform={`translate(13 ${mt + ph / 2}) rotate(-90)`} textAnchor="middle" className="plot-axis-label">{yLabel}</text>
            )}

            {/* Crosshair */}
            {tip && (
              <g pointerEvents="none">
                <line x1={sx(tip.anchorX)} x2={sx(tip.anchorX)} y1={mt} y2={mt + ph} className="plot-cross" />
                {tip.rows.filter((r) => !r.band && valid(r.x, r.y) && r.type !== 'bars').map((r, i) => (
                  <circle key={i} cx={sx(r.x)} cy={sy(r.y)} r={3.5} fill={r.color} stroke="#0b0d12" strokeWidth={1.5} />
                ))}
              </g>
            )}
          </svg>
        )}

        {tip && (
          <div
            className="plot-tip"
            style={{
              top: Math.max(4, Math.min(hover.py - 10, height - 40)),
              ...(hover.px > width * 0.6
                ? { right: width - hover.px + 14 }
                : { left: hover.px + 14 }),
            }}
          >
            <div className="plot-tip-head">{tip.header}</div>
            {tip.rows.map((r, i) => (
              <div key={i} className="plot-tip-row">
                <span className="plot-tip-dot" style={{ background: r.color }} />
                <span className="plot-tip-name">{r.name}</span>
                <span className="plot-tip-val">{r.value}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {legend && legendItems.length > 0 && (
        <div className="plot-legend">
          {legendItems.map((s, i) => (
            <span key={i} className="plot-legend-item">
              <LegendSwatch s={s} />
              {s.name}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

function LegendSwatch({ s }) {
  if (s.type === 'bars') return <span className="sw sw-bar" style={{ background: s.color }} />
  if (s.type === 'band') return <span className="sw sw-band" style={{ background: s.color }} />
  if (s.type === 'points') return <span className="sw sw-pt" style={{ background: s.color }} />
  return (
    <span
      className="sw sw-line"
      style={s.dash ? { borderTop: `2px dashed ${s.color}` } : { background: s.color }}
    />
  )
}
