/** Number formatting helpers shared across the UI. */

const isNum = (v) => typeof v === 'number' && Number.isFinite(v)

export const f = (v, d = 3) => (isNum(v) ? v.toFixed(d) : '—')

export const fi = (v) => (isNum(v) ? Math.round(v).toLocaleString() : '—')

export const pct = (v, d = 2) => (isNum(v) ? `${v.toFixed(d)}%` : '—')

/** Signed formatting, e.g. +0.0123 */
export const fs = (v, d = 3) => (isNum(v) ? `${v >= 0 ? '+' : ''}${v.toFixed(d)}` : '—')

/** Compact axis tick formatting. */
export function tick(v, step) {
  if (!isNum(v)) return ''
  const a = Math.abs(v)
  if (a === 0) return '0'
  if (a >= 1e6) return `${+(v / 1e6).toFixed(a >= 1e7 ? 0 : 1)}M`
  if (a >= 1e4) return `${+(v / 1e3).toFixed(a >= 1e5 ? 0 : 1)}k`
  if (step !== undefined && step > 0) {
    const dec = Math.max(0, Math.min(6, -Math.floor(Math.log10(step) + 1e-9)))
    return v.toFixed(dec)
  }
  if (a >= 100) return v.toFixed(0)
  if (a >= 1) return +v.toFixed(2) + ''
  return +v.toPrecision(2) + ''
}

/** Value formatting for tooltips. */
export function val(v) {
  if (!isNum(v)) return '—'
  const a = Math.abs(v)
  if (a === 0) return '0'
  if (a >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 1 })
  if (a >= 1) return v.toFixed(3)
  return v.toPrecision(3)
}

export const SUB = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉']
export const coordName = (i) => `x${SUB[i + 1]}`

export function downloadFile(name, content, type) {
  const blob = new Blob([content], { type })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = name
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}
