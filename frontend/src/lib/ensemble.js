// Prefix statistics of one fixed ensemble: walker 1, then 1..2, through 1..N.
export function prefixMSD(positions, theory) {
  let mean = 0, m2 = 0
  return positions.map((position, index) => {
    const count = index + 1
    const r2 = position.reduce((sum, x) => sum + x * x, 0)
    const delta = r2 - mean
    mean += delta / count
    m2 += delta * (r2 - mean)
    const se = count > 1 ? Math.sqrt(Math.max(0, m2) / (count - 1) / count) : null
    return { count, r2, msd: mean, se,
      lo: se === null ? null : Math.max(0, mean - 1.96 * se),
      hi: se === null ? null : mean + 1.96 * se,
      errorPct: Math.abs(mean - theory) / theory * 100 }
  })
}

// Fixed bins from the completed run make prefixes directly comparable.
export function prefixHistogram(values, count, template) {
  const width = template.width
  const x = template.x
  const edges = template.edges ?? [x[0] - width / 2, ...x.map(center => center + width / 2)]
  const counts = Array(x.length).fill(0)
  const included = Math.max(0, Math.min(values.length, Math.trunc(count)))
  let mean = 0, lastBin = null
  for (let i = 0; i < included; i++) {
    const value = values[i]
    if (!Number.isFinite(value) || value < edges[0] || value > edges.at(-1)) {
      throw new Error('An ensemble value falls outside the histogram bins')
    }
    // NumPy convention: left-closed bins, with the final right edge included.
    let lo = 0, hi = edges.length - 1
    while (hi - lo > 1) {
      const middle = (lo + hi) >> 1
      if (value < edges[middle]) hi = middle
      else lo = middle
    }
    const bin = Math.min(lo, x.length - 1)
    counts[bin]++
    mean += (value - mean) / (i + 1)
    lastBin = bin
  }
  return { x, edges, width, counts, included, mean: included ? mean : null,
    y: counts.map(n => included ? n / (included * width) : 0),
    lastBin, lastValue: included ? values[included - 1] : null }
}
