import { test } from 'node:test'
import assert from 'node:assert/strict'
import { prefixMSD, prefixHistogram } from './ensemble.js'

test('prefix means and uncertainty follow individual squared distances', () => {
  const rows = prefixMSD([[1], [3], [5]], 10)
  assert.deepEqual(rows.map(r => r.count), [1, 2, 3])
  assert.equal(rows[0].msd, 1)
  assert.equal(rows[0].se, null)
  assert.equal(rows[0].lo, null)
  assert.equal(rows[1].msd, 5)
  assert.equal(rows[1].se, 4)
  assert.ok(Math.abs(rows[2].msd - 35 / 3) < 1e-12)
})

test('all dimensions contribute and constant samples have zero spread', () => {
  const rows = prefixMSD([[1,2,3,4,5],[-1,-2,-3,-4,-5]], 55)
  assert.equal(rows[1].msd, 55)
  assert.equal(rows[1].se, 0)
  assert.equal(rows[1].errorPct, 0)
})

test('every prefix agrees with direct sample mean and sample SE', () => {
  const positions = Array.from({length: 137}, (_, i) => [i % 11 - 5, i % 7, i % 5 - 2])
  const rows = prefixMSD(positions, 40)
  rows.forEach((row, i) => {
    const values = positions.slice(0, i+1).map(p => p.reduce((sum, x) => sum + x*x, 0))
    const mean = values.reduce((a,b) => a+b,0)/values.length
    assert.ok(Math.abs(mean - row.msd) < 1e-12)
    if (i) {
      const variance = values.reduce((a,b) => a + (b-mean)**2,0)/i
      assert.ok(Math.abs(Math.sqrt(variance/values.length) - row.se) < 1e-12)
    }
  })
})


test('histogram builds counts from the selected prefix and stays normalized', () => {
  const template = { x: [-1, 0, 1], width: 1, edges: [-1.5, -.5, .5, 1.5] }
  const values = [-1, 1, 1, 0]
  for (let count = 1; count <= values.length; count++) {
    const hist = prefixHistogram(values, count, template)
    assert.equal(hist.counts.reduce((a,b) => a+b,0), count)
    assert.ok(Math.abs(hist.y.reduce((a,b) => a+b*hist.width,0) - 1) < 1e-12)
    assert.equal(hist.lastValue, values[count - 1])
  }
  assert.deepEqual(prefixHistogram(values, 2, template).counts, [1, 0, 1])
  assert.deepEqual(prefixHistogram(values, 4, template).counts, [1, 1, 2])
})

test('histogram uses left-closed bins and includes the last right edge', () => {
  const hist = prefixHistogram([0, 1, 2], 3, { x: [.5, 1.5], width: 1, edges: [0, 1, 2] })
  assert.deepEqual(hist.counts, [1, 2])
  assert.equal(hist.lastBin, 1)
})

test('histogram fallback preserves old run bins and zero prefix is empty', () => {
  const template = {x: [-2, 0, 2], width: 2}
  assert.deepEqual(prefixHistogram([-2, 2, 0], 3, template).counts, [1, 1, 1])
  const empty = prefixHistogram([-2], 0, template)
  assert.deepEqual(empty.y, [0, 0, 0])
  assert.equal(empty.lastValue, null)
  assert.equal(empty.mean, null)
})
