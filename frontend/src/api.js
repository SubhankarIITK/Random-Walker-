const BASE = 'http://localhost:8000'

async function post(path, body, signal) {
  let res
  try {
    res = await fetch(`${BASE}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    })
  } catch (err) {
    if (err.name === 'AbortError') throw err
    throw new Error('BACKEND_UNREACHABLE')
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: `Server error ${res.status}` }))
    const detail = Array.isArray(err.detail) ? err.detail.map((d) => d.msg).join('; ') : err.detail
    throw new Error(detail || `Request failed (${res.status})`)
  }
  return res.json()
}

export async function health() {
  try {
    const r = await fetch(`${BASE}/health`)
    return r.ok
  } catch {
    return false
  }
}

export const simulate     = (b, s) => post('/simulate', b, s)
export const getEnsemble  = (b, s) => post('/ensemble', b, s)
export const compareDims  = (b, s) => post('/compare', b, s)
export const convergence  = (b, s) => post('/convergence', b, s)
export const recurrence   = (b, s) => post('/recurrence', b, s)
export const sites        = (b, s) => post('/sites', b, s)
export const firstPassage = (b, s) => post('/first_passage', b, s)
export const biased       = (b, s) => post('/biased', b, s)

export const BACKEND_HINT =
  'Cannot reach the backend on port 8000. From the project root run: python -m uvicorn backend.main:app --reload --port 8000'

export function errorMessage(err) {
  return err?.message === 'BACKEND_UNREACHABLE' ? BACKEND_HINT : err?.message || 'Unknown error'
}

export async function exportEnsemble(config) {
  const res = await fetch(`${BASE}/ensemble/export`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  })
  if (!res.ok) throw new Error(`Ensemble export failed (${res.status})`)
  return res.blob()
}

export const inspectWalker = (body, signal) => post('/ensemble/walker', body, signal)
