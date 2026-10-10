import { useCallback, useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { Line2 } from 'three/addons/lines/Line2.js'
import { LineMaterial } from 'three/addons/lines/LineMaterial.js'
import { LineGeometry } from 'three/addons/lines/LineGeometry.js'
import { coordName } from '../lib/format'

/**
 * WalkerCanvas
 *
 * WebGL viewer for a single random-walk trajectory.
 *
 * World mapping (Three.js is Y-up):
 *   1D   X = step t (scaled), Y = x1            grid in the XY plane, front view
 *   2D   X = x_a, Z = -x_b                       grid in the XZ plane, top view
 *   3D+  X = x_a, Z = -x_b, Y = x_c              right-handed, x_c points up
 * For 4D/5D the user picks which three coordinates (a, b, c) are projected.
 *
 * Animation state lives in refs, so the render loop never triggers React renders.
 */

const AXIS_COLORS = ['#f87171', '#4ade80', '#60a5fa']
const PALETTE = [
  [0.231, 0.510, 0.965], // blue
  [0.024, 0.714, 0.831], // cyan
  [0.063, 0.725, 0.506], // emerald
  [0.961, 0.620, 0.043], // amber
  [0.957, 0.247, 0.369], // rose
]
export const PALETTE_CSS = 'linear-gradient(90deg, #3b82f6, #06b6d4, #10b981, #f59e0b, #f43f5e)'

function paletteAt(t) {
  const x = Math.min(1, Math.max(0, t)) * (PALETTE.length - 1)
  const i = Math.min(PALETTE.length - 2, Math.floor(x))
  const f = x - i
  const a = PALETTE[i]
  const b = PALETTE[i + 1]
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]
}

function niceStep(range, target) {
  const raw = (range || 1) / Math.max(target, 1)
  const mag = 10 ** Math.floor(Math.log10(raw))
  const n = raw / mag
  return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * mag
}

function fmtTick(v) {
  const a = Math.abs(v)
  if (a >= 10000) return `${+(v / 1000).toFixed(a >= 100000 ? 0 : 1)}k`
  return `${+v.toPrecision(6)}`
}

function makeLabel(text, color, worldH, weight = 500) {
  const fontPx = 44
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')
  const font = `${weight} ${fontPx}px Inter, system-ui, sans-serif`
  ctx.font = font
  const w = Math.ceil(ctx.measureText(text).width) + 24
  const h = fontPx + 20
  canvas.width = w
  canvas.height = h
  ctx.font = font
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.shadowColor = 'rgba(0,0,0,0.9)'
  ctx.shadowBlur = 10
  ctx.fillStyle = color
  ctx.fillText(text, w / 2, h / 2 + 1)
  const tex = new THREE.CanvasTexture(canvas)
  tex.colorSpace = THREE.SRGBColorSpace
  tex.anisotropy = 4
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false, depthWrite: false })
  const sp = new THREE.Sprite(mat)
  sp.scale.set((worldH * w) / h, worldH, 1)
  sp.renderOrder = 20
  return sp
}

// A compact round sprite keeps thousands of endpoints smooth and inexpensive.
function makePointTexture(selected = false) {
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = 64
  const ctx = canvas.getContext('2d')
  const radius = selected ? 17 : 30
  const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, radius)
  gradient.addColorStop(0, 'rgba(255,255,255,1)')
  gradient.addColorStop(0.28, 'rgba(255,255,255,0.95)')
  gradient.addColorStop(0.55, 'rgba(255,255,255,0.65)')
  gradient.addColorStop(0.82, 'rgba(255,255,255,0.16)')
  gradient.addColorStop(1, 'rgba(255,255,255,0)')
  ctx.fillStyle = gradient
  ctx.fillRect(0, 0, 64, 64)
  if (selected) {
    ctx.beginPath()
    ctx.arc(32, 32, 23, 0, Math.PI * 2)
    ctx.strokeStyle = 'rgba(255,255,255,0.95)'
    ctx.lineWidth = 3
    ctx.stroke()
  }
  const texture = new THREE.CanvasTexture(canvas)
  texture.colorSpace = THREE.SRGBColorSpace
  return texture
}

function disposeTree(obj) {
  obj.traverse((o) => {
    o.geometry?.dispose()
    if (o.material) {
      const mats = Array.isArray(o.material) ? o.material : [o.material]
      mats.forEach((m) => { m.map?.dispose(); m.dispose() })
    }
  })
}

const unit = (dir) => new THREE.Vector3(dir === 0 ? 1 : 0, dir === 1 ? 1 : 0, dir === 2 ? 1 : 0)

/** Grid lines on the plane spanned by world axes `ua`, `va`. */
function gridLines(ua, va, [u0, u1], [v0, v1], su, sv, color, opacity) {
  const pts = []
  const U = unit(ua)
  const V = unit(va)
  const push = (u, v) => { const p = U.clone().multiplyScalar(u).add(V.clone().multiplyScalar(v)); pts.push(p.x, p.y, p.z) }
  const nU = Math.round((u1 - u0) / su)
  const nV = Math.round((v1 - v0) / sv)
  if (nU > 400 || nV > 400) return null
  for (let i = 0; i <= nU; i++) { const u = u0 + i * su; push(u, v0); push(u, v1) }
  for (let j = 0; j <= nV; j++) { const v = v0 + j * sv; push(u0, v); push(u1, v) }
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3))
  const m = new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false })
  const ls = new THREE.LineSegments(g, m)
  ls.renderOrder = -1
  return ls
}

const SPEEDS = [0.25, 0.5, 1, 2, 4]

export default function WalkerCanvas({ trajectory, trajectories, steps, dimension, nSteps, proj, caption, expanded, onToggleExpand, endpoints, endpointCount, selectedWalkerId, onSelectWalker, cloudOnly = false, colorBy = 'time' }) {
  const mountRef = useRef(null)
  const threeRef = useRef(null)
  const dataRef = useRef(null)
  const animRef = useRef({ playing: false, count: 0, total: 0, speed: 1, acc: 0 })
  const prevTrajRef = useRef(null)
  const selectWalkerRef = useRef(onSelectWalker)
  selectWalkerRef.current = onSelectWalker
  const endpointCountRef = useRef(endpointCount)
  endpointCountRef.current = endpointCount

  const stepRef = useRef(null)
  const posRef = useRef(null)
  const rRef = useRef(null)
  const sqrtRef = useRef(null)
  const scrubRef = useRef(null)

  const [playing, setPlaying] = useState(false)
  const [speedMult, setSpeedMult] = useState(1)
  const [autoRotate, setAutoRotate] = useState(false)
  const [legend, setLegend] = useState(null)
  const [total, setTotal] = useState(0)

  const setPlayingRef = useRef(setPlaying)
  setPlayingRef.current = setPlaying

  // ── HUD + draw-range update (DOM writes only) ─────────────────────────
  const applyCount = useCallback((count) => {
    const d = dataRef.current
    if (!d) return
    const n = d.n
    const c = Math.max(1, Math.min(count, n))
    const i = c - 1
    d.geos.forEach(g => { g.instanceCount = d.cloudOnly ? 0 : Math.max(0, i) })
    
    // Update markers based on first trajectory
    d.cdot.position.set(d.poss[0][i * 3], d.poss[0][i * 3 + 1], d.poss[0][i * 3 + 2])
    d.halo.position.copy(d.cdot.position)
    d.edot.visible = c >= n && !d.cloudOnly
    const p = d.trajs[0][i]
    const t = d.steps ? d.steps[i] : i
    let r2 = 0
    for (const v of p) r2 += v * v
    if (stepRef.current) stepRef.current.textContent = t.toLocaleString()
    if (posRef.current) posRef.current.textContent = `(${p.join(', ')})`
    if (rRef.current) rRef.current.textContent = Math.sqrt(r2).toFixed(2)
    if (sqrtRef.current) sqrtRef.current.textContent = Math.sqrt(t).toFixed(2)
    if (scrubRef.current) {
      scrubRef.current.value = String(c)
      scrubRef.current.style.setProperty('--fill', `${(c / n) * 100}%`)
    }
  }, [])

  // ── Camera presets ─────────────────────────────────────────────────────
  const applyView = useCallback((name) => {
    const t = threeRef.current
    const d = dataRef.current
    if (!t || !d) return
    const { camera, controls } = t
    const v = name === 'fit' ? d.defaultView : name
    let direction
    if (v === 'top') direction = new THREE.Vector3(0, 1, 0.0015)
    else if (v === 'front') direction = new THREE.Vector3(0, 0, 1)
    else if (v === 'side') direction = new THREE.Vector3(1, 0, 0)
    else if (v === 'tilt') direction = new THREE.Vector3(0, 0.8, 0.62)
    else direction = new THREE.Vector3(0.62, 0.48, 0.78)
    direction.normalize()
    const right = new THREE.Vector3().crossVectors(camera.up, direction).normalize()
    const up = new THREE.Vector3().crossVectors(direction, right).normalize()
    const tanV = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))
    const tanH = tanV * Math.max(0.1, camera.aspect)
    let distance = 1
    for (const point of d.fitPoints) {
      const relative = point.clone().sub(d.center)
      const depth = relative.dot(direction)
      distance = Math.max(distance,
        Math.abs(relative.dot(right)) / tanH + depth,
        Math.abs(relative.dot(up)) / tanV + depth)
    }
    const D = distance * 1.12
    const off = direction.multiplyScalar(D)
    d.view = v
    controls.target.copy(d.center)
    camera.position.copy(d.center).add(off)
    camera.near = Math.max(0.01, D / 2000)
    camera.far = D * 40
    camera.updateProjectionMatrix()
    controls.update()
  }, [])

  // ── Scene setup (once) ─────────────────────────────────────────────────
  useEffect(() => {
    const el = mountRef.current
    if (!el) return

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, el.clientWidth / Math.max(1, el.clientHeight), 0.1, 100000)
    camera.position.set(60, 45, 75)

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(el.clientWidth, el.clientHeight)
    renderer.setClearColor(0x000000, 0)
    el.appendChild(renderer.domElement)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.autoRotateSpeed = 1.0

    let pointerStart = null
    const onPointerDown = e => { pointerStart = [e.clientX, e.clientY] }
    const onPointerUp = e => {
      if (!pointerStart || Math.hypot(e.clientX - pointerStart[0], e.clientY - pointerStart[1]) > 5) return
      const d = dataRef.current
      if (!d?.cloud) return
      const bounds = renderer.domElement.getBoundingClientRect()
      const mouse = new THREE.Vector2((e.clientX - bounds.left) / bounds.width * 2 - 1,
        -(e.clientY - bounds.top) / bounds.height * 2 + 1)
      const ray = new THREE.Raycaster()
      ray.params.Points.threshold = d.span * .013
      ray.setFromCamera(mouse, camera)
      const hits = ray.intersectObject(d.cloud).filter(hit => hit.index < (endpointCountRef.current ?? d.cloud.geometry.attributes.position.count))
      if (hits.length) selectWalkerRef.current?.(hits[0].index + 1)
    }
    renderer.domElement.addEventListener('pointerdown', onPointerDown)
    renderer.domElement.addEventListener('pointerup', onPointerUp)

    const lineMats = new Set()
    threeRef.current = { scene, camera, renderer, controls, lineMats }

    let raf
    const loop = () => {
      raf = requestAnimationFrame(loop)
      const a = animRef.current
      if (a.playing && a.count < a.total) {
        a.acc += a.speed
        const inc = Math.floor(a.acc)
        if (inc > 0) {
          a.acc -= inc
          a.count = Math.min(a.total, a.count + inc)
          applyCount(a.count)
        }
        if (a.count >= a.total) {
          a.playing = false
          setPlayingRef.current(false)
        }
      }
      controls.update()
      renderer.render(scene, camera)
    }
    loop()

    const ro = new ResizeObserver(() => {
      const w = el.clientWidth
      const h = Math.max(1, el.clientHeight)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
      lineMats.forEach((m) => m.resolution.set(w, h))
      if (dataRef.current) applyView(dataRef.current.view || 'fit')
    })
    ro.observe(el)

    return () => {
      renderer.domElement.removeEventListener('pointerdown', onPointerDown)
      renderer.domElement.removeEventListener('pointerup', onPointerUp)
      cancelAnimationFrame(raf)
      ro.disconnect()
      controls.dispose()
      disposeTree(scene)
      renderer.dispose()
      if (el.contains(renderer.domElement)) el.removeChild(renderer.domElement)
      threeRef.current = null
    }
  }, [applyCount, applyView])

  useEffect(() => {
    if (threeRef.current) threeRef.current.controls.autoRotate = autoRotate
  }, [autoRotate])

  // ── Build trajectory + decorations ─────────────────────────────────────
  useEffect(() => {
    const three = threeRef.current
    if (!three) return
    const { scene, lineMats, renderer, camera } = three
    const a = animRef.current

    const old = dataRef.current
    if (old) {
      scene.remove(old.root)
      lineMats.forEach((m) => { if (old.mats.includes(m)) lineMats.delete(m) })
      disposeTree(old.root)
      dataRef.current = null
    }

    if (!trajectories && !trajectory) {
      a.playing = false
      a.count = 0
      a.total = 0
      setTotal(0)
      setPlaying(false)
      setLegend(null)
      prevTrajRef.current = null
      return
    }

    const trajs = trajectories || [trajectory]
    if (!trajs[0] || trajs[0].length < 2) return

    const isNew = prevTrajRef.current !== trajs[0]
    prevTrajRef.current = trajs[0]

    const n = trajs[0].length
    const T = nSteps || (steps ? steps[n - 1] : n - 1)
    const [pa, pb, pc] = proj || [0, 1, 2]

    // Extents
    let maxAbs = 1
    const usedIdx = dimension === 1 ? [0] : dimension === 2 ? [pa, pb] : [pa, pb, pc]
    for (const path of trajs) for (const p of path) for (const k of usedIdx) maxAbs = Math.max(maxAbs, Math.abs(p[k]))
    if (endpoints) for (const p of endpoints) for (const k of usedIdx) maxAbs = Math.max(maxAbs, Math.abs(p[k]))
    const step = niceStep(maxAbs * 1.05, 4)
    const half = Math.ceil((maxAbs * 1.04) / step) * step
    const timeLen = 2.6 * half
    const tScale = timeLen / Math.max(1, T)

    const map = (p, i) => {
      if (dimension === 1) return [(steps ? steps[i] : i) * tScale, p[0], 0]
      if (dimension === 2) return [p[pa], 0, -p[pb]]
      return [p[pa], p[pc], -p[pb]]
    }

    // Colour source
    let cMin = 0
    let cMax = T
    const colorIdx = colorBy === 'time' ? null : Number(colorBy)
    if (colorIdx !== null) {
      cMin = Infinity
      cMax = -Infinity
      for (const p of trajectory) { cMin = Math.min(cMin, p[colorIdx]); cMax = Math.max(cMax, p[colorIdx]) }
      if (cMin === cMax) cMax = cMin + 1
    }

    const root = new THREE.Group()
    const mats = []
    const geos = []
    const poss = []
    const w = renderer.domElement.clientWidth
    const h = renderer.domElement.clientHeight

    trajs.forEach((tData, tIdx) => {
      const pos = new Float32Array(n * 3)
      const col = new Float32Array(n * 3)
      for (let i = 0; i < n; i++) {
        const [x, y, z] = map(tData[i], i)
        pos[i * 3] = x
        pos[i * 3 + 1] = y
        pos[i * 3 + 2] = z
        const tv = colorIdx === null
          ? (steps ? steps[i] : i) / Math.max(1, T)
          : (tData[i][colorIdx] - cMin) / (cMax - cMin)
        const c = paletteAt(tv)
        // Differentiate walkers slightly by dimming secondary ones
        const m = tIdx === 0 ? 1 : 0.6
        col[i * 3] = c[0] * m
        col[i * 3 + 1] = c[1] * m
        col[i * 3 + 2] = c[2] * m
      }

      const geo = new LineGeometry()
      geo.setPositions(pos)
      geo.setColors(col)
      const lmat = new LineMaterial({ linewidth: tIdx === 0 ? 2.2 : 1.2, vertexColors: true, worldUnits: false })
      lmat.resolution.set(w, h)
      mats.push(lmat)
      const line = new Line2(geo, lmat)
      line.computeLineDistances()
      line.renderOrder = tIdx === 0 ? 2 : 1
      root.add(line)
      geos.push(geo)
      poss.push(pos)
    })

    let cloud = null
    let selectedDot = null
    const cloudPoints = []
    if (endpoints) {
      const positions = new Float32Array(endpoints.length * 3)
      endpoints.forEach((point, i) => {
        const world = map(point, n - 1)
        positions.set(world, i * 3)
        cloudPoints.push(new THREE.Vector3(...world))
      })
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
      geometry.setDrawRange(0, endpointCountRef.current ?? endpoints.length)
      cloud = new THREE.Points(geometry, new THREE.PointsMaterial({
        map: makePointTexture(), color: '#7dd3fc',
        size: endpoints.length > 2000 ? 7 : 9, sizeAttenuation: false,
        transparent: true, opacity: .92, alphaTest: .015, depthWrite: false,
      }))
      cloud.renderOrder = 3
      root.add(cloud)
    }

    // Decorations
    const span = dimension === 1 ? Math.max(timeLen, 2 * half) : 2 * half
    const lh = span * 0.026
    const dotR = span * 0.0065

    const axisLine = (from, to, color) => {
      const g = new LineGeometry()
      g.setPositions([...from, ...to])
      const m = new LineMaterial({ color: new THREE.Color(color), linewidth: 1.6, worldUnits: false, transparent: true, opacity: 0.95 })
      m.resolution.set(w, h)
      mats.push(m)
      const l = new Line2(g, m)
      l.computeLineDistances()
      l.renderOrder = 1
      root.add(l)
    }

    const tickMarks = []
    const addTick = (p, q) => tickMarks.push(...p, ...q)
    const label = (text, color, at, size = lh, weight = 500) => {
      const s = makeLabel(text, color, size, weight)
      s.position.set(at[0], at[1], at[2])
      root.add(s)
    }

    const MAJOR = 0x5a6578
    const MINOR = 0x3a4252
    const tickColor = '#cbd5e1'

    if (dimension === 1) {
      const tStep = niceStep(T, 5)
      const tMax = Math.ceil(T / tStep) * tStep
      const g1 = gridLines(0, 1, [0, tMax * tScale], [-half, half], tStep * tScale, step, MAJOR, 0.85)
      const g2 = gridLines(0, 1, [0, tMax * tScale], [-half, half], (tStep * tScale) / 5, step / 5, MINOR, 0.55)
      g1 && root.add(g1)
      g2 && root.add(g2)
      axisLine([0, 0, 0], [tMax * tScale * 1.03, 0, 0], AXIS_COLORS[0])
      axisLine([0, -half, 0], [0, half * 1.06, 0], AXIS_COLORS[1])
      for (let t = tStep; t <= tMax + 1e-9; t += tStep) {
        label(fmtTick(t), tickColor, [t * tScale, -half - lh * 1.1, 0])
        addTick([t * tScale, -half, 0], [t * tScale, -half - lh * 0.4, 0])
      }
      for (let v = -half; v <= half + 1e-9; v += step) {
        label(fmtTick(v), tickColor, [-lh * 1.8, v, 0])
        addTick([0, v, 0], [-lh * 0.4, v, 0])
      }
      label('step t', AXIS_COLORS[0], [tMax * tScale * 1.03 + lh * 2.2, 0, 0], lh * 1.3, 600)
      label(coordName(0), AXIS_COLORS[1], [0, half * 1.06 + lh * 1.2, 0], lh * 1.4, 600)
    } else {
      const g1 = gridLines(0, 2, [-half, half], [-half, half], step, step, MAJOR, 0.85)
      const g2 = gridLines(0, 2, [-half, half], [-half, half], step / 5, step / 5, MINOR, 0.55)
      g1 && root.add(g1)
      g2 && root.add(g2)
      const ext = half * 1.08
      axisLine([-half, 0, 0], [ext, 0, 0], AXIS_COLORS[0])
      axisLine([0, 0, half], [0, 0, -ext], AXIS_COLORS[1])
      if (dimension >= 3) axisLine([0, -half, 0], [0, ext, 0], AXIS_COLORS[2])

      for (let v = -half; v <= half + 1e-9; v += step) {
        if (Math.abs(v) < 1e-9) continue
        // x_a along +X, labels just below the axis (towards +Z)
        label(fmtTick(v), tickColor, [v, 0, lh * 1.3])
        addTick([v, 0, 0], [v, 0, lh * 0.45])
        // x_b along -Z, labels to the left of the axis
        label(fmtTick(v), tickColor, [-lh * 1.9, 0, -v])
        addTick([0, 0, -v], [-lh * 0.45, 0, -v])
        if (dimension >= 3) {
          label(fmtTick(v), tickColor, [-lh * 1.9, v, 0])
          addTick([0, v, 0], [-lh * 0.45, v, 0])
        }
      }
      label('0', tickColor, [-lh * 1.1, 0, lh * 1.1])
      label(coordName(pa), AXIS_COLORS[0], [ext + lh * 1.6, 0, 0], lh * 1.45, 600)
      label(coordName(pb), AXIS_COLORS[1], [0, 0, -ext - lh * 1.6], lh * 1.45, 600)
      if (dimension >= 3) label(coordName(pc), AXIS_COLORS[2], [0, ext + lh * 1.4, 0], lh * 1.45, 600)
    }

    if (tickMarks.length) {
      const tg = new THREE.BufferGeometry()
      tg.setAttribute('position', new THREE.Float32BufferAttribute(tickMarks, 3))
      root.add(new THREE.LineSegments(tg, new THREE.LineBasicMaterial({ color: 0xa3abb9 })))
    }

    // Markers
    const sphere = (r, color, opacity = 1) => new THREE.Mesh(
      new THREE.SphereGeometry(r, 20, 20),
      new THREE.MeshBasicMaterial({ color, transparent: opacity < 1, opacity, depthWrite: opacity >= 1 })
    )
    const origin = sphere(dotR * 0.8, 0xe5e7eb, 0.55)
    root.add(origin)
    const sdot = sphere(dotR * 1.1, 0x10b981)
    sdot.position.set(poss[0][0], poss[0][1], poss[0][2])
    root.add(sdot)
    const edot = sphere(dotR * 1.25, 0xf43f5e)
    edot.position.set(poss[0][(n - 1) * 3], poss[0][(n - 1) * 3 + 1], poss[0][(n - 1) * 3 + 2])
    edot.visible = false
    root.add(edot)
    const cdot = sphere(dotR * 1.3, 0xffffff)
    cdot.renderOrder = 5
    root.add(cdot)
    const halo = sphere(dotR * 3.2, 0xffffff, 0.16)
    halo.renderOrder = 4
    root.add(halo)

    if (cloudOnly) {
      sdot.visible = false
      cdot.visible = false
      halo.visible = false
    }
    if (endpoints && selectedWalkerId && endpoints[selectedWalkerId - 1]) {
      const selectedGeometry = new THREE.BufferGeometry()
      selectedGeometry.setAttribute('position', new THREE.Float32BufferAttribute(
        map(endpoints[selectedWalkerId - 1], n - 1), 3))
      selectedDot = new THREE.Points(selectedGeometry, new THREE.PointsMaterial({
        map: makePointTexture(true), color: '#fbbf24', size: 24,
        sizeAttenuation: false, transparent: true, alphaTest: .015,
        depthWrite: false, depthTest: false,
      }))
      selectedDot.renderOrder = 10
      selectedDot.visible = selectedWalkerId <= (endpointCountRef.current ?? endpoints.length)
      root.add(selectedDot)
    }

    scene.add(root)
    mats.forEach((m) => lineMats.add(m))

    const center = dimension === 1 ? new THREE.Vector3(timeLen / 2, 0, 0) : new THREE.Vector3(0, 0, 0)
    const defaultView = dimension === 1 ? 'front' : dimension === 2 ? 'top' : 'iso'
    // Fit the actual projected paths and visible grid, using the current viewport.
    const fitPoints = [...cloudPoints]
    for (const positions of poss) {
      for (let i = 0; i < positions.length; i += 3) {
        fitPoints.push(new THREE.Vector3(positions[i], positions[i + 1], positions[i + 2]))
      }
    }
    if (dimension === 1) {
      for (const x of [0, timeLen]) for (const y of [-half, half]) fitPoints.push(new THREE.Vector3(x, y, 0))
    } else {
      for (const x of [-half, half]) for (const z of [-half, half]) fitPoints.push(new THREE.Vector3(x, 0, z))
      if (dimension >= 3) fitPoints.push(new THREE.Vector3(0, half * 1.12, 0))
    }
    dataRef.current = { root, mats, geos, poss, n, trajs, steps, cdot, halo, edot, center, span, defaultView, fitPoints, cloud, selectedDot, selectedWalkerId, cloudOnly }
    camera.aspect = renderer.domElement.clientWidth / Math.max(1, renderer.domElement.clientHeight)


    setLegend(colorIdx === null
      ? { label: 'step t', min: 0, max: T }
      : { label: coordName(colorIdx), min: cMin, max: cMax })

    a.total = n
    setTotal(n)
    if (isNew) {
      applyView('fit')
      a.count = 1
      a.acc = 0
      a.playing = true
      setPlaying(true)
    } else {
      applyView(old?.view || 'fit')
      a.count = Math.min(Math.max(1, a.count), n)
    }
    if (cloudOnly) { a.count = n; a.playing = false; setPlaying(false) }
    a.speed = (speedMult * n) / 480
    applyCount(a.count)
  }, [trajectory, trajectories, steps, dimension, nSteps, proj, colorBy, endpoints, selectedWalkerId, cloudOnly, applyCount, applyView]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const data = dataRef.current
    if (data?.cloud) {
      data.cloud.geometry.setDrawRange(0, endpointCount)
      if (data.selectedDot) data.selectedDot.visible = data.selectedWalkerId <= endpointCount
    }
  }, [endpointCount])

  // ── Transport ──────────────────────────────────────────────────────────
  useEffect(() => {
    animRef.current.speed = (speedMult * Math.max(1, animRef.current.total)) / 480
  }, [speedMult])

  const togglePlay = () => {
    const a = animRef.current
    if (!a.total) return
    if (a.playing) {
      a.playing = false
      setPlaying(false)
      return
    }
    if (a.count >= a.total) { a.count = 1; applyCount(1) }
    a.playing = true
    setPlaying(true)
  }
  const restart = () => {
    const a = animRef.current
    a.count = 1
    a.acc = 0
    applyCount(1)
    a.playing = true
    setPlaying(true)
  }
  const toEnd = () => {
    const a = animRef.current
    a.count = a.total
    a.playing = false
    setPlaying(false)
    applyCount(a.total)
  }
  const onScrub = (e) => {
    const a = animRef.current
    a.count = Number(e.target.value)
    a.playing = false
    setPlaying(false)
    applyCount(a.count)
  }

  const viewButtons = dimension === 1
    ? []
    : dimension === 2
      ? [['top', 'Top'], ['tilt', 'Tilt']]
      : [['iso', 'Iso'], ['top', 'Top'], ['front', 'Front'], ['side', 'Side']]

  return (
    <div className="viewer">
      <div ref={mountRef} className="viewer-gl" />

      {total > 0 && (
        <>
          <div className="hud" aria-live="off">
            <div className="hud-row">
              <span className="hud-k">Step</span>
              <span className="hud-v"><span ref={stepRef}>0</span><span className="hud-dim"> / {(nSteps || 0).toLocaleString()}</span></span>
            </div>
            <div className="hud-row">
              <span className="hud-k">Position</span>
              <span className="hud-v hud-pos" ref={posRef}>—</span>
            </div>
            <div className="hud-row">
              <span className="hud-k">|X|</span>
              <span className="hud-v" ref={rRef}>—</span>
            </div>
            <div className="hud-row">
              <span className="hud-k">√t</span>
              <span className="hud-v hud-muted" ref={sqrtRef}>—</span>
            </div>
          </div>

          <div className="view-tools">
            <button type="button" className="chip" onClick={() => applyView('fit')}>Fit</button>
            <button type="button" className="chip" onClick={onToggleExpand}>{expanded ? 'Show panels' : 'Expand view'}</button>
            {viewButtons.map(([k, l]) => (
              <button key={k} type="button" className="chip" onClick={() => applyView(k)}>{l}</button>
            ))}
            {dimension >= 2 && (
              <button type="button" className={`chip ${autoRotate ? 'active' : ''}`} onClick={() => setAutoRotate((v) => !v)}>
                Rotate
              </button>
            )}
          </div>

          {caption}

          {legend && (
            <div className="color-legend">
              {endpoints && <div className="color-legend-label">Blue: ensemble endpoints<br />Amber: selected walker</div>}
              {!cloudOnly && <>
              <div className="color-legend-label">Colour: {legend.label}</div>
              <div className="color-legend-bar" style={{ background: PALETTE_CSS }} />
              <div className="color-legend-ticks">
                <span>{fmtTick(legend.min)}</span>
                <span>{fmtTick(legend.max)}</span>
              </div>
              <div className="marker-legend">
                <span><i style={{ background: '#10b981' }} />start</span>
                <span><i style={{ background: '#ffffff' }} />walker</span>
                <span><i style={{ background: '#f43f5e' }} />end</span>
              </div>
              </>}
            </div>
          )}

          {!cloudOnly && <div className="transport">
            <button type="button" className="icon-btn" onClick={togglePlay} aria-label={playing ? 'Pause' : 'Play'} id="btn-play">
              {playing ? (
                <svg viewBox="0 0 16 16" width="14" height="14"><rect x="3" y="2" width="3.5" height="12" rx="1" fill="currentColor" /><rect x="9.5" y="2" width="3.5" height="12" rx="1" fill="currentColor" /></svg>
              ) : (
                <svg viewBox="0 0 16 16" width="14" height="14"><path d="M4 2.5v11a.8.8 0 0 0 1.2.7l9-5.5a.8.8 0 0 0 0-1.4l-9-5.5A.8.8 0 0 0 4 2.5z" fill="currentColor" /></svg>
              )}
            </button>
            <button type="button" className="icon-btn" onClick={restart} aria-label="Restart" id="btn-restart">
              <svg viewBox="0 0 16 16" width="14" height="14"><path d="M3 8a5 5 0 1 0 1.6-3.7" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" /><path d="M2.5 2.2v3.6h3.6" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>
            </button>
            <button type="button" className="icon-btn" onClick={toEnd} aria-label="Show complete path" id="btn-end">
              <svg viewBox="0 0 16 16" width="14" height="14"><path d="M3 2.8v10.4a.7.7 0 0 0 1.1.6L11 8.6a.7.7 0 0 0 0-1.2L4.1 2.2a.7.7 0 0 0-1.1.6z" fill="currentColor" /><rect x="11.5" y="2.5" width="2" height="11" rx="0.8" fill="currentColor" /></svg>
            </button>
            <input
              ref={scrubRef}
              className="scrub"
              type="range"
              min={1}
              max={total}
              defaultValue={1}
              onInput={onScrub}
              aria-label="Animation position"
            />
            <div className="speed">
              {SPEEDS.map((s) => (
                <button key={s} type="button" className={`speed-btn ${speedMult === s ? 'active' : ''}`} onClick={() => setSpeedMult(s)}>
                  {s}x
                </button>
              ))}
            </div>
          </div>}
        </>
      )}
    </div>
  )
}
