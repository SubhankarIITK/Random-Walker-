import { test } from 'node:test'
import assert from 'node:assert/strict'
import { build } from 'esbuild'

// Run the actual event handlers with a small hook shim, without WebGL.
async function loadHandlers(file, window, cleanups) {
  const compiled = await build({ entryPoints: [file], bundle: true, write: false,
    format: 'cjs', platform: 'node', jsx: 'automatic', plugins: [{ name: 'event-test-hooks', setup(api) {
      api.onResolve({filter: /^react(\/jsx-runtime)?$/}, args => ({path: args.path, namespace:'test-hooks'}))
      api.onLoad({filter: /.*/, namespace: 'test-hooks'}, args => ({contents: args.path === 'react'
        ? 'export const useRef = value => ({current:value}); export const useEffect = callback => effectCleanups.push(callback());'
        : 'export const jsx = (type,props) => ({type,props}); export const jsxs = jsx;'}))
    }}] })
  const module = {exports:{}}
  new Function('module', 'window', 'effectCleanups', compiled.outputFiles[0].text)(module, window, cleanups)
  return module.exports.default
}

test('divider follows window pointer moves, clamps widths and stops on release', async () => {
  const listeners = new Map(), cleanups = []
  const window = {addEventListener(type, handler){listeners.set(type,handler)}, removeEventListener(type){listeners.delete(type)}}
  const Divider = await loadHandlers('src/components/PanelDivider.jsx',window,cleanups)
  let width = 320, active = false
  const element = Divider({width:320,maxWidth:720,onChange:value=>{width=value},onActiveChange:value=>{active=value}})
  const event = x => ({pointerId:7,clientX:x,preventDefault(){}})
  element.props.onPointerDown({...event(800),button:0,currentTarget:{focus(){},setPointerCapture(){},nextElementSibling:{getBoundingClientRect:()=>({width:320})}}})
  assert.equal(active,true)
  listeners.get('pointermove')(event(650))
  assert.equal(width,470)
  listeners.get('pointermove')({...event(300),pointerId:99})
  assert.equal(width,470)
  listeners.get('pointermove')(event(0))
  assert.equal(width,720)
  listeners.get('pointermove')(event(2000))
  assert.equal(width,260)
  listeners.get('pointerup')(event(2000))
  assert.equal(active,false)
  listeners.get('pointermove')(event(0))
  assert.equal(width,260)
  element.props.onKeyDown({key:'ArrowLeft',preventDefault(){}})
  assert.equal(width,330)
  element.props.onDoubleClick()
  assert.equal(width,320)
  cleanups.forEach(cleanup=>cleanup())
  assert.equal(listeners.size,0)
})

test('visible panel width slider passes an exact numeric width and resets', async () => {
  const SizeControl = await loadHandlers('src/components/PanelSizeControl.jsx',{},[])
  let width = 320
  const element = SizeControl({sizing:{width:320,maximum:720,onChange:value=>{width=value}}})
  const row = element.props.children[1]
  const [slider, reset] = row.props.children
  slider.props.onChange({target:{value:'515'}})
  assert.equal(width,515)
  reset.props.onClick()
  assert.equal(width,320)
})
