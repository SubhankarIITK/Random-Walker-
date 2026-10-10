import { useEffect, useRef } from 'react'

export default function PanelDivider({ width, maxWidth, onChange, onActiveChange }) {
  const drag = useRef(null)
  const callbacks = useRef({ maxWidth, onChange, onActiveChange })
  callbacks.current = { maxWidth, onChange, onActiveChange }
  const clamp = value => Math.max(260, Math.min(maxWidth, Math.round(value)))

  useEffect(() => {
    // Keep listening when the pointer leaves the narrow divider or crosses WebGL.
    const move = event => {
      if (drag.current?.id !== event.pointerId) return
      event.preventDefault()
      const { maxWidth, onChange } = callbacks.current
      const value = drag.current.width + drag.current.x - event.clientX
      onChange(Math.max(260, Math.min(maxWidth, Math.round(value))))
    }
    const stop = event => {
      if (!drag.current || (event.pointerId != null && drag.current.id !== event.pointerId)) return
      drag.current = null
      callbacks.current.onActiveChange(false)
    }
    window.addEventListener('pointermove', move, { capture: true, passive: false })
    window.addEventListener('pointerup', stop, true)
    window.addEventListener('pointercancel', stop, true)
    window.addEventListener('blur', stop)
    return () => {
      window.removeEventListener('pointermove', move, true)
      window.removeEventListener('pointerup', stop, true)
      window.removeEventListener('pointercancel', stop, true)
      window.removeEventListener('blur', stop)
    }
  }, [])

  return <div className="panel-divider" role="separator" tabIndex={0}
    aria-label="Resize statistics panel" aria-orientation="vertical"
    aria-valuemin={260} aria-valuemax={maxWidth} aria-valuenow={width}
    aria-valuetext={`${width} pixels wide`} aria-controls="statistics-panel"
    title="Drag left to widen statistics. Double-click to reset. Arrow keys also resize."
    onPointerDown={event => {
      if (event.button !== 0) return
      event.preventDefault()
      event.currentTarget.focus()
      const measuredWidth = event.currentTarget.nextElementSibling?.getBoundingClientRect().width || width
      drag.current = { id: event.pointerId, x: event.clientX, width: measuredWidth }
      event.currentTarget.setPointerCapture(event.pointerId)
      onActiveChange(true)
    }}
    onDoubleClick={() => onChange(Math.min(320, maxWidth))}
    onKeyDown={event => {
      const step = event.shiftKey ? 50 : 10
      let next
      if (event.key === 'ArrowLeft') next = width + step
      else if (event.key === 'ArrowRight') next = width - step
      else if (event.key === 'Home') next = 260
      else if (event.key === 'End') next = maxWidth
      else return
      event.preventDefault()
      onChange(clamp(next))
    }}>
    <span className="panel-divider-grip" aria-hidden="true" />
  </div>
}
