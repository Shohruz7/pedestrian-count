import { useEffect } from 'react'

/**
 * Custom hook for handling keyboard shortcuts
 * @param {Array} shortcuts - Array of shortcut objects with:
 *   - key: string (e.g., '1', 'ctrl+r', 'ctrl+/')
 *   - handler: function to execute when shortcut is pressed
 *   - description: string description (optional)
 */
export function useKeyboardShortcuts(shortcuts) {
  useEffect(() => {
    const handleKeyDown = (event) => {
      // Don't trigger shortcuts when user is typing in input fields
      const isInputFocused = 
        event.target.tagName === 'INPUT' ||
        event.target.tagName === 'TEXTAREA' ||
        event.target.isContentEditable

      if (isInputFocused && !event.ctrlKey && !event.metaKey) {
        return
      }

      // Check each shortcut
      for (const shortcut of shortcuts) {
        const { key, handler } = shortcut
        
        // Parse the key combination
        const parts = key.toLowerCase().split('+').map(s => s.trim())
        const hasCtrl = parts.includes('ctrl') || parts.includes('cmd')
        const hasShift = parts.includes('shift')
        const hasAlt = parts.includes('alt')
        const keyPart = parts.find(p => !['ctrl', 'cmd', 'shift', 'alt'].includes(p))

        // Check if modifiers match
        const ctrlMatch = hasCtrl ? (event.ctrlKey || event.metaKey) : !(event.ctrlKey || event.metaKey)
        const shiftMatch = hasShift ? event.shiftKey : !event.shiftKey
        const altMatch = hasAlt ? event.altKey : !event.altKey

        // Check if the key matches
        let keyMatch = false
        if (keyPart) {
          const normalizedKey = event.key.toLowerCase()
          if (keyPart === normalizedKey) {
            keyMatch = true
          } else if (keyPart === '/' && normalizedKey === '/') {
            keyMatch = true
          } else if (keyPart.length === 1 && normalizedKey === keyPart) {
            keyMatch = true
          }
        }

        // If all conditions match, execute the handler
        if (ctrlMatch && shiftMatch && altMatch && keyMatch) {
          event.preventDefault()
          handler(event)
          break // Only execute one shortcut per keypress
        }
      }
    }

    // Add event listener
    window.addEventListener('keydown', handleKeyDown)

    // Cleanup
    return () => {
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [shortcuts]) // Re-run if shortcuts change
}


