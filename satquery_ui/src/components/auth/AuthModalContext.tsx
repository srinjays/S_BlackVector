/**
 * AuthModalContext — Shared state for the on-demand Auth Modal
 *
 * Provides:
 * - isOpen: boolean
 * - mode: 'login' | 'signup'
 * - openAuthModal: (mode?: 'login' | 'signup') => void
 * - closeAuthModal: () => void
 * - setMode: (mode: 'login' | 'signup') => void
 *
 * Handles:
 * - Body scroll lock while modal is open
 * - Focus capture and restoration to the trigger element
 * - Keyboard 'Escape' key listener
 */

import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react'

export type AuthModalMode = 'login' | 'signup'

interface AuthModalContextType {
  isOpen: boolean
  mode: AuthModalMode
  openAuthModal: (mode?: AuthModalMode) => void
  closeAuthModal: () => void
  setMode: (mode: AuthModalMode) => void
}

const AuthModalContext = createContext<AuthModalContextType | undefined>(undefined)

export const AuthModalProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isOpen, setIsOpen] = useState(false)
  const [mode, setMode] = useState<AuthModalMode>('login')
  const previousActiveElementRef = useRef<HTMLElement | null>(null)

  const openAuthModal = useCallback((initialMode: AuthModalMode = 'login') => {
    // Store current active element to restore focus later
    if (document.activeElement instanceof HTMLElement) {
      previousActiveElementRef.current = document.activeElement
    }
    setMode(initialMode)
    setIsOpen(true)
  }, [])

  const closeAuthModal = useCallback(() => {
    setIsOpen(false)
    // Restore focus to trigger element with a short tick
    setTimeout(() => {
      if (previousActiveElementRef.current && typeof previousActiveElementRef.current.focus === 'function') {
        previousActiveElementRef.current.focus()
      }
    }, 50)
  }, [])

  // Lock body scroll when modal is open
  useEffect(() => {
    if (isOpen) {
      const originalOverflow = document.body.style.overflow
      document.body.style.overflow = 'hidden'
      return () => {
        document.body.style.overflow = originalOverflow
      }
    }
  }, [isOpen])

  // Close on Escape key
  useEffect(() => {
    if (!isOpen) return

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault()
        closeAuthModal()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, closeAuthModal])

  return (
    <AuthModalContext.Provider
      value={{
        isOpen,
        mode,
        openAuthModal,
        closeAuthModal,
        setMode,
      }}
    >
      {children}
    </AuthModalContext.Provider>
  )
}

export function useAuthModal() {
  const context = useContext(AuthModalContext)
  if (!context) {
    throw new Error('useAuthModal must be used within an <AuthModalProvider>')
  }
  return context
}

export default AuthModalContext
