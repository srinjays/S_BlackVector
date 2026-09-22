/**
 * AuthModal — On-demand popup modal for Sign In / Sign Up
 *
 * Renders as a true overlay on top of any page view.
 * Only appears when explicitly triggered via openAuthModal().
 * Automatically closes upon successful login/signup, backdrop click, "X" button, or Escape key.
 */

import React, { useEffect, useRef } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { X, Info } from 'lucide-react'
import { useAuthModal } from './AuthModalContext'
import SignUpForm from './SignUpForm'
import LoginForm from './LoginForm'
import GoogleSignInButton from './GoogleSignInButton'
import { isSupabaseConfigured } from '@/lib/supabase'

export const AuthModal: React.FC = () => {
  const { isOpen, mode, closeAuthModal, setMode } = useAuthModal()
  const modalBoxRef = useRef<HTMLDivElement>(null)

  // Focus the first interactive input field inside the modal on open/tab change
  useEffect(() => {
    if (isOpen && modalBoxRef.current) {
      const timer = setTimeout(() => {
        const firstInput = modalBoxRef.current?.querySelector('input')
        if (firstInput && typeof firstInput.focus === 'function') {
          firstInput.focus()
        }
      }, 100)
      return () => clearTimeout(timer)
    }
  }, [isOpen, mode])

  return (
    <AnimatePresence>
      {isOpen && (
        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby="auth-modal-title"
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 99999,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '16px',
            overflowY: 'auto',
          }}
        >
          {/* Backdrop with blur */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            onClick={closeAuthModal}
            style={{
              position: 'fixed',
              inset: 0,
              background: 'rgba(0, 0, 0, 0.72)',
              backdropFilter: 'blur(12px)',
              WebkitBackdropFilter: 'blur(12px)',
            }}
          />

          {/* Modal Dialog Card */}
          <motion.div
            ref={modalBoxRef}
            initial={{ opacity: 0, scale: 0.95, y: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 10 }}
            transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
            onClick={(e) => e.stopPropagation()}
            style={{
              position: 'relative',
              width: '100%',
              maxWidth: 420,
              maxHeight: 'calc(100vh - 32px)',
              overflowY: 'auto',
              borderRadius: 20,
              background: 'rgba(18, 18, 20, 0.95)',
              backdropFilter: 'blur(28px)',
              WebkitBackdropFilter: 'blur(28px)',
              border: '1px solid rgba(255, 255, 255, 0.10)',
              boxShadow: '0 24px 80px rgba(0, 0, 0, 0.65), inset 0 1px 0 rgba(255, 255, 255, 0.06)',
              padding: '32px 28px 28px',
              display: 'flex',
              flexDirection: 'column',
              gap: 20,
              zIndex: 10,
            }}
          >
            {/* Explicit "X" Close Button */}
            <button
              onClick={closeAuthModal}
              aria-label="Close dialog"
              style={{
                position: 'absolute',
                top: 18,
                right: 18,
                width: 32,
                height: 32,
                borderRadius: '50%',
                border: 'none',
                background: 'rgba(255, 255, 255, 0.05)',
                color: 'rgba(255, 255, 255, 0.5)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                transition: 'background 150ms, color 150ms',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.12)'
                e.currentTarget.style.color = '#fff'
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)'
                e.currentTarget.style.color = 'rgba(255, 255, 255, 0.5)'
              }}
            >
              <X size={16} />
            </button>

            {/* Logo + Header */}
            <div style={{ textAlign: 'center' }}>
              <img
                src="/logo.png"
                alt="SatQuery AI"
                style={{
                  height: 38,
                  margin: '0 auto 10px',
                  display: 'block',
                  objectFit: 'contain',
                  filter: 'drop-shadow(0 2px 8px rgba(0,0,0,0.4))',
                }}
              />
              <h2
                id="auth-modal-title"
                style={{
                  fontSize: 20,
                  fontWeight: 700,
                  color: 'rgba(255,255,255,0.92)',
                  fontFamily: 'Inter, system-ui, sans-serif',
                  letterSpacing: '-0.02em',
                  margin: 0,
                }}
              >
                {mode === 'login' ? 'Welcome back' : 'Create your account'}
              </h2>
              <p
                style={{
                  fontSize: 13,
                  color: 'rgba(255,255,255,0.45)',
                  fontFamily: 'Inter, system-ui, sans-serif',
                  marginTop: 5,
                  marginBottom: 0,
                }}
              >
                {mode === 'login'
                  ? 'Sign in to access your satellite intelligence workspace'
                  : 'Start analyzing satellite imagery with agentic AI'}
              </p>
            </div>

            {/* Supabase unconfigured notice */}
            {!isSupabaseConfigured && (
              <div
                style={{
                  padding: '9px 12px',
                  borderRadius: 8,
                  background: 'rgba(234, 179, 8, 0.08)',
                  border: '1px solid rgba(234, 179, 8, 0.22)',
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: 8,
                }}
              >
                <Info size={15} style={{ color: '#facc15', flexShrink: 0, marginTop: 2 }} />
                <div style={{ fontSize: 11.5, color: 'rgba(255,255,255,0.7)', fontFamily: 'Inter, system-ui, sans-serif', lineHeight: 1.4 }}>
                  <span style={{ fontWeight: 600, color: '#fef08a' }}>Demo Mode:</span> Add your <code style={{ color: '#fef08a', background: 'rgba(255,255,255,0.08)', padding: '1px 4px', borderRadius: 4 }}>VITE_SUPABASE_URL</code> in <code style={{ color: '#fef08a', background: 'rgba(255,255,255,0.08)', padding: '1px 4px', borderRadius: 4 }}>.env.local</code> for live Supabase Auth.
                </div>
              </div>
            )}

            {/* Tab switcher: Sign In / Sign Up */}
            <div
              style={{
                display: 'flex',
                borderRadius: 10,
                background: 'rgba(255,255,255,0.04)',
                padding: 3,
                gap: 2,
              }}
            >
              <button
                type="button"
                onClick={() => setMode('login')}
                style={{
                  flex: 1,
                  padding: '8px 0',
                  borderRadius: 8,
                  border: 'none',
                  background: mode === 'login' ? 'rgba(255,255,255,0.12)' : 'transparent',
                  color: mode === 'login' ? 'rgba(255,255,255,0.95)' : 'rgba(255,255,255,0.4)',
                  fontSize: 13,
                  fontWeight: 600,
                  fontFamily: 'Inter, system-ui, sans-serif',
                  cursor: 'pointer',
                  transition: 'background 150ms, color 150ms',
                }}
              >
                Sign In
              </button>
              <button
                type="button"
                onClick={() => setMode('signup')}
                style={{
                  flex: 1,
                  padding: '8px 0',
                  borderRadius: 8,
                  border: 'none',
                  background: mode === 'signup' ? 'rgba(255,255,255,0.12)' : 'transparent',
                  color: mode === 'signup' ? 'rgba(255,255,255,0.95)' : 'rgba(255,255,255,0.4)',
                  fontSize: 13,
                  fontWeight: 600,
                  fontFamily: 'Inter, system-ui, sans-serif',
                  cursor: 'pointer',
                  transition: 'background 150ms, color 150ms',
                }}
              >
                Sign Up
              </button>
            </div>

            {/* Google OAuth Button */}
            <GoogleSignInButton
              label={mode === 'login' ? 'Sign in with Google' : 'Sign up with Google'}
            />

            {/* Divider */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 12,
              }}
            >
              <div style={{ flex: 1, height: 1, background: 'rgba(255,255,255,0.08)' }} />
              <span
                style={{
                  fontSize: 11,
                  color: 'rgba(255,255,255,0.28)',
                  fontFamily: 'Inter, system-ui, sans-serif',
                  fontWeight: 500,
                  textTransform: 'uppercase',
                  letterSpacing: '0.08em',
                }}
              >
                or
              </span>
              <div style={{ flex: 1, height: 1, background: 'rgba(255,255,255,0.08)' }} />
            </div>

            {/* Form */}
            {mode === 'login' ? (
              <LoginForm
                onSuccess={closeAuthModal}
                onSwitchToSignUp={() => setMode('signup')}
              />
            ) : (
              <SignUpForm
                onSuccess={closeAuthModal}
                onSwitchToLogin={() => setMode('login')}
              />
            )}
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  )
}

export default AuthModal
