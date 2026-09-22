/**
 * GoogleSignInButton — "Continue with Google" OAuth button
 *
 * Triggers Supabase OAuth flow with Google provider.
 * Handles popup/redirect blocked, user cancellation, and network errors.
 */

import React, { useState } from 'react'
import { supabase } from '@/lib/supabase'
import { AlertCircle } from 'lucide-react'

/* ─────────────────────────────────────────────────────────
   Google "G" Logo SVG
───────────────────────────────────────────────────────── */
const GoogleLogo = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
    <path
      d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"
      fill="#4285F4"
    />
    <path
      d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
      fill="#34A853"
    />
    <path
      d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
      fill="#FBBC05"
    />
    <path
      d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
      fill="#EA4335"
    />
  </svg>
)

/* ─────────────────────────────────────────────────────────
   Component
───────────────────────────────────────────────────────── */
interface GoogleSignInButtonProps {
  label?: string
}

export const GoogleSignInButton: React.FC<GoogleSignInButtonProps> = ({
  label = 'Continue with Google',
}) => {
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [hovered, setHovered] = useState(false)

  const handleGoogleSignIn = async () => {
    setError(null)
    setIsLoading(true)

    const { error: oauthError } = await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: {
        redirectTo: window.location.origin,
      },
    })

    // If signInWithOAuth returns an error, it means the redirect was blocked
    if (oauthError) {
      setIsLoading(false)
      const msg = oauthError.message.toLowerCase()

      if (msg.includes('popup') || msg.includes('blocked')) {
        setError('Popup was blocked. Please allow popups and try again.')
      } else if (msg.includes('cancelled') || msg.includes('canceled') || msg.includes('user_cancelled')) {
        // User cancelled — silently reset, no error toast
        setError(null)
      } else if (msg.includes('already registered') || msg.includes('already been registered')) {
        setError('This email is already registered. Please sign in with your password instead.')
      } else if (msg.includes('network') || msg.includes('fetch')) {
        setError('Network error. Please check your connection and try again.')
      } else {
        setError(oauthError.message)
      }
      return
    }

    // If no error, Supabase will redirect to Google — loading stays true until redirect
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <button
        onClick={handleGoogleSignIn}
        disabled={isLoading}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        style={{
          width: '100%',
          padding: '12px 0',
          borderRadius: 10,
          border: '1px solid rgba(255,255,255,0.15)',
          background: hovered ? 'rgba(255,255,255,0.08)' : 'rgba(255,255,255,0.04)',
          color: 'rgba(255,255,255,0.88)',
          fontSize: 14,
          fontWeight: 500,
          fontFamily: 'Inter, system-ui, sans-serif',
          cursor: isLoading ? 'not-allowed' : 'pointer',
          transition: 'background 150ms, border-color 150ms',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 10,
          opacity: isLoading ? 0.6 : 1,
        }}
      >
        <GoogleLogo />
        {isLoading ? 'Redirecting…' : label}
      </button>

      {error && (
        <div style={{
          padding: '8px 12px',
          borderRadius: 8,
          background: 'rgba(239,68,68,0.1)',
          border: '1px solid rgba(239,68,68,0.2)',
          color: '#fca5a5',
          fontSize: 12,
          fontFamily: 'Inter, system-ui, sans-serif',
          display: 'flex',
          alignItems: 'center',
          gap: 6,
        }}>
          <AlertCircle size={14} />
          {error}
        </div>
      )}
    </div>
  )
}

export default GoogleSignInButton
