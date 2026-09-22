/**
 * AuthPage — Combined signup/login page with tab toggle + Google OAuth
 *
 * Dark-themed glass card, centered on screen.
 * Handles OAuth callback by detecting hash params on mount.
 */

import React, { useState } from 'react'
import SignUpForm from './SignUpForm'
import LoginForm from './LoginForm'
import GoogleSignInButton from './GoogleSignInButton'
import { isSupabaseConfigured } from '@/lib/supabase'
import { Info } from 'lucide-react'

/* ─────────────────────────────────────────────────────────
   Tab type
───────────────────────────────────────────────────────── */
type AuthTab = 'login' | 'signup'

interface AuthPageProps {
  onSuccess?: () => void
  onSkip?: () => void
}

/* ─────────────────────────────────────────────────────────
   Component
───────────────────────────────────────────────────────── */
export const AuthPage: React.FC<AuthPageProps> = ({ onSuccess, onSkip }) => {
  const [tab, setTab] = useState<AuthTab>('login')

  return (
    <div
      style={{
        width: '100vw',
        height: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 20,
        position: 'relative',
        zIndex: 10,
      }}
    >
      {/* Glass card container */}
      <div
        style={{
          width: '100%',
          maxWidth: 420,
          borderRadius: 20,
          background: 'rgba(18, 18, 20, 0.92)',
          backdropFilter: 'blur(24px)',
          WebkitBackdropFilter: 'blur(24px)',
          border: '1px solid rgba(255,255,255,0.08)',
          boxShadow: '0 24px 80px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.04)',
          padding: '36px 32px 32px',
          display: 'flex',
          flexDirection: 'column',
          gap: 24,
        }}
      >
        {/* Logo + Title */}
        <div style={{ textAlign: 'center' }}>
          <img
            src="/logo.png"
            alt="SatQuery AI"
            style={{
              height: 40,
              margin: '0 auto 12px',
              display: 'block',
              objectFit: 'contain',
              filter: 'drop-shadow(0 2px 8px rgba(0,0,0,0.4))',
            }}
          />
          <h1
            style={{
              fontSize: 22,
              fontWeight: 700,
              color: 'rgba(255,255,255,0.92)',
              fontFamily: 'Inter, system-ui, sans-serif',
              letterSpacing: '-0.02em',
              margin: 0,
            }}
          >
            {tab === 'login' ? 'Welcome back' : 'Create your account'}
          </h1>
          <p
            style={{
              fontSize: 13.5,
              color: 'rgba(255,255,255,0.4)',
              fontFamily: 'Inter, system-ui, sans-serif',
              marginTop: 6,
            }}
          >
            {tab === 'login'
              ? 'Sign in to continue to SatQuery AI'
              : 'Get started with satellite intelligence'}
          </p>
        </div>

        {/* Supabase configuration notice if project ref is placeholder */}
        {!isSupabaseConfigured && (
          <div
            style={{
              padding: '10px 14px',
              borderRadius: 10,
              background: 'rgba(234, 179, 8, 0.08)',
              border: '1px solid rgba(234, 179, 8, 0.25)',
              display: 'flex',
              alignItems: 'flex-start',
              gap: 10,
            }}
          >
            <Info size={16} style={{ color: '#facc15', flexShrink: 0, marginTop: 2 }} />
            <div style={{ fontSize: 12, color: 'rgba(255,255,255,0.7)', fontFamily: 'Inter, system-ui, sans-serif', lineHeight: 1.4 }}>
              <span style={{ fontWeight: 600, color: '#fef08a' }}>Setup Note:</span> Set your <code style={{ color: '#fef08a', background: 'rgba(255,255,255,0.08)', padding: '2px 4px', borderRadius: 4 }}>VITE_SUPABASE_URL</code> in <code style={{ color: '#fef08a', background: 'rgba(255,255,255,0.08)', padding: '2px 4px', borderRadius: 4 }}>.env.local</code> to enable live Supabase authentication.
            </div>
          </div>
        )}

        {/* Tab switcher */}
        <div
          style={{
            display: 'flex',
            borderRadius: 10,
            background: 'rgba(255,255,255,0.04)',
            padding: 3,
            gap: 2,
          }}
        >
          {(['login', 'signup'] as AuthTab[]).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              style={{
                flex: 1,
                padding: '9px 0',
                borderRadius: 8,
                border: 'none',
                background: tab === t ? 'rgba(255,255,255,0.10)' : 'transparent',
                color: tab === t ? 'rgba(255,255,255,0.92)' : 'rgba(255,255,255,0.4)',
                fontSize: 13,
                fontWeight: 600,
                fontFamily: 'Inter, system-ui, sans-serif',
                cursor: 'pointer',
                transition: 'background 150ms, color 150ms',
              }}
            >
              {t === 'login' ? 'Sign In' : 'Sign Up'}
            </button>
          ))}
        </div>

        {/* Google OAuth */}
        <GoogleSignInButton
          label={tab === 'login' ? 'Sign in with Google' : 'Sign up with Google'}
        />

        {/* Divider */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 14,
          }}
        >
          <div style={{ flex: 1, height: 1, background: 'rgba(255,255,255,0.08)' }} />
          <span
            style={{
              fontSize: 12,
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
        {tab === 'login' ? (
          <LoginForm
            onSuccess={onSuccess}
            onSwitchToSignUp={() => setTab('signup')}
          />
        ) : (
          <SignUpForm
            onSuccess={onSuccess}
            onSwitchToLogin={() => setTab('login')}
          />
        )}

        {/* Skip / Guest option */}
        {onSkip && (
          <div style={{ textAlign: 'center', borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: 16 }}>
            <button
              type="button"
              onClick={onSkip}
              style={{
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                color: 'rgba(255,255,255,0.45)',
                fontSize: 12.5,
                fontFamily: 'Inter, system-ui, sans-serif',
                transition: 'color 150ms',
              }}
              onMouseEnter={(e) => (e.currentTarget.style.color = 'rgba(255,255,255,0.85)')}
              onMouseLeave={(e) => (e.currentTarget.style.color = 'rgba(255,255,255,0.45)')}
            >
              Continue to preview as guest &rarr;
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

export default AuthPage
