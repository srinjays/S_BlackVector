/**
 * LoginForm — Email/password sign-in with inline validation
 */

import React, { useState } from 'react'
import { supabase } from '@/lib/supabase'
import { Loader2, Eye, EyeOff, AlertCircle } from 'lucide-react'

/* ─────────────────────────────────────────────────────────
   Styles (shared with SignUpForm)
───────────────────────────────────────────────────────── */
const inputStyle: React.CSSProperties = {
  width: '100%',
  padding: '12px 14px',
  borderRadius: 10,
  border: '1px solid rgba(255,255,255,0.12)',
  background: 'rgba(255,255,255,0.04)',
  color: 'rgba(255,255,255,0.92)',
  fontSize: 14,
  fontFamily: 'Inter, system-ui, sans-serif',
  outline: 'none',
  transition: 'border-color 150ms, box-shadow 150ms',
}

const inputFocusStyle: React.CSSProperties = {
  borderColor: 'rgba(99,102,241,0.6)',
  boxShadow: '0 0 0 3px rgba(99,102,241,0.15)',
}

const labelStyle: React.CSSProperties = {
  display: 'block',
  fontSize: 13,
  fontWeight: 500,
  color: 'rgba(255,255,255,0.6)',
  marginBottom: 6,
  fontFamily: 'Inter, system-ui, sans-serif',
}

const errorTextStyle: React.CSSProperties = {
  fontSize: 12,
  color: '#f87171',
  marginTop: 4,
  display: 'flex',
  alignItems: 'center',
  gap: 4,
  fontFamily: 'Inter, system-ui, sans-serif',
}

/* ─────────────────────────────────────────────────────────
   Component
───────────────────────────────────────────────────────── */
interface LoginFormProps {
  onSuccess?: () => void
  onSwitchToSignUp?: () => void
}

export const LoginForm: React.FC<LoginFormProps> = ({ onSuccess, onSwitchToSignUp }) => {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [focusedField, setFocusedField] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const validationErrors: Record<string, string> = {}
    if (!email.trim()) validationErrors.email = 'Email is required'
    if (!password) validationErrors.password = 'Password is required'
    setErrors(validationErrors)
    if (Object.keys(validationErrors).length > 0) return

    setIsSubmitting(true)
    setErrors({})

    const { error } = await supabase.auth.signInWithPassword({
      email: email.trim(),
      password,
    })

    setIsSubmitting(false)

    if (error) {
      const msg = error.message.toLowerCase()
      if (msg.includes('invalid login') || msg.includes('invalid credentials') || msg.includes('wrong password')) {
        setErrors({ form: 'Invalid email or password. Please try again.' })
      } else if (msg.includes('email not confirmed')) {
        setErrors({ form: 'Please confirm your email address before signing in.' })
      } else {
        setErrors({ form: error.message })
      }
      return
    }

    onSuccess?.()
  }

  return (
    <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      {/* Global form error */}
      {errors.form && (
        <div style={{
          padding: '10px 14px',
          borderRadius: 8,
          background: 'rgba(239,68,68,0.1)',
          border: '1px solid rgba(239,68,68,0.25)',
          color: '#fca5a5',
          fontSize: 13,
          fontFamily: 'Inter, system-ui, sans-serif',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
        }}>
          <AlertCircle size={16} />
          {errors.form}
        </div>
      )}

      {/* Email */}
      <div>
        <label style={labelStyle}>Email</label>
        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          onFocus={() => setFocusedField('email')}
          onBlur={() => setFocusedField(null)}
          placeholder="you@example.com"
          autoComplete="email"
          style={{ ...inputStyle, ...(focusedField === 'email' ? inputFocusStyle : {}) }}
        />
        {errors.email && <p style={errorTextStyle}><AlertCircle size={12} />{errors.email}</p>}
      </div>

      {/* Password */}
      <div>
        <label style={labelStyle}>Password</label>
        <div style={{ position: 'relative' }}>
          <input
            type={showPassword ? 'text' : 'password'}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            onFocus={() => setFocusedField('password')}
            onBlur={() => setFocusedField(null)}
            placeholder="Enter your password"
            autoComplete="current-password"
            style={{ ...inputStyle, paddingRight: 44, ...(focusedField === 'password' ? inputFocusStyle : {}) }}
          />
          <button
            type="button"
            onClick={() => setShowPassword(!showPassword)}
            tabIndex={-1}
            style={{
              position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)',
              background: 'none', border: 'none', cursor: 'pointer', padding: 4,
              color: 'rgba(255,255,255,0.4)',
            }}
          >
            {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
          </button>
        </div>
        {errors.password && <p style={errorTextStyle}><AlertCircle size={12} />{errors.password}</p>}
      </div>

      {/* Submit */}
      <button
        type="submit"
        disabled={isSubmitting}
        style={{
          width: '100%',
          padding: '12px 0',
          borderRadius: 10,
          border: 'none',
          background: isSubmitting ? 'rgba(99,102,241,0.4)' : 'rgb(99,102,241)',
          color: '#fff',
          fontSize: 14,
          fontWeight: 600,
          fontFamily: 'Inter, system-ui, sans-serif',
          cursor: isSubmitting ? 'not-allowed' : 'pointer',
          transition: 'background 150ms, transform 100ms',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 8,
        }}
      >
        {isSubmitting && <Loader2 size={18} style={{ animation: 'spin 1s linear infinite' }} />}
        {isSubmitting ? 'Signing in…' : 'Sign In'}
      </button>

      {/* Switch to signup */}
      <p style={{
        textAlign: 'center', fontSize: 13,
        color: 'rgba(255,255,255,0.4)', fontFamily: 'Inter, system-ui, sans-serif',
      }}>
        Don&apos;t have an account?{' '}
        <button
          type="button"
          onClick={onSwitchToSignUp}
          style={{
            background: 'none', border: 'none', cursor: 'pointer',
            color: 'rgb(129,140,248)', fontWeight: 500, fontSize: 13,
            textDecoration: 'underline',
          }}
        >
          Create one
        </button>
      </p>
    </form>
  )
}

export default LoginForm
