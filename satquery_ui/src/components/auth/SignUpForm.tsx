/**
 * SignUpForm — Email/password registration with inline validation
 */

import React, { useState } from 'react'
import { supabase } from '@/lib/supabase'
import { Loader2, Eye, EyeOff, CheckCircle2, AlertCircle } from 'lucide-react'

/* ─────────────────────────────────────────────────────────
   Validation helpers
───────────────────────────────────────────────────────── */
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/

function validateForm(name: string, email: string, password: string, confirmPassword: string) {
  const errors: Record<string, string> = {}
  if (!name.trim()) errors.name = 'Full name is required'
  if (!email.trim()) errors.email = 'Email is required'
  else if (!EMAIL_RE.test(email)) errors.email = 'Please enter a valid email address'
  if (!password) errors.password = 'Password is required'
  else if (password.length < 8) errors.password = 'Password must be at least 8 characters'
  else if (!/[A-Z]/.test(password)) errors.password = 'Include at least one uppercase letter'
  else if (!/[0-9]/.test(password)) errors.password = 'Include at least one number'
  if (password !== confirmPassword) errors.confirmPassword = 'Passwords do not match'
  return errors
}

function getPasswordStrength(pw: string): { score: number; label: string; color: string } {
  let score = 0
  if (pw.length >= 8) score++
  if (pw.length >= 12) score++
  if (/[A-Z]/.test(pw) && /[a-z]/.test(pw)) score++
  if (/[0-9]/.test(pw)) score++
  if (/[^A-Za-z0-9]/.test(pw)) score++

  if (score <= 1) return { score, label: 'Weak', color: '#ef4444' }
  if (score <= 2) return { score, label: 'Fair', color: '#f59e0b' }
  if (score <= 3) return { score, label: 'Good', color: '#3b82f6' }
  return { score, label: 'Strong', color: '#22c55e' }
}

/* ─────────────────────────────────────────────────────────
   Styles
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
interface SignUpFormProps {
  onSuccess?: () => void
  onSwitchToLogin?: () => void
}

export const SignUpForm: React.FC<SignUpFormProps> = ({ onSuccess, onSwitchToLogin }) => {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [success, setSuccess] = useState(false)
  const [focusedField, setFocusedField] = useState<string | null>(null)

  const strength = getPasswordStrength(password)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const validationErrors = validateForm(name, email, password, confirmPassword)
    setErrors(validationErrors)
    if (Object.keys(validationErrors).length > 0) return

    setIsSubmitting(true)
    setErrors({})

    const { error } = await supabase.auth.signUp({
      email: email.trim(),
      password,
      options: {
        data: { full_name: name.trim() },
      },
    })

    setIsSubmitting(false)

    if (error) {
      // Map Supabase errors to user-friendly inline messages
      const msg = error.message.toLowerCase()
      if (msg.includes('already registered') || msg.includes('already been registered')) {
        setErrors({ email: 'This email is already registered. Try signing in instead.' })
      } else if (msg.includes('password') && (msg.includes('weak') || msg.includes('short'))) {
        setErrors({ password: 'Password is too weak. Use at least 8 characters with uppercase and numbers.' })
      } else if (msg.includes('valid email') || msg.includes('invalid')) {
        setErrors({ email: 'Please enter a valid email address.' })
      } else {
        setErrors({ form: error.message })
      }
      return
    }

    setSuccess(true)
    onSuccess?.()
  }

  if (success) {
    return (
      <div style={{ textAlign: 'center', padding: '32px 0' }}>
        <CheckCircle2 size={48} style={{ color: '#22c55e', margin: '0 auto 16px' }} />
        <p style={{ color: 'rgba(255,255,255,0.88)', fontSize: 16, fontWeight: 600, fontFamily: 'Inter, system-ui, sans-serif' }}>
          Account created!
        </p>
        <p style={{ color: 'rgba(255,255,255,0.45)', fontSize: 13, marginTop: 8, fontFamily: 'Inter, system-ui, sans-serif' }}>
          Check your email for a confirmation link, then sign in.
        </p>
      </div>
    )
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

      {/* Full Name */}
      <div>
        <label style={labelStyle}>Full Name</label>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          onFocus={() => setFocusedField('name')}
          onBlur={() => setFocusedField(null)}
          placeholder="John Doe"
          style={{ ...inputStyle, ...(focusedField === 'name' ? inputFocusStyle : {}) }}
        />
        {errors.name && <p style={errorTextStyle}><AlertCircle size={12} />{errors.name}</p>}
      </div>

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
            placeholder="Min. 8 characters"
            autoComplete="new-password"
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
        {/* Password strength bar */}
        {password.length > 0 && (
          <div style={{ marginTop: 8 }}>
            <div style={{
              height: 4, borderRadius: 2, background: 'rgba(255,255,255,0.08)',
              overflow: 'hidden',
            }}>
              <div style={{
                height: '100%',
                width: `${(strength.score / 5) * 100}%`,
                background: strength.color,
                borderRadius: 2,
                transition: 'width 250ms ease, background 250ms ease',
              }} />
            </div>
            <p style={{
              fontSize: 11, color: strength.color, marginTop: 4,
              fontFamily: 'Inter, system-ui, sans-serif', fontWeight: 500,
            }}>
              {strength.label}
            </p>
          </div>
        )}
        {errors.password && <p style={errorTextStyle}><AlertCircle size={12} />{errors.password}</p>}
      </div>

      {/* Confirm Password */}
      <div>
        <label style={labelStyle}>Confirm Password</label>
        <input
          type="password"
          value={confirmPassword}
          onChange={(e) => setConfirmPassword(e.target.value)}
          onFocus={() => setFocusedField('confirm')}
          onBlur={() => setFocusedField(null)}
          placeholder="Re-enter your password"
          autoComplete="new-password"
          style={{ ...inputStyle, ...(focusedField === 'confirm' ? inputFocusStyle : {}) }}
        />
        {errors.confirmPassword && <p style={errorTextStyle}><AlertCircle size={12} />{errors.confirmPassword}</p>}
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
        {isSubmitting ? 'Creating account…' : 'Create Account'}
      </button>

      {/* Switch to login */}
      <p style={{
        textAlign: 'center', fontSize: 13,
        color: 'rgba(255,255,255,0.4)', fontFamily: 'Inter, system-ui, sans-serif',
      }}>
        Already have an account?{' '}
        <button
          type="button"
          onClick={onSwitchToLogin}
          style={{
            background: 'none', border: 'none', cursor: 'pointer',
            color: 'rgb(129,140,248)', fontWeight: 500, fontSize: 13,
            textDecoration: 'underline',
          }}
        >
          Sign in
        </button>
      </p>
    </form>
  )
}

export default SignUpForm
