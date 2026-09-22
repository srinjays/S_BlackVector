/**
 * AuthProvider — React context for Supabase session management
 *
 * ⚠️  TEMPORARY: Auth is bypassed — a mock user is always provided.
 *     To re-enable real Supabase auth, restore the commented-out
 *     implementation below.
 *
 * Original behavior:
 *   Wraps the app and provides { user, session, loading, signOut }.
 *   Listens to supabase.auth.onAuthStateChange for real-time updates.
 */

import React, { createContext, useContext, useCallback } from 'react'
// import { supabase } from '@/lib/supabase'      // ← restore for real auth
// import type { User, Session } from '@supabase/supabase-js'

/* ─────────────────────────────────────────────────────────
   Context Type
───────────────────────────────────────────────────────── */
interface AuthContextType {
  user: any          // Using `any` while Supabase types are unused
  session: any
  loading: boolean
  signOut: () => Promise<void>
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  session: null,
  loading: false,
  signOut: async () => {},
})

/* ─────────────────────────────────────────────────────────
   Hook
───────────────────────────────────────────────────────── */
export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within <AuthProvider>')
  return ctx
}

/* ─────────────────────────────────────────────────────────
   Mock user — always "logged in"
───────────────────────────────────────────────────────── */
const MOCK_USER = {
  id: 'mock-user-001',
  email: 'analyst@satquery.ai',
  user_metadata: {
    full_name: 'SatQuery Analyst',
    name: 'SatQuery Analyst',
    avatar_url: null,
  },
}

/* ─────────────────────────────────────────────────────────
   Provider — BYPASSED (always authenticated)
───────────────────────────────────────────────────────── */
export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const signOut = useCallback(async () => {
    console.info('[AuthProvider] signOut called — auth is currently bypassed, no-op.')
  }, [])

  return (
    <AuthContext.Provider value={{ user: MOCK_USER, session: { user: MOCK_USER } as any, loading: false, signOut }}>
      {children}
    </AuthContext.Provider>
  )
}

export default AuthProvider

/* ─────────────────────────────────────────────────────────
   🔒 ORIGINAL SUPABASE IMPLEMENTATION (restore to re-enable auth)
   ─────────────────────────────────────────────────────────

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react'
import { supabase } from '@/lib/supabase'
import type { User, Session } from '@supabase/supabase-js'

interface AuthContextType {
  user: User | null
  session: Session | null
  loading: boolean
  signOut: () => Promise<void>
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  session: null,
  loading: true,
  signOut: async () => {},
})

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within <AuthProvider>')
  return ctx
}

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null)
  const [session, setSession] = useState<Session | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    supabase.auth.getSession()
      .then(({ data }) => {
        setSession(data?.session ?? null)
        setUser(data?.session?.user ?? null)
        setLoading(false)
      })
      .catch((err) => {
        console.warn('[AuthProvider] getSession error:', err)
        setLoading(false)
      })

    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (_event, s) => {
        setSession(s)
        setUser(s?.user ?? null)
        setLoading(false)
      }
    )

    return () => { subscription.unsubscribe() }
  }, [])

  const signOut = useCallback(async () => {
    await supabase.auth.signOut()
    setUser(null)
    setSession(null)
  }, [])

  return (
    <AuthContext.Provider value={{ user, session, loading, signOut }}>
      {children}
    </AuthContext.Provider>
  )
}

export default AuthProvider

───────────────────────────────────────────────────────── */
