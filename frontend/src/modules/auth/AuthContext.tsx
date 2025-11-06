import React, { createContext, useContext, useMemo, useState } from 'react'
import { api } from '../../shared/api'

type UserProfile = {
  user_id: string
  google_id?: string
  email: string
  name: string
  picture?: string
}

type AuthToken = {
  access_token: string
  token_type: string
  expires_in: number
  user_profile: UserProfile
}

type AuthContextType = {
  token: string | null
  user: UserProfile | null
  isAuthenticated: boolean
  loading: boolean
  loginWithGoogle: () => Promise<void>
  handleAuthSuccess: (data: AuthToken) => void
  logout: () => void
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

const TOKEN_KEY = 'bisee.token'
const USER_KEY = 'bisee.user'

export function AuthProvider({ children }: { children: React.ReactNode }) {
  // Read from localStorage synchronously to avoid initial redirect flicker
  const [token, setToken] = useState<string | null>(() => {
    try { return localStorage.getItem(TOKEN_KEY) } catch { return null }
  })
  const [user, setUser] = useState<UserProfile | null>(() => {
    try {
      const raw = localStorage.getItem(USER_KEY)
      return raw ? JSON.parse(raw) as UserProfile : null
    } catch {
      return null
    }
  })
  const loading = false

  const handleAuthSuccess = (data: AuthToken) => {
    setToken(data.access_token)
    setUser(data.user_profile)
    localStorage.setItem(TOKEN_KEY, data.access_token)
    localStorage.setItem(USER_KEY, JSON.stringify(data.user_profile))
  }

  const loginWithGoogle = async () => {
    // Ask backend to generate auth URL with our frontend callback to capture the code
    const redirect_uri = `${window.location.origin}/auth/callback`
    const res = await api.get<{ auth_url: string }>(`/auth/google/url?redirect_uri=${encodeURIComponent(redirect_uri)}`)
    const { auth_url } = res
    window.location.href = auth_url
  }

  const logout = () => {
    setToken(null)
    setUser(null)
    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem(USER_KEY)
  }

  const value = useMemo(() => ({
    token,
    user,
    isAuthenticated: !!token,
    loading,
    loginWithGoogle,
    handleAuthSuccess,
    logout,
  }), [token, user])

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
