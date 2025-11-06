import { useEffect } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import { useAuth } from '../modules/auth/AuthContext'
import { api } from '../shared/api'

export function CallbackPage() {
  const [params] = useSearchParams()
  const navigate = useNavigate()
  const { handleAuthSuccess } = useAuth()

  useEffect(() => {
    const code = params.get('code')
    if (!code) return
    const run = async () => {
      try {
        const redirect_uri = `${window.location.origin}/auth/callback`
        const res = await api.post('/auth/google/callback', { code, redirect_uri })
        handleAuthSuccess(res as any)
        navigate('/chat', { replace: true })
      } catch (e) {
        console.error(e)
        navigate('/', { replace: true })
      }
    }
    run()
  }, [params, navigate, handleAuthSuccess])

  return (
    <div className="min-h-screen grid place-items-center">
      <div>Completing sign-in…</div>
    </div>
  )
}
