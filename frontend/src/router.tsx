import { createBrowserRouter, Navigate } from 'react-router-dom'
import { LoginPage } from './views/LoginPage'
import { useAuth } from './modules/auth/AuthContext'
import { CallbackPage } from './views/CallbackPage'
import { ChatPage } from './views/ChatPage'

function ProtectedRoute({ children }: { children: JSX.Element }) {
  const { isAuthenticated, loading } = useAuth()
  // Avoid redirect during initial hydration
  if (loading) return null
  if (!isAuthenticated) return <Navigate to="/" replace />
  return children
}

export const router = createBrowserRouter([
  { path: '/', element: <LoginPage /> },
  { path: '/auth/callback', element: <CallbackPage /> },
  { path: '/chat', element: (
      <ProtectedRoute>
        <ChatPage />
      </ProtectedRoute>
    )
  },
])
