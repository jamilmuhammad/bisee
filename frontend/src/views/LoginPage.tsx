import { useAuth } from '../modules/auth/AuthContext'
import { Button } from '../widgets/ui/button'

export function LoginPage() {
  const { loginWithGoogle, isAuthenticated } = useAuth()
  return (
    <div className="min-h-screen flex items-center justify-center p-6">
      <div className="w-full max-w-md space-y-6">
        <h1 className="text-2xl font-semibold">BISEE AI</h1>
        <p className="text-sm text-muted-foreground">Sign in to start chatting with your data.</p>
        <Button className="w-full" onClick={loginWithGoogle}>
          Continue with Google
        </Button>
        {isAuthenticated && (
          <a className="text-blue-600 underline block text-center" href="/chat">Go to chat</a>
        )}
      </div>
    </div>
  )
}
