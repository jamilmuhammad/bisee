import { FormEvent, useEffect, useMemo, useRef, useState } from 'react'
import { useAuth } from '../modules/auth/AuthContext'
import { api } from '../shared/api'
import { Button } from '../widgets/ui/button'

type ChatResponse = {
  response: string
  session_id: string
  query_types: string[]
  context: string
  sql_query?: string
  query_results?: { success?: boolean; row_count?: number; data?: any[] }
  query_data?: Record<string, any>[]
  final_response: string
}

export function ChatPage() {
  const { token, user, logout } = useAuth()
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState<{ role: 'user' | 'assistant'; text: string }[]>([])
  const [lastTable, setLastTable] = useState<any[] | null>(null)
  const endRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  const header = useMemo(() => (
    <div className="flex items-center justify-between p-4 border-b">
      <div className="flex items-center gap-3">
        {user?.picture && <img src={user.picture} className="w-8 h-8 rounded-full" />}
        <div className="text-sm">
          <div className="font-medium">{user?.name || 'Guest'}</div>
          <div className="text-muted-foreground">{user?.email}</div>
        </div>
      </div>
      <Button variant="outline" onClick={logout}>Logout</Button>
    </div>
  ), [user, logout])

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault()
    const text = input.trim()
    if (!text) return
    setInput('')
    setMessages(m => [...m, { role: 'user', text }])
    try {
      const res = await api.post<ChatResponse>('/rag-chat/rag', { message: text, session_id: sessionId }, token || undefined)
      setSessionId(res.session_id)
      const display = res.final_response || res.response
      setMessages(m => [...m, { role: 'assistant', text: display }])
      setLastTable(res.query_data && Array.isArray(res.query_data) && res.query_data.length ? res.query_data : null)
    } catch (e: any) {
      setMessages(m => [...m, { role: 'assistant', text: `Error: ${e.message}` }])
    }
  }

  return (
    <div className="min-h-screen flex flex-col">
      {header}
      <div className="flex-1 p-4 max-w-3xl w-full mx-auto space-y-3">
        {messages.map((m, i) => (
          <div key={i} className={m.role === 'user' ? 'text-right' : 'text-left'}>
            <div className={m.role === 'user' ? 'inline-block bg-primary text-white px-3 py-2 rounded-md' : 'inline-block bg-muted px-3 py-2 rounded-md'}>
              {m.text}
            </div>
          </div>
        ))}
        {lastTable && (
          <div className="overflow-x-auto border rounded-md">
            <table className="min-w-full text-sm">
              <thead className="bg-muted">
                <tr>
                  {Object.keys(lastTable[0]).map((k) => (
                    <th key={k} className="text-left px-3 py-2 font-medium">{k}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {lastTable.map((row, idx) => (
                  <tr key={idx} className="even:bg-muted/50">
                    {Object.values(row).map((v, i) => (
                      <td key={i} className="px-3 py-2 whitespace-nowrap">{String(v)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <div ref={endRef} />
      </div>
      <form onSubmit={onSubmit} className="p-4 border-t">
        <div className="max-w-3xl mx-auto flex gap-2">
          <input
            className="flex-1 border rounded-md px-3 py-2 focus:outline-none focus:ring"
            placeholder="Ask your data…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
          />
          <Button type="submit">Send</Button>
        </div>
      </form>
    </div>
  )
}
