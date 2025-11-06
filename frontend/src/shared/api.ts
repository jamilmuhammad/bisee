type HttpMethod = 'GET' | 'POST' | 'PUT' | 'DELETE'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000/api/v1'

async function request<T>(path: string, method: HttpMethod, body?: any, token?: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method,
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: body ? JSON.stringify(body) : undefined,
    credentials: 'include',
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || `HTTP ${res.status}`)
  }
  const ct = res.headers.get('content-type') || ''
  if (ct.includes('application/json')) return res.json() as Promise<T>
  return ({} as unknown) as T
}

export const api = {
  get: <T>(path: string, token?: string) => request<T>(path, 'GET', undefined, token),
  post: <T>(path: string, body?: any, token?: string) => request<T>(path, 'POST', body, token),
}
