// The Studio API client: JSON in and out, with the session's CSRF token on every call.

export function createApi(fetchImpl, headers) {
  return async (path, options = {}) => {
    const response = await fetchImpl(`/api/v1${path}`, {
      ...options,
      headers: { "Content-Type": "application/json", ...headers(), ...(options.headers || {}) },
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.error || `Request failed (${response.status})`);
    return body;
  };
}
