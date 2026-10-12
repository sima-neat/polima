// JSON requests with session recovery after a rejected CSRF token.
export function createApi(fetchImpl, headers, updateSession = () => {}) {
  return async (path, options = {}) => {
    const requestHeaders = { "Content-Type": "application/json", ...headers(), ...(options.headers || {}) };
    for (let attempt = 0; attempt < 2; attempt++) {
      const response = await fetchImpl(`/api/v1${path}`, { ...options, headers: { ...requestHeaders } });
      const body = await response.json().catch(() => ({}));
      if (response.ok) return body;
      if (attempt === 0 && response.status === 403 && body.error === "invalid CSRF token") {
        const sessionResponse = await fetchImpl("/api/v1/session", { cache: "no-store" });
        const session = await sessionResponse.json();
        if (!sessionResponse.ok || !session.csrf_token) throw new Error("Could not renew the Studio session. Reload the page.");
        requestHeaders["X-CSRF-Token"] = session.csrf_token;
        updateSession(session.csrf_token);
        continue;
      }
      throw new Error(body.error || `Request failed (${response.status})`);
    }
  };
}
