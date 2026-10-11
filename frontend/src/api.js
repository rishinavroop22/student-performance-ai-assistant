async function request(path, options = {}) {
  const res = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(body.error || "Request failed");
    err.fields = body.errors || {};
    throw err;
  }
  return body;
}

export const getOptions = () => request("/options");
export const getMetrics = () => request("/metrics");
export const getHealth = () => request("/health");
export const explain = (student) =>
  request("/explain", { method: "POST", body: JSON.stringify(student) });
