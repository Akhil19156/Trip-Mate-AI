import { createContext, useContext, useEffect, useMemo, useState } from "react";

const AuthContext = createContext(null);

function formatApiError(data, fallback) {
  if (typeof data.detail === "string") return data.detail;
  if (Array.isArray(data.detail)) {
    return data.detail
      .map((issue) => {
        const location = Array.isArray(issue.loc) ? issue.loc : [];
        const field = location
          .filter((part) => !["body", "query", "path"].includes(part))
          .at(-1);
        const label = field
          ? `${String(field).replaceAll("_", " ")}: `
          : "";
        const message = String(issue.msg || fallback).replace(
          /^Value error,\s*/i,
          "",
        );
        return `${label}${message}`;
      })
      .join(" ");
  }
  return data.error || fallback;
}

function csrfToken() {
  const csrfCookie = document.cookie
    .split("; ")
    .find((cookie) => cookie.startsWith("tripmate_csrf="));
  return csrfCookie ? decodeURIComponent(csrfCookie.split("=").slice(1).join("=")) : "";
}

export async function apiRequest(path, options = {}, canRefresh = true) {
  const method = (options.method || "GET").toUpperCase();
  const headers = new Headers(options.headers || {});
  const csrf = csrfToken();
  if (!["GET", "HEAD", "OPTIONS"].includes(method) && csrf) {
    headers.set("X-CSRF-Token", csrf);
  }
  const response = await fetch(path, {
    ...options,
    headers,
    credentials: "include",
  });
  const publicAuthEndpoint = ["/api/auth/login", "/api/auth/signup", "/api/auth/refresh"].includes(path);
  if (
    response.status === 401 &&
    canRefresh &&
    !publicAuthEndpoint &&
    csrf
  ) {
    const refreshed = await fetch("/api/auth/refresh", {
      method: "POST",
      credentials: "include",
      headers: { "X-CSRF-Token": csrf },
    });
    if (refreshed.ok) return apiRequest(path, options, false);
  }
  return response;
}

async function responseJson(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(
      formatApiError(data, "The request could not be completed."),
    );
  }
  return data;
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [preferences, setPreferences] = useState(null);
  const [loading, setLoading] = useState(true);

  async function loadCurrentUser() {
    const response = await apiRequest("/api/auth/me");
    const data = await responseJson(response);
    setUser(data.user);
    setPreferences(data.preferences);
    return data;
  }

  useEffect(() => {
    let active = true;
    apiRequest("/api/auth/me")
      .then(responseJson)
      .then((data) => {
        if (!active) return;
        setUser(data.user);
        setPreferences(data.preferences);
      })
      .catch(() => {
        if (active) {
          setUser(null);
          setPreferences(null);
        }
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  async function authenticate(endpoint, credentials) {
    const response = await apiRequest(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(credentials),
    });
    const data = await responseJson(response);
    setUser(data.user);
    setPreferences(data.preferences);
    return data;
  }

  async function login(credentials) {
    return authenticate("/api/auth/login", credentials);
  }

  async function signup(credentials) {
    return authenticate("/api/auth/signup", credentials);
  }

  async function logout() {
    await responseJson(
      await apiRequest("/api/auth/logout", { method: "POST" }),
    );
    localStorage.removeItem("travel_thread_id");
    setUser(null);
    setPreferences(null);
  }

  async function updateProfile(payload) {
    const data = await responseJson(
      await apiRequest("/api/auth/me", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }),
    );
    setUser(data);
    return data;
  }

  async function updatePreferences(payload) {
    const data = await responseJson(
      await apiRequest("/api/auth/preferences", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }),
    );
    setPreferences(data);
    return data;
  }

  const value = useMemo(
    () => ({
      user,
      preferences,
      loading,
      login,
      signup,
      logout,
      updateProfile,
      updatePreferences,
      loadCurrentUser,
    }),
    [user, preferences, loading],
  );
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used within AuthProvider.");
  return context;
}

export function responseError(response, fallback = "The request could not be completed.") {
  return response
    .json()
    .catch(() => ({}))
    .then((data) => {
      if (!response.ok) throw new Error(formatApiError(data, fallback));
      return data;
    });
}
