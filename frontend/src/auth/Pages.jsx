import { useCallback, useEffect, useState } from "react";
import {
  Link,
  Navigate,
  Outlet,
  useNavigate,
  useParams,
} from "react-router-dom";
import { apiRequest, responseError, useAuth } from "./AuthContext";

export function ProtectedRoute() {
  const { user, loading } = useAuth();
  if (loading) return <div className="auth-loading">Loading your TripMate account…</div>;
  return user ? <Outlet /> : <Navigate to="/login" replace />;
}

export function AppShell() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [logoutError, setLogoutError] = useState("");

  async function handleLogout() {
    setLogoutError("");
    try {
      await logout();
      navigate("/login", { replace: true });
    } catch (error) {
      setLogoutError(error.message);
    }
  }

  return (
    <div className="site-shell">
      <header className="topbar app-topbar">
        <Link className="brand" to="/" aria-label="TripMate dashboard">
          <span className="brand-mark">t.</span>
          <span>tripmate<span className="brand-ai"> ai</span></span>
        </Link>
        <nav className="account-nav" aria-label="Account navigation">
          <Link to="/">Overview</Link>
          <Link to="/plan">Plan a trip</Link>
          <Link to="/trips">My trips</Link>
          <Link to="/profile">Profile</Link>
        </nav>
        <div className="account-menu">
          <span className="account-name">{user?.name}</span>
          <button type="button" className="text-button" onClick={handleLogout}>
            Log out
          </button>
        </div>
      </header>
      <main className="main-wrap account-main">
        {logoutError && <div className="error-notice" role="alert">{logoutError}</div>}
        <Outlet />
      </main>
      <footer className="site-footer account-footer">
        <Link className="brand footer-brand" to="/">
          <span className="brand-mark">t.</span>
          <span>tripmate<span className="brand-ai"> ai</span></span>
        </Link>
        <span>Plan thoughtfully. Travel curiously.</span>
      </footer>
    </div>
  );
}

export function AuthPage({ mode }) {
  const isSignup = mode === "signup";
  const { user, loading, login, signup } = useAuth();
  const navigate = useNavigate();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  if (!loading && user) return <Navigate to="/" replace />;

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    if (isSignup && password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }
    setSubmitting(true);
    try {
      if (isSignup) await signup({ name, email, password });
      else await login({ email, password });
      navigate("/", { replace: true });
    } catch (submitError) {
      setError(submitError.message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="auth-page">
      <Link className="brand auth-brand" to="/">
        <span className="brand-mark">t.</span>
        <span>tripmate<span className="brand-ai"> ai</span></span>
      </Link>
      <section className="auth-card">
        <p className="kicker">{isSignup ? "YOUR TRIPMATE ACCOUNT" : "WELCOME BACK"}</p>
        <h1>{isSignup ? "Make room for the next trip." : "Good to see you again."}</h1>
        <p className="muted">
          {isSignup
            ? "Save your travel plans and pick up where you left off."
            : "Sign in to see your saved trips and keep planning."}
        </p>
        {error && <div className="error-notice" role="alert">{error}</div>}
        <form className="auth-form" onSubmit={handleSubmit}>
          {isSignup && (
            <label>
              Full name
              <input
                autoComplete="name"
                required
                maxLength={120}
                value={name}
                onChange={(event) => setName(event.target.value)}
              />
            </label>
          )}
          <label>
            Email
            <input
              type="email"
              autoComplete="email"
              required
              maxLength={320}
              value={email}
              onChange={(event) => setEmail(event.target.value)}
            />
          </label>
          <label>
            Password
            <input
              type="password"
              autoComplete={isSignup ? "new-password" : "current-password"}
              required
              minLength={isSignup ? 12 : 1}
              maxLength={128}
              value={password}
              onChange={(event) => setPassword(event.target.value)}
            />
          </label>
          {isSignup && (
            <>
              <small className="password-help">
                Use at least 12 characters with uppercase, lowercase, a number,
                and a symbol.
              </small>
              <label>
                Confirm password
                <input
                  type="password"
                  autoComplete="new-password"
                  required
                  maxLength={128}
                  value={confirmPassword}
                  onChange={(event) => setConfirmPassword(event.target.value)}
                />
              </label>
            </>
          )}
          <button className="primary-button auth-submit" disabled={submitting}>
            {submitting ? "Please wait…" : isSignup ? "Create account" : "Log in"}
          </button>
        </form>
        <p className="auth-switch">
          {isSignup ? "Already have an account?" : "New to TripMate?"}{" "}
          <Link to={isSignup ? "/login" : "/signup"}>
            {isSignup ? "Log in" : "Create an account"}
          </Link>
        </p>
      </section>
    </div>
  );
}

export function DashboardPage() {
  const { user, preferences } = useAuth();
  const [trips, setTrips] = useState([]);
  const [tripCount, setTripCount] = useState(0);
  const [error, setError] = useState("");

  useEffect(() => {
    apiRequest("/api/trips/summary")
      .then((response) => responseError(response))
      .then((summary) => {
        setTrips(summary.recent_trips);
        setTripCount(summary.total_trips);
      })
      .catch((requestError) => setError(requestError.message));
  }, []);

  return (
    <div className="account-page">
      <section className="dashboard-welcome">
        <div>
          <p className="kicker">YOUR TRIP DESK</p>
          <h1>Welcome back, {user?.name?.split(" ")[0]}.</h1>
          <p className="muted">Your saved plans and next ideas, all in one place.</p>
        </div>
        <Link className="primary-button link-button" to="/plan">Plan a trip <span>↗</span></Link>
      </section>
      {error && <div className="error-notice" role="alert">{error}</div>}
      <div className="dashboard-stats">
        <article className="account-card stat-card">
          <span className="detail-label">TRIPS SAVED</span>
          <strong>{tripCount}</strong>
          <span className="muted">Recent trips in your history</span>
        </article>
        <article className="account-card stat-card">
          <span className="detail-label">TRAVEL STYLE</span>
          <strong>{preferences?.travel_style || "Your own pace"}</strong>
          <span className="muted">Personalized from your preferences</span>
        </article>
      </div>
      <section className="account-card">
        <div className="account-section-heading">
          <div>
            <p className="kicker">RECENT PLANS</p>
            <h2>Trips you’ve been thinking about</h2>
          </div>
          <Link className="text-button" to="/trips">See all trips →</Link>
        </div>
        <TripList trips={trips} empty="Your first saved itinerary will appear here." />
      </section>
    </div>
  );
}

function TripList({ trips, empty, onDelete, onResume }) {
  if (!trips.length) return <p className="empty-state">{empty}</p>;
  return (
    <div className="trip-list">
      {trips.map((trip) => (
        <article className="trip-row" key={trip.id}>
          <div className="trip-destination-mark">{trip.destination.slice(0, 1).toUpperCase()}</div>
          <div className="trip-row-main">
            <Link to={`/trips/${trip.id}`} className="trip-title">
              {trip.title || `Trip to ${trip.destination}`}
            </Link>
            <span className="muted">
              {trip.origin ? `${trip.origin} → ` : ""}
              {trip.destination}
              {trip.travel_start_date ? ` · ${trip.travel_start_date}` : ""}
              {trip.travel_end_date ? ` – ${trip.travel_end_date}` : ""}
            </span>
          </div>
          <span className={`trip-status status-${trip.status}`}>{trip.status.replaceAll("_", " ")}</span>
          {onResume && trip.thread_id && ["pending_approval", "revision_requested"].includes(trip.status) && (
            <button type="button" className="text-button" onClick={() => onResume(trip)}>
              Resume
            </button>
          )}
          {onDelete && (
            <button type="button" className="text-button danger-text" onClick={() => onDelete(trip)}>
              Delete
            </button>
          )}
        </article>
      ))}
    </div>
  );
}

export function TripsPage() {
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("");
  const [trips, setTrips] = useState([]);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const loadTrips = useCallback(async () => {
    const params = new URLSearchParams();
    if (search.trim()) params.set("search", search.trim());
    if (status) params.set("status", status);
    try {
      setTrips(await responseError(await apiRequest(`/api/trips?${params}`)));
      setError("");
    } catch (requestError) {
      setError(requestError.message);
    }
  }, [search, status]);

  useEffect(() => {
    loadTrips();
  }, [loadTrips]);

  async function deleteTrip(trip) {
    if (!window.confirm(`Delete your trip to ${trip.destination}?`)) return;
    try {
      await responseError(
        await apiRequest(`/api/trips/${trip.id}`, { method: "DELETE" }),
      );
      await loadTrips();
    } catch (requestError) {
      setError(requestError.message);
    }
  }

  async function resumeTrip(trip) {
      navigate(`/plan?trip_id=${trip.id}`);
    }

  return (
    <section className="account-card account-page">
      <div className="account-section-heading">
        <div>
          <p className="kicker">YOUR TRAVEL HISTORY</p>
          <h1>My trips</h1>
        </div>
        <Link className="primary-button link-button" to="/plan">Plan a trip</Link>
      </div>
      <div className="trip-filters">
        <input
          aria-label="Search trips by destination"
          placeholder="Search destinations"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
        <select aria-label="Filter by trip status" value={status} onChange={(event) => setStatus(event.target.value)}>
          <option value="">All statuses</option>
          <option value="pending_approval">Waiting for review</option>
          <option value="approved">Approved</option>
          <option value="completed">Completed</option>
          <option value="saved">Saved</option>
        </select>
      </div>
      {error && <div className="error-notice" role="alert">{error}</div>}
      <TripList trips={trips} empty="No trips found. Start with a destination you’re curious about." onDelete={deleteTrip} onResume={resumeTrip} />
    </section>
  );
}

export function TripDetailPage() {
  const { tripId } = useParams();
  const [detail, setDetail] = useState(null);
  const [editForm, setEditForm] = useState(null);
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    apiRequest(`/api/trips/${tripId}`)
      .then((response) => responseError(response))
      .then(setDetail)
      .catch((requestError) => setError(requestError.message));
  }, [tripId]);

  if (!detail && error) return <div className="error-notice" role="alert">{error}</div>;
  if (!detail) return <div className="auth-loading">Loading your trip…</div>;
  const { trip, conversations } = detail;

  function startEditing() {
    setError("");
    setEditForm({
      title: trip.title || "",
      origin: trip.origin || "",
      destination: trip.destination || "",
      travel_start_date: trip.travel_start_date || "",
      travel_end_date: trip.travel_end_date || "",
      budget: trip.budget ?? "",
    });
  }

  async function saveTrip(event) {
    event.preventDefault();
    setError("");
    setSaving(true);
    try {
      const updated = await responseError(
        await apiRequest(`/api/trips/${trip.id}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ...editForm,
            travel_start_date: editForm.travel_start_date || null,
            travel_end_date: editForm.travel_end_date || null,
            budget: editForm.budget === "" ? null : Number(editForm.budget),
          }),
        }),
      );
      setDetail((current) => ({ ...current, trip: updated }));
      setEditForm(null);
    } catch (saveError) {
      setError(saveError.message);
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="account-card account-page">
      <Link className="text-button" to="/trips">← My trips</Link>
      <p className="kicker">SAVED ITINERARY</p>
      <h1>{trip.title || `Trip to ${trip.destination}`}</h1>
      <p className="muted">{trip.origin ? `${trip.origin} → ` : ""}{trip.destination} · {trip.status.replaceAll("_", " ")}</p>
      {error && <div className="error-notice" role="alert">{error}</div>}
      {editForm ? (
        <form className="profile-form trip-edit-form" onSubmit={saveTrip}>
          <label>Trip title<input maxLength={200} value={editForm.title} onChange={(event) => setEditForm({ ...editForm, title: event.target.value })} /></label>
          <label>Origin<input maxLength={160} value={editForm.origin} onChange={(event) => setEditForm({ ...editForm, origin: event.target.value })} /></label>
          <label>Destination<input required maxLength={160} value={editForm.destination} onChange={(event) => setEditForm({ ...editForm, destination: event.target.value })} /></label>
          <label>Start date<input type="date" value={editForm.travel_start_date} onChange={(event) => setEditForm({ ...editForm, travel_start_date: event.target.value })} /></label>
          <label>End date<input type="date" value={editForm.travel_end_date} onChange={(event) => setEditForm({ ...editForm, travel_end_date: event.target.value })} /></label>
          <label>Budget<input type="number" min="0" step="0.01" value={editForm.budget} onChange={(event) => setEditForm({ ...editForm, budget: event.target.value })} /></label>
          <div className="profile-wide trip-edit-actions">
            <button className="primary-button" type="submit" disabled={saving}>{saving ? "Saving…" : "Save trip details"}</button>
            <button className="text-button" type="button" onClick={() => setEditForm(null)} disabled={saving}>Cancel</button>
          </div>
        </form>
      ) : (
        <button type="button" className="text-button" onClick={startEditing}>Edit trip details</button>
      )}
      {trip.itinerary && <pre className="saved-itinerary">{trip.itinerary}</pre>}
      {trip.thread_id && ["pending_approval", "revision_requested"].includes(trip.status) && (
        <button type="button" className="primary-button" onClick={() => navigate(`/plan?trip_id=${trip.id}`)}>
          Resume planning
        </button>
      )}
      <div className="conversation-history">
        <h2>Conversation</h2>
        {conversations.map((message) => (
          <article className={`conversation-message ${message.role}`} key={message.id}>
            <strong>{message.role === "user" ? "You" : "TripMate"}</strong>
            <p>{message.message}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

export function ProfilePage() {
  const { user, preferences, updateProfile, updatePreferences } = useAuth();
  const [name, setName] = useState(user?.name || "");
  const [form, setForm] = useState({
    preferred_language: preferences?.preferred_language || "en",
    preferred_currency: preferences?.preferred_currency || "INR",
    travel_style: preferences?.travel_style || "",
    accommodation_preferences: preferences?.accommodation_preferences || "",
    preferred_budget: preferences?.preferred_budget ?? "",
  });
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");

  async function saveProfile(event) {
    event.preventDefault();
    setError("");
    setNotice("");
    try {
      await updateProfile({ name });
      await updatePreferences({
        ...form,
        preferred_budget: form.preferred_budget === "" ? null : Number(form.preferred_budget),
      });
      setNotice("Your profile and travel preferences are saved.");
    } catch (saveError) {
      setError(saveError.message);
    }
  }

  return (
    <section className="account-card account-page profile-card">
      <p className="kicker">YOUR SETTINGS</p>
      <h1>Profile & preferences</h1>
      <p className="muted">These details help tailor future trip ideas. You can change them anytime.</p>
      {notice && <div className="success-notice" role="status">{notice}</div>}
      {error && <div className="error-notice" role="alert">{error}</div>}
      <form className="profile-form" onSubmit={saveProfile}>
        <label>
          Full name
          <input required maxLength={120} value={name} onChange={(event) => setName(event.target.value)} />
        </label>
        <label>
          Email
          <input type="email" value={user?.email || ""} disabled />
        </label>
        <label>
          Preferred language
          <select value={form.preferred_language} onChange={(event) => setForm({ ...form, preferred_language: event.target.value })}>
            <option value="en">English</option>
            <option value="hi">Hindi</option>
            <option value="hinglish">Hinglish</option>
          </select>
        </label>
        <label>
          Preferred currency
          <input required minLength={3} maxLength={8} value={form.preferred_currency} onChange={(event) => setForm({ ...form, preferred_currency: event.target.value.toUpperCase() })} />
        </label>
        <label>
          Travel style
          <select value={form.travel_style} onChange={(event) => setForm({ ...form, travel_style: event.target.value })}>
            <option value="">No preference</option>
            <option value="budget">Budget</option>
            <option value="adventure">Adventure</option>
            <option value="family">Family</option>
            <option value="luxury">Luxury</option>
            <option value="culture">Culture</option>
          </select>
        </label>
        <label>
          Preferred budget
          <input type="number" min="0" step="100" value={form.preferred_budget} onChange={(event) => setForm({ ...form, preferred_budget: event.target.value })} placeholder="Optional amount" />
        </label>
        <label className="profile-wide">
          Accommodation preferences
          <textarea maxLength={2000} value={form.accommodation_preferences} onChange={(event) => setForm({ ...form, accommodation_preferences: event.target.value })} placeholder="For example: quiet area, family room, near public transport" />
        </label>
        <button className="primary-button" type="submit">Save preferences</button>
      </form>
    </section>
  );
}
