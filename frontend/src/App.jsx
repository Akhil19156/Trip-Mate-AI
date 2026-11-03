import { useCallback, useEffect, useRef, useState } from "react";
import { marked } from "marked";
import DOMPurify from "dompurify";
import html2pdf from "html2pdf.js";
import { BrowserRouter, Navigate, Route, Routes, useSearchParams } from "react-router-dom";
import { apiRequest, AuthProvider, responseError, useAuth } from "./auth/AuthContext";
import {
  AppShell,
  AuthPage,
  DashboardPage,
  ProfilePage,
  ProtectedRoute,
  TripDetailPage,
  TripsPage,
} from "./auth/Pages";

const AGENT_LABELS = {
  flight_agent: "Flights",
  hotel_agent: "Stays",
  weather_agent: "Weather",
  budget_agent: "Budget",
  itinerary_agent: "Itinerary",
};

const QUICK_PROMPTS = [
  {
    label: "Jaipur · 3 days",
    text: "Plan a 3-day trip to Jaipur from Delhi with flights, stay options, weather, attractions and a budget of ₹15,000.",
  },
  {
    label: "Goa · 4 days",
    text: "Plan a 4-day trip to Goa from Delhi with flights, stay options, weather, attractions and a budget of ₹25,000.",
  },
  {
    label: "Mumbai · 3 days",
    text: "Plan a 3-day trip to Mumbai from Delhi with flights, stay options, weather, attractions and a budget of ₹20,000.",
  },
  {
    label: "Varanasi · 3 days",
    text: "Plan a 3-day trip to Varanasi from Delhi with flights, stay options, weather, attractions and a budget of ₹15,000.",
  },
];

function Markdown({ value }) {
  const html = DOMPurify.sanitize(marked.parse(value || ""));
  return (
    <div className="markdown-body" dangerouslySetInnerHTML={{ __html: html }} />
  );
}

async function readApiResponse(response) {
  const body = await response.text();
  if (!body) {
    throw new Error(
      "The server returned an empty response. Check that the TripMate backend is running.",
    );
  }

  try {
    return JSON.parse(body);
  } catch {
    throw new Error(
      "The server returned an invalid response. Check that the TripMate backend is running.",
    );
  }
}

function TravelPlanner() {
  const { preferences } = useAuth();
  const [searchParams, setSearchParams] = useSearchParams();
  const [message, setMessage] = useState("");
  const [threadId, setThreadId] = useState(null);
  const [workflow, setWorkflow] = useState(null);
  const [answer, setAnswer] = useState("");
  const [isDraft, setIsDraft] = useState(false);
  const [waitingForApproval, setWaitingForApproval] = useState(false);
  const [feedback, setFeedback] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [copyLabel, setCopyLabel] = useState("Copy plan");
  const [showDetails, setShowDetails] = useState(false);
  const [inputMode, setInputMode] = useState("text");
  const [selectedLanguage, setSelectedLanguage] = useState(
    preferences?.preferred_language || "en",
  );
  const [recognizedText, setRecognizedText] = useState("");
  const [recognizedLanguage, setRecognizedLanguage] = useState("");
  const [listening, setListening] = useState(false);
  const [recording, setRecording] = useState(false);
  const [voiceSessionActive, setVoiceSessionActive] = useState(false);
  const [speechStatus, setSpeechStatus] = useState("idle");
  const resultRef = useRef(null);
  const recognitionRef = useRef(null);
  const recorderRef = useRef(null);
  const recordingChunksRef = useRef([]);
  const recordingTargetRef = useRef("message");
  const voiceModeRef = useRef("text");
  const voiceSessionRef = useRef(false);
  const loadingRef = useRef(false);
  const languageRef = useRef("en");
  const threadIdRef = useRef(threadId);
  const approvalPendingRef = useRef(false);
  const listeningRef = useRef(false);
  const recordingRef = useRef(false);
  const speakTextRef = useRef(null);
  const resumedTripId = searchParams.get("trip_id");

  useEffect(() => {
    voiceModeRef.current = inputMode;
    voiceSessionRef.current = voiceSessionActive;
    loadingRef.current = loading;
    languageRef.current = selectedLanguage;
    threadIdRef.current = threadId;
    approvalPendingRef.current = waitingForApproval;
  }, [
    inputMode,
    voiceSessionActive,
    loading,
    selectedLanguage,
    threadId,
    waitingForApproval,
  ]);

  useEffect(
    () => () => {
      recognitionRef.current?.abort();
      const recorder = recorderRef.current;
      if (recorder?.state === "recording") {
        recorder.onstop = null;
        recorder.stream.getTracks().forEach((track) => track.stop());
        recorder.stop();
      }
      window.speechSynthesis?.cancel();
    },
    [],
  );

  useEffect(() => {
    if (answer && resultRef.current) {
      resultRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [answer]);

  function choosePrompt(text) {
    setMessage(text);
    setError("");
  }

  function chooseInputMode(mode) {
    if (mode !== "conversation" && voiceSessionRef.current) {
      voiceSessionRef.current = false;
      setVoiceSessionActive(false);
      stopListening();
      window.speechSynthesis?.cancel();
      setSpeechStatus("idle");
    }
    setInputMode(mode);
  }

  function stopListening() {
    recognitionRef.current?.stop();
    if (recorderRef.current?.state === "recording") recorderRef.current.stop();
    listeningRef.current = false;
    recordingRef.current = false;
    setListening(false);
    setRecording(false);
  }

  function speakText(text) {
    if (!("speechSynthesis" in window)) {
      setError("Spoken playback is not supported in this browser.");
      return;
    }
    window.speechSynthesis.cancel();
    const plainText = String(text || "")
      .replace(/```[\s\S]*?```/g, " ")
      .replace(/[#*_`>|-]/g, " ")
      .replace(/\s+/g, " ")
      .trim();
    const chunks = plainText.match(/[^.!?।]{1,700}(?:[.!?।]+|$)/g) || [plainText];
    const voices = window.speechSynthesis.getVoices();
    const preferredLanguage = languageRef.current;
    const voiceLanguage =
      preferredLanguage === "hi" || preferredLanguage === "hinglish"
        ? "hi"
        : preferredLanguage === "auto"
          ? recognizedLanguage || "en"
          : preferredLanguage;
    const selectedVoice = voices.find((voice) =>
      voice.lang.toLowerCase().startsWith(voiceLanguage),
    );
    setSpeechStatus("speaking");
    const speakChunk = (index) => {
      if (index >= chunks.length) {
        setSpeechStatus("idle");
        if (voiceSessionRef.current) {
          window.setTimeout(() => startListening(), 500);
        }
        return;
      }
      const utterance = new SpeechSynthesisUtterance(chunks[index].trim());
      utterance.lang =
        voiceLanguage === "hi"
          ? "hi-IN"
          : voiceLanguage === "en"
            ? "en-US"
            : voiceLanguage;
      if (selectedVoice) utterance.voice = selectedVoice;
      utterance.onend = () => speakChunk(index + 1);
      utterance.onerror = () => {
        setSpeechStatus("idle");
        setError("The selected browser voice couldn't play this response.");
      };
      window.speechSynthesis.speak(utterance);
    };
    speakChunk(0);
  }
  speakTextRef.current = speakText;

  function responseLanguage(detected = "", text = "") {
    if (languageRef.current !== "auto") return languageRef.current;
    const hasDevanagari = /[\u0900-\u097f]/.test(text);
    const hasLatin = /[a-z]/i.test(text);
    if (hasDevanagari && hasLatin) return "hinglish";
    if (hasDevanagari) return "hi";
    return detected.toLowerCase().startsWith("hi") ? "hi" : "en";
  }

  async function submitRecognizedText(text, detected = "") {
    const cleanText = text.trim();
    if (!cleanText) return;
    if (approvalPendingRef.current) {
      const approves =
        /^(yes|yeah|yep|approve|approved|looks good|haan|han|ji haan|okay|ठीक है|हाँ|हां|जी हाँ|स्वीकार)([,.! ]|$)/i.test(
          cleanText,
        );
      await submitApproval(
        approves,
        approves ? "" : cleanText.toLowerCase() === "no" ? "Please revise the draft." : cleanText,
        responseLanguage(detected, cleanText),
      );
      return;
    }
    await sendMessage(cleanText, responseLanguage(detected, cleanText));
  }

  async function transcribeAudio(blob, target = "message") {
    if (!blob?.size) {
      setError("The recording was empty. Please try speaking again.");
      return;
    }
    const extension = blob.type.includes("ogg")
      ? "ogg"
      : blob.type.includes("mp4")
        ? "m4a"
        : blob.type.includes("mpeg")
          ? "mp3"
          : "webm";
    const formData = new FormData();
    formData.append("audio", blob, `voice-request.${extension}`);
    formData.append("language", languageRef.current);
    setLoading(true);
    setError("");
    try {
      const response = await apiRequest("/api/voice/transcribe", {
        method: "POST",
        body: formData,
      });
      const data = await readApiResponse(response);
      if (!response.ok)
        throw new Error(data.detail || "Voice transcription is unavailable.");
      const text = data.text.trim();
      if (target === "feedback") {
        setFeedback((previous) =>
          previous ? `${previous.trim()} ${text}` : text,
        );
        if (languageRef.current === "auto") {
          setSelectedLanguage(responseLanguage(data.detected_language || "", text));
        }
      } else {
        setMessage(text);
      }
      setRecognizedText(text);
      setRecognizedLanguage(data.detected_language || "");
      if (target === "feedback") {
        return;
      }
      if (voiceModeRef.current === "conversation") {
        await submitRecognizedText(text, data.detected_language || "");
      }
    } catch (err) {
      setError(
        err.message ||
          "Couldn't transcribe this recording. Check microphone access or type your request.",
      );
    } finally {
      setLoading(false);
    }
  }

  function startRecordingFallback() {
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      setError(
        "Speech recognition is not supported here. Try a current Chrome or Edge browser, or type your request.",
      );
      return;
    }
    navigator.mediaDevices
      .getUserMedia({ audio: true })
      .then((stream) => {
        const preferredType = [
          "audio/webm;codecs=opus",
          "audio/ogg;codecs=opus",
          "audio/mp4",
        ].find((type) => MediaRecorder.isTypeSupported(type));
        const recorder = new MediaRecorder(
          stream,
          preferredType ? { mimeType: preferredType } : undefined,
        );
        recordingChunksRef.current = [];
        recorderRef.current = recorder;
        recorder.ondataavailable = (event) => {
          if (event.data.size) recordingChunksRef.current.push(event.data);
        };
        recorder.onerror = () => {
          stream.getTracks().forEach((track) => track.stop());
          recordingRef.current = false;
          listeningRef.current = false;
          setRecording(false);
          setListening(false);
          setError("The microphone recording failed. Please try again.");
        };
        recorder.onstop = () => {
          const blob = new Blob(recordingChunksRef.current, {
            type: recorder.mimeType || "audio/webm",
          });
          stream.getTracks().forEach((track) => track.stop());
          recordingRef.current = false;
          listeningRef.current = false;
          setRecording(false);
          setListening(false);
          transcribeAudio(blob, recordingTargetRef.current);
        };
        recorder.start();
        recordingRef.current = true;
        listeningRef.current = true;
        setRecording(true);
        setListening(true);
        setError("");
      })
      .catch(() => {
        setError(
          "Microphone access was denied or unavailable. Allow microphone access in your browser settings, or type your request.",
        );
      });
  }

  function startListening(target = "message") {
    if (loadingRef.current || recordingRef.current || listeningRef.current) return;
    recordingTargetRef.current = target;
    const Recognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (languageRef.current === "auto" || !Recognition) {
      startRecordingFallback();
      return;
    }
    try {
      const recognition = new Recognition();
      recognitionRef.current = recognition;
      recognition.lang =
        languageRef.current === "hi" || languageRef.current === "hinglish"
          ? "hi-IN"
          : "en-US";
      recognition.continuous = false;
      recognition.interimResults = true;
      let finalText = "";
      recognition.onresult = (event) => {
        let interimText = "";
        for (let index = event.resultIndex; index < event.results.length; index += 1) {
          const result = event.results[index];
          if (result.isFinal) finalText += `${result[0].transcript} `;
          else interimText += result[0].transcript;
        }
        const preview = `${finalText}${interimText}`.trim();
        setRecognizedText(preview);
        if (target === "feedback") {
          if (finalText.trim()) {
            setFeedback((previous) => {
              const prefix = previous.trim();
              const spoken = finalText.trim();
              return prefix ? `${prefix} ${spoken}` : spoken;
            });
          }
        } else {
          setMessage(preview);
        }
        if (finalText.trim()) {
          recognition.stop();
          listeningRef.current = false;
          setListening(false);
          if (target === "message" && voiceModeRef.current === "conversation") {
            submitRecognizedText(finalText);
          }
        }
      };
      recognition.onerror = (event) => {
        listeningRef.current = false;
        setListening(false);
        if (event.error === "not-allowed" || event.error === "service-not-allowed") {
          if (event.error === "service-not-allowed") {
            setError("Browser speech recognition is unavailable. Switching to audio-upload transcription.");
            startRecordingFallback();
          } else {
            setError(
              "Microphone access was denied. Allow microphone access in your browser settings, or type your request.",
            );
          }

        } else if (event.error !== "no-speech" && event.error !== "aborted") {
          if (event.error === "network") {
            setError("Browser speech recognition lost its connection. Switching to audio-upload transcription.");
            startRecordingFallback();
          } else {
            setError(
              "Speech recognition failed. Try again, use the audio-upload fallback, or type your request.",
            );
          }
        }
      };
      recognition.onend = () => {
        listeningRef.current = false;
        setListening(false);
      };
      recognition.start();
      listeningRef.current = true;
      setListening(true);
      setError("");
    } catch {
      setError("Speech recognition couldn't start. Check microphone permissions.");
    }
  }

  function toggleVoiceSession() {
    if (voiceSessionActive) {
      setVoiceSessionActive(false);
      voiceSessionRef.current = false;
      stopListening();
      window.speechSynthesis?.cancel();
      setSpeechStatus("idle");
      return;
    }
    setInputMode("conversation");
    setVoiceSessionActive(true);
    voiceSessionRef.current = true;
    startListening();
  }

  const applyResponse = useCallback((data, draftMode = false) => {
    const responseThreadId = data.thread_id || threadIdRef.current;
    setThreadId(responseThreadId);
    threadIdRef.current = responseThreadId;
    if (responseThreadId)
      localStorage.setItem("travel_thread_id", responseThreadId);
    setWorkflow(data);
    setAnswer(
      data.requires_approval
        ? data.itinerary || data.answer || ""
        : data.answer || "",
    );
    setIsDraft(draftMode || Boolean(data.requires_approval));
    setWaitingForApproval(Boolean(data.requires_approval));
    approvalPendingRef.current = Boolean(data.requires_approval);
    setError("");
    if (voiceSessionRef.current) {
      stopListening();
      speakTextRef.current?.(
        data.requires_approval
          ? data.itinerary || data.answer || ""
          : data.answer || "",
      );
    }
  }, []);

  useEffect(() => {
    if (!resumedTripId) return;
    let active = true;
    async function restoreTrip() {
      setLoading(true);
      setError("");
      try {
        const resume = await responseError(
          await apiRequest(`/api/trips/${resumedTripId}/resume`, {
            method: "POST",
          }),
          "This trip cannot be resumed.",
        );
        if (!active) return;
        const trip = resume.trip;
        const restored = {
          trip_id: trip.id,
          thread_id: trip.thread_id,
          requires_approval: resume.requires_approval,
          itinerary: trip.itinerary,
          answer: trip.itinerary,
          preferred_language: preferences?.preferred_language || "en",
          selected_agents: [],
        };
        applyResponse(restored, resume.requires_approval);
        setSearchParams({}, { replace: true });
      } catch (restoreError) {
        if (active) setError(restoreError.message);
      } finally {
        if (active) setLoading(false);
      }
    }
    restoreTrip();
    return () => {
      active = false;
    };
  }, [
    applyResponse,
    preferences?.preferred_language,
    resumedTripId,
    setSearchParams,
  ]);

  async function sendMessage(messageOverride = message, languageOverride = selectedLanguage) {
    setError("");
    if (approvalPendingRef.current) {
      setError(
        "Finish reviewing the current draft before starting another trip.",
      );
      return;
    }
    const cleanMessage = messageOverride.trim();
    if (!cleanMessage) {
      setError(
        "Tell us where you want to go and what kind of trip you have in mind.",
      );
      return;
    }
    const requestLanguage =
      languageOverride === "auto"
        ? responseLanguage("", cleanMessage)
        : languageOverride;

    setLoading(true);
    try {
      console.log("api hit");
      const response = await apiRequest("/api/travel", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: cleanMessage,
          thread_id: threadId,
          language: requestLanguage,
        }),
      });
      console.log("api response");
      const data = await readApiResponse(response);
      if (!response.ok || !data.success)
        throw new Error(
          data.error || "The trip planner couldn't complete that request.",
        );
      applyResponse(data, Boolean(data.requires_approval));
    } catch (err) {
      setError(
        err.message ||
          "Couldn't reach the server. Check that the backend is running.",
      );
    } finally {
      setLoading(false);
    }
  }

  async function submitApproval(
    approved,
    feedbackOverride = feedback,
    languageOverride = selectedLanguage,
  ) {
    const approvalLanguage =
      languageOverride === "auto"
        ? workflow?.preferred_language || responseLanguage("", feedbackOverride)
        : languageOverride;
    setError("");
    if (!threadIdRef.current || !approvalPendingRef.current) {
      setError("There isn't a draft waiting for review.");
      return;
    }
    if (!approved && !feedbackOverride.trim()) {
      setError(
        "Add a note about what you'd like changed before requesting a revision.",
      );
      return;
    }

    setLoading(true);
    try {
      const response = await apiRequest("/api/travel/approve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          thread_id: threadIdRef.current,
          approved,
          feedback: feedbackOverride.trim(),
          language: approvalLanguage,
        }),
      });
      const data = await readApiResponse(response);
      if (!response.ok || !data.success)
        throw new Error(data.error || "The review couldn't be submitted.");
      applyResponse(data, false);
      setWaitingForApproval(false);
      setFeedback("");
    } catch (err) {
      setError(err.message || "Couldn't resume the travel workflow.");
    } finally {
      setLoading(false);
    }
  }

  async function copyPlan() {
    if (!answer) return;
    try {
      await navigator.clipboard.writeText(answer);
      setCopyLabel("Copied");
      window.setTimeout(() => setCopyLabel("Copy plan"), 1500);
    } catch {
      setError(
        "Copying wasn't available in this browser. You can select the plan text and copy it manually.",
      );
    }
  }

  async function downloadPlan() {
    const content = document.getElementById("printable-plan");
    if (!content || !answer) {
      setError("Generate a plan first, then you can save it as a PDF.");
      return;
    }
    setLoading(true);
    try {
      await html2pdf()
        .set({
          margin: [12, 12, 14, 12],
          filename: "tripmate-travel-plan.pdf",
          image: { type: "jpeg", quality: 0.96 },
          html2canvas: { scale: 2, useCORS: true, backgroundColor: "#ffffff" },
          jsPDF: { unit: "mm", format: "a4", orientation: "portrait" },
          pagebreak: { mode: ["css", "legacy"] },
        })
        .from(content)
        .save();
    } catch {
      setError(
        "The PDF couldn't be created. Try using your browser's Print option instead.",
      );
    } finally {
      setLoading(false);
    }
  }

  function togglePlayback() {
    if (speechStatus === "paused") {
      window.speechSynthesis?.resume();
      setSpeechStatus("speaking");
      return;
    }
    if (speechStatus === "speaking") {
      window.speechSynthesis?.pause();
      setSpeechStatus("paused");
      return;
    }
    speakText(answer);
  }

  const selectedAgents = workflow?.selected_agents || [];
  const liveEvaluation = workflow?.evaluation;

  return (
    <div className="site-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="TripMate home">
          <span className="brand-mark">t.</span>
          <span>
            tripmate<span className="brand-ai"> ai</span>
          </span>
        </a>
        <div className="topbar-note">
          <span className="status-dot" /> Your trip-planning desk
        </div>
      </header>

      <main className="main-wrap">
        <section className="intro">
          <div className="intro-copy">
            <p className="kicker">YOUR NEXT TRIP, MADE EASIER</p>
            <h1>
              Somewhere new
              <br />
              <em>starts here.</em>
            </h1>
            <p className="intro-text">
              Tell us what you have in mind. TripMate can pull together the
              route, places to stay, weather notes and a realistic budget in one
              plan.
            </p>
            <div className="intro-foot">
              <span className="tiny-star">✳</span> A good trip starts with a
              rough idea.
            </div>
          </div>
          <div
            className="travel-art"
            aria-label="Illustration of a sun over a mountain landscape"
          >
            <div className="sun" />
            <div className="ridge ridge-back" />
            <div className="ridge ridge-front" />
            <div className="art-stamp">
              GO
              <br />
              YOUR
              <br />
              OWN WAY
            </div>
            <span className="art-caption">THE WORLD IS STILL OUT THERE</span>
          </div>
        </section>

        <section className="planner-panel">
          <div className="panel-heading">
            <div>
              <p className="kicker">01 / YOUR IDEA</p>
              <h2>Where are we going?</h2>
              <p className="muted">
                A destination, a budget, a few days off — start anywhere.
              </p>
            </div>
            <span className="step-number">01</span>
          </div>

          <div className="voice-options" aria-label="Choose how to interact">
            <button
              type="button"
              className={inputMode === "text" ? "voice-mode active" : "voice-mode"}
              aria-pressed={inputMode === "text"}
              onClick={() => chooseInputMode("text")}
            >
              <strong>Type your message</strong>
              <span>Use the existing trip planner chat.</span>
            </button>
            <button
              type="button"
              className={inputMode === "review" ? "voice-mode active" : "voice-mode"}
              aria-pressed={inputMode === "review"}
              onClick={() => chooseInputMode("review")}
            >
              <strong>Speak and review</strong>
              <span>Edit the transcript before sending.</span>
            </button>
            <button
              type="button"
              className={
                inputMode === "conversation" ? "voice-mode active" : "voice-mode"
              }
              aria-pressed={inputMode === "conversation"}
              onClick={() => chooseInputMode("conversation")}
            >
              <strong>Talk directly</strong>
              <span>Speak, hear the response, and continue hands-free.</span>
            </button>
          </div>
          <div className="voice-toolbar">
            <label htmlFor="voice-language">Response and recognition language</label>
            <select
              id="voice-language"
              value={selectedLanguage}
              onChange={(event) => setSelectedLanguage(event.target.value)}
              disabled={loading || listening || recording}
            >
              <option value="en">English</option>
              <option value="hi">Hindi</option>
              <option value="hinglish">Hinglish</option>
              <option value="auto">Auto-detect (audio upload)</option>
            </select>
            {inputMode === "review" && (
              <button
                type="button"
                className="voice-control"
                onClick={() => (listening || recording ? stopListening() : startListening())}
                disabled={loading}
                aria-pressed={listening || recording}
              >
                {recording
                  ? "Stop recording"
                  : listening
                    ? "Stop listening"
                    : "Start microphone"}
              </button>
            )}
            {inputMode === "conversation" && (
              <button
                type="button"
                className="voice-control"
                onClick={toggleVoiceSession}
                disabled={loading && !voiceSessionActive}
                aria-pressed={voiceSessionActive}
              >
                {voiceSessionActive
                  ? "Stop voice conversation"
                  : "Start voice conversation"}
              </button>
            )}
            {(listening || recording) && (
              <span className="listening-indicator" role="status">
                <span className="listening-pulse" />
                {recording ? "Recording audio…" : "Listening…"}
              </span>
            )}
          </div>
          {inputMode !== "text" && (
            <p className="voice-privacy-note">
              Browser speech recognition may use your browser vendor's speech service.
              Audio-upload transcription sends the recording to Groq; TripMate does not save it.
            </p>
          )}
          {recognizedText && (
            <div className="transcript-preview" aria-live="polite">
              <strong>
                Recognized speech
                {recognizedLanguage ? ` · ${recognizedLanguage}` : ""}
              </strong>
              <span>{recognizedText}</span>
              <small>Edit the text above before sending if needed.</small>
            </div>
          )}

          <label className="sr-only" htmlFor="trip-request">
            Describe your trip
          </label>
          <textarea
            id="trip-request"
            value={message}
            onChange={(event) => setMessage(event.target.value)}
            placeholder="For example: A week in Japan from Delhi, good food, a mix of city and nature, under ₹2 lakh..."
            onKeyDown={(event) => {
              if (
                event.key === "Enter" &&
                !event.shiftKey &&
                !event.isComposing
              ) {
                event.preventDefault();
                if (!loading) sendMessage();
              }
            }}
            maxLength={2000}
          />
          <div className="composer-bottom">
            <button
              type="button"
              className="primary-button"
              onClick={() => sendMessage()}
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className="spinner" /> Working on it
                </>
              ) : (
                <>
                  Build my trip <span aria-hidden="true">↗</span>
                </>
              )}
            </button>
          </div>

          <div className="prompt-row">
            <span className="prompt-label">TRY A TRIP</span>
            {QUICK_PROMPTS.map((prompt) => (
              <button
                className="prompt-chip"
                key={prompt.label}
                onClick={() => choosePrompt(prompt.text)}
              >
                {prompt.label}
              </button>
            ))}
          </div>
        </section>

        {error && (
          <div className="error-notice" role="alert">
            <span className="error-symbol">!</span>
            <span>{error}</span>
            <button onClick={() => setError("")} aria-label="Dismiss error">
              ×
            </button>
          </div>
        )}

        {workflow && (
          <section className="workflow-panel">
            <div className="section-title-row">
              <div>
                <p className="kicker">02 / HOW IT'S PUT TOGETHER</p>
                <h2>Trip planning in progress</h2>
              </div>
              <button
                className="text-button"
                onClick={() => setShowDetails((value) => !value)}
              >
                {showDetails ? "Hide details −" : "View details +"}
              </button>
            </div>
            <div className="agent-track">
              {selectedAgents.map((agent, index) => (
                <div className="agent-step" key={agent}>
                  <span className="agent-index">
                    {String(index + 1).padStart(2, "0")}
                  </span>
                  <span>
                    {AGENT_LABELS[agent] || agent.replaceAll("_", " ")}
                  </span>
                </div>
              ))}
              {selectedAgents.length === 0 && (
                <span className="muted">
                  No specialist agents were selected for this request.
                </span>
              )}
            </div>
            {liveEvaluation && (
              <div className="live-evaluation">
                <div className="live-evaluation-heading">
                  <strong>Live workflow checks</strong>
                  <span
                    className={
                      liveEvaluation.passed
                        ? "evaluation-score passed"
                        : "evaluation-score failed"
                    }
                  >
                    {liveEvaluation.score == null
                      ? "No score"
                      : `${liveEvaluation.score}% · ${
                          liveEvaluation.passed ? "Passed" : "Review needed"
                        }`}
                  </span>
                </div>
                <ul>
                  {liveEvaluation.checks.map((check) => (
                    <li key={check.name} className={`evaluation-${check.status}`}>
                      <span>
                        {check.status === "passed"
                          ? "✓"
                          : check.status === "failed"
                            ? "!"
                            : "–"}
                      </span>
                      <span>
                        <strong>{check.label}:</strong> {check.detail}
                      </span>
                    </li>
                  ))}
                </ul>
                <p>{liveEvaluation.note}</p>
              </div>
            )}
            {showDetails && (
              <div className="workflow-details">
                <div>
                  <span className="detail-label">Input check</span>
                  <span
                    className={
                      workflow.guardrail_allowed === false
                        ? "guardrail blocked"
                        : "guardrail"
                    }
                  >
                    {workflow.guardrail_allowed === false
                      ? "Needs a travel-related request"
                      : "Passed"}
                  </span>
                </div>
                <div>
                  <span className="detail-label">Supervisor note</span>
                  <p>
                    {workflow.supervisor_reasoning ||
                      "The supervisor finished routing the request."}
                  </p>
                </div>
                {threadId && (
                  <div>
                    <span className="detail-label">Thread reference</span>
                    <code>{threadId}</code>
                  </div>
                )}
              </div>
            )}
          </section>
        )}

        {answer && (
          <section className="result-panel" ref={resultRef}>
            <div className="result-top">
              <div>
                <p className="kicker">03 / THE PLAN</p>
                <h2>
                  {isDraft
                    ? "First draft, ready for a look"
                    : "Your trip, mapped out"}
                </h2>
                <p className="muted">
                  {isDraft
                    ? "Check the details and tell us what you'd change."
                    : "Keep this handy while you make the arrangements."}
                </p>
              </div>
              <div className="result-buttons">
                <button
                  className="secondary-button"
                  onClick={togglePlayback}
                  disabled={!answer}
                >
                  {speechStatus === "paused"
                    ? "Resume audio"
                    : speechStatus === "speaking"
                      ? "Pause audio"
                      : "Play response"}
                </button>
                <button
                  className="secondary-button"
                  onClick={() => {
                    window.speechSynthesis?.cancel();
                    setSpeechStatus("idle");
                  }}
                  disabled={speechStatus === "idle"}
                >
                  Stop audio
                </button>
                <button className="secondary-button" onClick={copyPlan}>
                  {copyLabel}
                </button>
                <button
                  className="dark-button"
                  onClick={downloadPlan}
                  disabled={loading}
                >
                  Save as PDF ↓
                </button>
              </div>
            </div>
            <div className="plan-paper" id="printable-plan">
              <div className="paper-head">
                <span className="paper-logo">
                  tripmate <b>ai</b>
                </span>
                <span>YOUR TRAVEL NOTES</span>
              </div>
              <Markdown value={answer} />
              <div className="paper-foot">
                Made for your next adventure · TripMate AI
              </div>
            </div>
            {threadId && (
              <p className="thread-note">
                Saved session reference: <code>{threadId}</code>
              </p>
            )}
          </section>
        )}

        {waitingForApproval && (
          <section className="review-panel">
            <div className="review-icon">✎</div>
            <div className="review-main">
              <p className="kicker">04 / YOUR CALL</p>
              <h2>Before we call it a plan…</h2>
              <p className="muted">
                Take a look at the draft above. Approve it as-is, or leave a
                note and we'll revise it.
              </p>
              <label htmlFor="review-feedback">
                What would you like to change?
              </label>
              <textarea
                id="review-feedback"
                className="feedback-box"
                value={feedback}
                onChange={(event) => setFeedback(event.target.value)}
                onKeyDown={(event) => {
                  if (
                    event.key === "Enter" &&
                    !event.shiftKey &&
                    !event.isComposing
                  ) {
                    event.preventDefault();
                    if (!loading) submitApproval(false);
                  }
                }}
                maxLength={1000}
                placeholder="Maybe a slower pace, cheaper stays, more time outdoors..."
              />
              <div className="feedback-voice-row">
                <button
                  type="button"
                  className="voice-control"
                  onClick={() =>
                    listening || recording
                      ? stopListening()
                      : startListening("feedback")
                  }
                  disabled={loading}
                  aria-pressed={listening || recording}
                >
                  {recording
                    ? "Stop feedback recording"
                    : listening
                      ? "Stop feedback listening"
                      : "Speak feedback"}
                </button>
                {(listening || recording) && (
                  <span className="listening-indicator" role="status">
                    <span className="listening-pulse" />
                    {recording ? "Recording feedback…" : "Listening for feedback…"}
                  </span>
                )}
                <span className="feedback-language">
                  {selectedLanguage === "hi"
                    ? "Hindi"
                    : selectedLanguage === "hinglish"
                      ? "Hinglish"
                      : selectedLanguage === "auto"
                        ? "Auto-detect"
                        : "English"}
                </span>
              </div>
              <div className="review-actions">
                <button
                  type="button"
                  className="approve-button"
                  onClick={() => submitApproval(true)}
                  disabled={loading}
                >
                  {loading ? "Sending…" : "Looks good — finish plan"}
                </button>
                <button
                  type="button"
                  className="revise-button"
                  onClick={() => submitApproval(false)}
                  disabled={loading}
                >
                  Ask for changes
                </button>
              </div>
            </div>
          </section>
        )}
      </main>

      <footer className="site-footer">
        <a className="brand footer-brand" href="/">
          <span className="brand-mark">t.</span>
          <span>
            tripmate<span className="brand-ai"> ai</span>
          </span>
        </a>
        <span>Plan thoughtfully. Travel curiously.</span>
      </footer>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<AuthPage mode="login" />} />
          <Route path="/signup" element={<AuthPage mode="signup" />} />
          <Route element={<ProtectedRoute />}>
            <Route element={<AppShell />}>
              <Route index element={<DashboardPage />} />
              <Route path="plan" element={<TravelPlanner />} />
              <Route path="trips" element={<TripsPage />} />
              <Route path="trips/:tripId" element={<TripDetailPage />} />
              <Route path="profile" element={<ProfilePage />} />
            </Route>
          </Route>
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}
