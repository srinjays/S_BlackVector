/**
 * useVoiceInput — Reusable hook for browser-based speech-to-text
 *
 * Uses the Web Speech API (SpeechRecognition / webkitSpeechRecognition).
 * Exposes a clean interface for any search bar to consume.
 *
 * Known Chrome quirks handled:
 *   - Recognition can silently end immediately after start (no error).
 *     We auto-restart up to MAX_RESTARTS times.
 *   - Releasing the getUserMedia stream before recognition.start() can
 *     cause Chrome to lose the audio source.  We keep the stream alive
 *     for the entire listening session.
 *   - The restart counter resets on each successful result, so long
 *     dictation sessions don't exhaust restarts.
 *
 * TODO: To switch to a paid API (OpenAI Whisper, Deepgram, Google Cloud Speech),
 *       replace the SpeechRecognition logic inside startListening() with a
 *       MediaRecorder → WebSocket/REST call, and keep the same public interface.
 */

import { useState, useRef, useCallback, useEffect } from 'react'

/* ─────────────────────────────────────────────────────────
   Types
───────────────────────────────────────────────────────── */
interface SpeechRecognitionEvent {
  results: SpeechRecognitionResultList
  resultIndex: number
}

interface SpeechRecognitionErrorEvent {
  error: string
  message?: string
}

type SpeechRecognitionInstance = {
  continuous: boolean
  interimResults: boolean
  lang: string
  start: () => void
  stop: () => void
  abort: () => void
  onresult: ((event: SpeechRecognitionEvent) => void) | null
  onerror: ((event: SpeechRecognitionErrorEvent) => void) | null
  onend: (() => void) | null
  onstart: (() => void) | null
}

export interface UseVoiceInputReturn {
  /** Whether the browser supports Web Speech API */
  isSupported: boolean
  /** Whether the mic is currently listening */
  isListening: boolean
  /** Final transcribed text (accumulated since last clearTranscript) */
  transcript: string
  /** Interim (in-progress) transcript shown in lighter text */
  interimTranscript: string
  /** Last error message, if any */
  error: string | null
  /** Start listening — requests mic permission */
  startListening: () => void
  /** Stop listening immediately */
  stopListening: () => void
  /** Toggle listening on/off */
  toggleListening: () => void
  /** Clear transcript and error state, stop listening */
  reset: () => void
  /** Clear transcript only (without stopping listening) */
  clearTranscript: () => void
}

/* ─────────────────────────────────────────────────────────
   Browser detection
───────────────────────────────────────────────────────── */
function getSpeechRecognitionCtor(): (new () => SpeechRecognitionInstance) | null {
  if (typeof window === 'undefined') return null
  const w = window as any
  return w.SpeechRecognition || w.webkitSpeechRecognition || null
}

/* ─────────────────────────────────────────────────────────
   Constants
───────────────────────────────────────────────────────── */
const MAX_RESTARTS = 5          // max consecutive auto-restarts without results
const SILENCE_TIMEOUT_MS = 8000 // auto-stop after 8s of silence
const MIN_SESSION_MS = 500      // if session lasts < this, it's "premature"

/* ─────────────────────────────────────────────────────────
   Friendly error messages
───────────────────────────────────────────────────────── */
function getUserFriendlyError(errorCode: string): string {
  switch (errorCode) {
    case 'not-allowed':
    case 'permission-denied':
      return 'Microphone access denied. Please allow mic permissions in your browser settings.'
    case 'no-speech':
      return 'No speech was detected. Please try speaking again.'
    case 'network':
      return 'Network error — speech recognition requires an internet connection (Chrome sends audio to Google servers).'
    case 'audio-capture':
      return 'No microphone detected. Please connect a microphone.'
    case 'aborted':
      return '' // User-initiated stop — not a real error
    case 'service-not-allowed':
      return 'Speech recognition service is not allowed. This may be a browser policy restriction.'
    case 'language-not-supported':
      return 'The selected language is not supported for speech recognition.'
    default:
      return `Speech recognition error: ${errorCode}`
  }
}

function getMicErrorMessage(err: any): string {
  if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
    return 'Microphone access denied. Please allow mic permissions in your browser settings and reload.'
  }
  if (err.name === 'NotFoundError') {
    return 'No microphone found. Please connect a microphone and try again.'
  }
  if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
    return 'Microphone is in use by another application.'
  }
  if (err.name === 'OverconstrainedError') {
    return 'Microphone constraints could not be satisfied.'
  }
  if (err.name === 'SecurityError') {
    return 'Microphone access blocked by browser security policy. Ensure the page is served over HTTPS.'
  }
  return `Microphone error: ${err.message || err.name || 'Unknown error'}`
}

/* ─────────────────────────────────────────────────────────
   Hook
───────────────────────────────────────────────────────── */
export function useVoiceInput(lang = 'en-US'): UseVoiceInputReturn {
  const [isListening, setIsListening] = useState(false)
  const [transcript, setTranscript] = useState('')
  const [interimTranscript, setInterimTranscript] = useState('')
  const [error, setError] = useState<string | null>(null)

  const recognitionRef = useRef<SpeechRecognitionInstance | null>(null)
  const silenceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const micStreamRef = useRef<MediaStream | null>(null)
  const wantListeningRef = useRef(false)       // true when user wants mic on
  const restartCountRef = useRef(0)
  const sessionStartRef = useRef(0)            // timestamp when recognition started
  const gotResultRef = useRef(false)            // did we get any result this session?
  const hadErrorRef = useRef(false)             // did onerror fire this session?
  const isSupported = getSpeechRecognitionCtor() !== null

  // Use a ref for the restart function to avoid stale closure in onend
  const createAndStartRef = useRef<() => void>(() => {})

  /* ── Cleanup on unmount ── */
  useEffect(() => {
    return () => {
      if (silenceTimerRef.current !== null) {
        clearTimeout(silenceTimerRef.current)
        silenceTimerRef.current = null
      }
      wantListeningRef.current = false
      if (recognitionRef.current) {
        try { recognitionRef.current.abort() } catch { /* noop */ }
        recognitionRef.current = null
      }
      releaseMicStream()
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  /* ── Release mic stream helper ── */
  const releaseMicStream = useCallback(() => {
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach(t => t.stop())
      micStreamRef.current = null
      console.log('[VoiceInput] Mic stream released')
    }
  }, [])

  /* ── Reset silence timer — auto-stop after SILENCE_TIMEOUT_MS ── */
  const resetSilenceTimer = useCallback(() => {
    if (silenceTimerRef.current !== null) {
      clearTimeout(silenceTimerRef.current)
    }
    silenceTimerRef.current = setTimeout(() => {
      console.log('[VoiceInput] Silence timeout — stopping')
      wantListeningRef.current = false
      if (recognitionRef.current) {
        try { recognitionRef.current.stop() } catch { /* noop */ }
      }
    }, SILENCE_TIMEOUT_MS)
  }, [])

  /* ── Internal: create and start a SpeechRecognition instance ── */
  const createAndStartRecognition = useCallback(() => {
    const Ctor = getSpeechRecognitionCtor()
    if (!Ctor) {
      console.warn('[VoiceInput] SpeechRecognition not available')
      setError('Voice input not supported in this browser. Use Chrome or Edge.')
      setIsListening(false)
      wantListeningRef.current = false
      return
    }

    // Abort previous if exists
    if (recognitionRef.current) {
      try { recognitionRef.current.abort() } catch { /* noop */ }
      recognitionRef.current = null
    }

    const recognition = new Ctor()
    recognition.continuous = true
    recognition.interimResults = true
    recognition.lang = lang

    recognition.onstart = () => {
      console.log('[VoiceInput] Recognition started — listening')
      sessionStartRef.current = Date.now()
      gotResultRef.current = false
      hadErrorRef.current = false
      setIsListening(true)
      setError(null)
      resetSilenceTimer()
    }

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      gotResultRef.current = true
      // Reset restart counter on successful results — so long dictation
      // sessions don't exhaust the restart budget.
      restartCountRef.current = 0
      resetSilenceTimer()

      let finalText = ''
      let interimText = ''

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i]
        const text = result[0].transcript
        if (result.isFinal) {
          finalText += text
        } else {
          interimText += text
        }
      }

      if (finalText) {
        console.log('[VoiceInput] Final transcript:', finalText)
        setTranscript(prev => {
          const separator = prev && !prev.endsWith(' ') ? ' ' : ''
          return prev + separator + finalText
        })
      }
      if (interimText) {
        console.log('[VoiceInput] Interim:', interimText)
      }
      setInterimTranscript(interimText)
    }

    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      console.error('[VoiceInput] Recognition error:', event.error, event.message)
      hadErrorRef.current = true

      if (silenceTimerRef.current !== null) {
        clearTimeout(silenceTimerRef.current)
        silenceTimerRef.current = null
      }

      // no-speech is non-fatal — let onend handle restart
      if (event.error === 'no-speech') {
        console.log('[VoiceInput] No speech detected — will auto-restart if session is still wanted')
        hadErrorRef.current = false  // allow restart
        return // Don't set isListening to false, don't set error
      }

      // aborted is user-initiated — not a real error
      if (event.error === 'aborted') {
        console.log('[VoiceInput] Recognition aborted (user-initiated)')
        // Don't set error, just let onend clean up
        setIsListening(false)
        setInterimTranscript('')
        return
      }

      // All other errors are terminal for this listening session
      wantListeningRef.current = false
      const friendlyMessage = getUserFriendlyError(event.error)
      if (friendlyMessage) {
        setError(friendlyMessage)
      }
      setIsListening(false)
      setInterimTranscript('')
    }

    recognition.onend = () => {
      const sessionDuration = Date.now() - sessionStartRef.current
      console.log(
        `[VoiceInput] Recognition ended (session lasted ${sessionDuration}ms, ` +
        `gotResult=${gotResultRef.current}, hadError=${hadErrorRef.current}, ` +
        `wantListening=${wantListeningRef.current}, restarts=${restartCountRef.current})`
      )

      if (silenceTimerRef.current !== null) {
        clearTimeout(silenceTimerRef.current)
        silenceTimerRef.current = null
      }

      // Auto-restart if the session ended and user still wants to listen
      if (wantListeningRef.current && !hadErrorRef.current && restartCountRef.current < MAX_RESTARTS) {
        restartCountRef.current++
        const delay = sessionDuration < MIN_SESSION_MS ? 300 : 100
        console.log(`[VoiceInput] Auto-restarting (attempt ${restartCountRef.current}/${MAX_RESTARTS}) in ${delay}ms...`)
        setTimeout(() => {
          if (wantListeningRef.current) {
            // Use the ref to call the latest version — avoids stale closure
            createAndStartRef.current()
          }
        }, delay)
        return // Don't set isListening to false yet
      }

      // If we exhausted restarts without getting results, show an error
      if (wantListeningRef.current && restartCountRef.current >= MAX_RESTARTS && !gotResultRef.current) {
        console.error('[VoiceInput] Exhausted auto-restarts without getting results')
        setError('Voice recognition keeps stopping. Try refreshing the page, or check that your microphone is working.')
        wantListeningRef.current = false
      }

      setIsListening(false)
      setInterimTranscript('')
      // Release mic stream when fully stopped
      if (!wantListeningRef.current) {
        releaseMicStream()
      }
    }

    recognitionRef.current = recognition

    try {
      recognition.start()
      console.log('[VoiceInput] recognition.start() called successfully')
    } catch (err: any) {
      console.error('[VoiceInput] recognition.start() threw:', err)
      // Handle "already started" — can happen during rapid toggle
      if (err.message && err.message.includes('already started')) {
        console.warn('[VoiceInput] Recognition was already running, ignoring duplicate start')
        return
      }
      setError(err.message ?? 'Failed to start voice input')
      setIsListening(false)
      wantListeningRef.current = false
      releaseMicStream()
    }
  }, [lang, resetSilenceTimer, releaseMicStream])

  // Keep ref in sync so onend's setTimeout always calls the latest version
  useEffect(() => {
    createAndStartRef.current = createAndStartRecognition
  }, [createAndStartRecognition])

  /* ── Start ── */
  const startListening = useCallback(async () => {
    console.log('[VoiceInput] startListening called')
    setError(null)
    setInterimTranscript('')
    restartCountRef.current = 0

    const Ctor = getSpeechRecognitionCtor()
    if (!Ctor) {
      const msg = 'Voice input not supported in this browser. Use Chrome or Edge.'
      console.warn('[VoiceInput]', msg)
      setError(msg)
      return
    }

    // Pre-check: acquire microphone stream and KEEP IT ALIVE.
    // Releasing the stream before recognition.start() can cause Chrome
    // to drop the audio source and fire onend immediately.
    try {
      console.log('[VoiceInput] Requesting microphone permission via getUserMedia...')
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      micStreamRef.current = stream // keep alive!
      console.log('[VoiceInput] Microphone permission granted — stream acquired')
    } catch (micErr: any) {
      console.error('[VoiceInput] Microphone access failed:', micErr)
      setError(getMicErrorMessage(micErr))
      return
    }

    wantListeningRef.current = true
    createAndStartRecognition()
  }, [createAndStartRecognition])

  /* ── Stop ── */
  const stopListening = useCallback(() => {
    console.log('[VoiceInput] stopListening called')
    wantListeningRef.current = false
    if (silenceTimerRef.current !== null) {
      clearTimeout(silenceTimerRef.current)
      silenceTimerRef.current = null
    }
    if (recognitionRef.current) {
      try { recognitionRef.current.stop() } catch { /* noop */ }
    }
    setIsListening(false)
    setInterimTranscript('')
    releaseMicStream()
  }, [releaseMicStream])

  /* ── Toggle ── */
  const toggleListening = useCallback(() => {
    if (isListening) {
      stopListening()
    } else {
      startListening()
    }
  }, [isListening, startListening, stopListening])

  /* ── Reset ── */
  const reset = useCallback(() => {
    stopListening()
    setTranscript('')
    setInterimTranscript('')
    setError(null)
  }, [stopListening])

  /* ── Clear transcript only (keep listening) ── */
  const clearTranscript = useCallback(() => {
    setTranscript('')
    setInterimTranscript('')
  }, [])

  return {
    isSupported,
    isListening,
    transcript,
    interimTranscript,
    error,
    startListening,
    stopListening,
    toggleListening,
    reset,
    clearTranscript,
  }
}

export default useVoiceInput
