/**
 * VoiceInputButton — Shared mic button with pulse animation + error tooltip
 *
 * Used by both the Homepage (AnimatedAIChat) and Workspace (AIChatInput)
 * search bars. Accepts the useVoiceInput hook's return values as props.
 */

import React, { useState, useEffect } from 'react'
import { Mic, MicOff } from 'lucide-react'

/* ─────────────────────────────────────────────────────────
   Keyframe injection (once)
───────────────────────────────────────────────────────── */
const STYLE_ID = 'voice-input-styles'

function injectStyles() {
  if (typeof document === 'undefined') return
  if (document.getElementById(STYLE_ID)) return

  const style = document.createElement('style')
  style.id = STYLE_ID
  style.textContent = `
    @keyframes voice-pulse {
      0%   { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.45); }
      70%  { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
      100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    @keyframes voice-wave-1 {
      0%, 100% { height: 6px; }
      50%      { height: 18px; }
    }
    @keyframes voice-wave-2 {
      0%, 100% { height: 10px; }
      50%      { height: 22px; }
    }
    @keyframes voice-wave-3 {
      0%, 100% { height: 4px; }
      50%      { height: 14px; }
    }
    .voice-pulse-ring {
      animation: voice-pulse 1.4s ease-in-out infinite;
    }
    .voice-wave-bar-1 { animation: voice-wave-1 0.8s ease-in-out infinite; }
    .voice-wave-bar-2 { animation: voice-wave-2 0.6s ease-in-out infinite 0.1s; }
    .voice-wave-bar-3 { animation: voice-wave-3 0.9s ease-in-out infinite 0.2s; }
    .voice-wave-bar-4 { animation: voice-wave-1 0.7s ease-in-out infinite 0.15s; }
    .voice-wave-bar-5 { animation: voice-wave-3 0.85s ease-in-out infinite 0.05s; }
  `
  document.head.appendChild(style)
}

/* ─────────────────────────────────────────────────────────
   Props
───────────────────────────────────────────────────────── */
export interface VoiceInputButtonProps {
  /** Is the browser supported? If false, button is disabled */
  isSupported: boolean
  /** Is the mic currently active? */
  isListening: boolean
  /** Error message to show inline */
  error: string | null
  /** Toggle start/stop */
  onToggle: () => void
  /** Button size in px (default 44) */
  size?: number
  /** Icon size in px (default 22) */
  iconSize?: number
}

/* ─────────────────────────────────────────────────────────
   Waveform visualizer (shown when listening)
───────────────────────────────────────────────────────── */
const WaveformBars: React.FC = () => (
  <div
    style={{
      display: 'flex',
      alignItems: 'center',
      gap: 2,
      height: 24,
      paddingRight: 4,
    }}
  >
    {['voice-wave-bar-1', 'voice-wave-bar-2', 'voice-wave-bar-3', 'voice-wave-bar-4', 'voice-wave-bar-5'].map(
      (cls, i) => (
        <div
          key={i}
          className={cls}
          style={{
            width: 3,
            borderRadius: 2,
            background: 'rgba(239, 68, 68, 0.7)',
            transition: 'height 0.15s ease',
          }}
        />
      )
    )}
  </div>
)

/* ─────────────────────────────────────────────────────────
   Error tooltip
───────────────────────────────────────────────────────── */
const ErrorTooltip: React.FC<{ message: string }> = ({ message }) => (
  <div
    style={{
      position: 'absolute',
      bottom: '100%',
      left: '50%',
      transform: 'translateX(-50%)',
      marginBottom: 8,
      padding: '6px 12px',
      borderRadius: 8,
      background: 'rgba(30, 30, 32, 0.96)',
      border: '1px solid rgba(239, 68, 68, 0.3)',
      color: 'rgba(255, 180, 180, 0.95)',
      fontSize: 12,
      fontWeight: 500,
      fontFamily: 'Inter, system-ui, sans-serif',
      whiteSpace: 'nowrap',
      pointerEvents: 'none',
      zIndex: 100,
      boxShadow: '0 4px 16px rgba(0,0,0,0.4)',
      maxWidth: 280,
    }}
  >
    {message}
  </div>
)

/* ─────────────────────────────────────────────────────────
   VoiceInputButton
───────────────────────────────────────────────────────── */
export const VoiceInputButton: React.FC<VoiceInputButtonProps> = ({
  isSupported,
  isListening,
  error,
  onToggle,
  size = 44,
  iconSize = 22,
}) => {
  const [hovered, setHovered] = useState(false)
  const [showError, setShowError] = useState(false)

  // Inject keyframe styles once
  useEffect(() => { injectStyles() }, [])

  // Show error tooltip briefly
  useEffect(() => {
    if (error) {
      setShowError(true)
      const t = setTimeout(() => setShowError(false), 8000)
      return () => clearTimeout(t)
    } else {
      setShowError(false)
    }
  }, [error])

  if (!isSupported) {
    return (
      <button
        disabled
        title="Voice input not supported in this browser"
        style={{
          width: size,
          height: size,
          minWidth: size,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          borderRadius: '50%',
          border: 'none',
          background: 'transparent',
          cursor: 'not-allowed',
          opacity: 0.3,
        }}
      >
        <MicOff size={iconSize} style={{ color: 'rgba(255,255,255,0.30)' }} />
      </button>
    )
  }

  return (
    <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
      {/* Waveform bars — visible when listening */}
      {isListening && <WaveformBars />}

      {/* Error tooltip */}
      {showError && error && <ErrorTooltip message={error} />}

      {/* Mic button */}
      <button
        onClick={(e) => { e.stopPropagation(); onToggle() }}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        title={isListening ? 'Stop listening' : 'Voice input'}
        type="button"
        tabIndex={-1}
        className={isListening ? 'voice-pulse-ring' : ''}
        style={{
          width: size,
          height: size,
          minWidth: size,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          borderRadius: '50%',
          border: 'none',
          background: isListening
            ? 'rgba(239, 68, 68, 0.15)'
            : hovered
              ? 'rgba(255, 255, 255, 0.09)'
              : 'transparent',
          cursor: 'pointer',
          transition: 'background 150ms ease',
          flexShrink: 0,
          position: 'relative',
        }}
      >
        <Mic
          size={iconSize}
          style={{
            color: isListening
              ? 'rgb(239, 68, 68)'
              : hovered
                ? 'rgba(255, 255, 255, 0.85)'
                : 'rgba(255, 255, 255, 0.60)',
            transition: 'color 150ms ease',
          }}
        />
      </button>
    </div>
  )
}

export default VoiceInputButton
