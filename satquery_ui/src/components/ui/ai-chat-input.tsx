"use client"

import * as React from "react"
import { useState, useEffect, useRef } from "react"
import { Paperclip, Send } from "lucide-react"
import { AnimatePresence, motion } from "framer-motion"
import { useVoiceInput } from "@/hooks/useVoiceInput"
import { VoiceInputButton } from "./VoiceInputButton"

const PLACEHOLDERS = [
  "Ask about satellite imagery...",
  "Detect changes between two dates...",
  "Classify land use in this region...",
  "Run SAR + optical fusion analysis...",
  "Describe what you see in this image...",
  "Identify water bodies in Sentinel-2...",
]

interface AIChatInputProps {
  onSubmit?: (query: string, files: File[], task: string | null) => void
  placeholder?: string
}

const AIChatInput = ({ onSubmit }: AIChatInputProps) => {
  const [placeholderIndex, setPlaceholderIndex] = useState(0)
  const [showPlaceholder, setShowPlaceholder] = useState(true)
  const [isActive, setIsActive] = useState(false)
  const [inputValue, setInputValue] = useState("")
  const [files, setFiles] = useState<File[]>([])
  const wrapperRef = useRef<HTMLDivElement>(null)
  const lastConsumedTranscriptRef = useRef("")

  /* ── Voice input hook ── */
  const voice = useVoiceInput()

  /* Cycle placeholder text when input is inactive */
  useEffect(() => {
    if (isActive || inputValue) return

    const interval = setInterval(() => {
      setShowPlaceholder(false)
      setTimeout(() => {
        setPlaceholderIndex((prev) => (prev + 1) % PLACEHOLDERS.length)
        setShowPlaceholder(true)
      }, 400)
    }, 3000)

    return () => clearInterval(interval)
  }, [isActive, inputValue])

  /* Close when clicking outside */
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (wrapperRef.current && !wrapperRef.current.contains(event.target as Node)) {
        if (!inputValue) setIsActive(false)
      }
    }
    document.addEventListener("mousedown", handleClickOutside)
    return () => document.removeEventListener("mousedown", handleClickOutside)
  }, [inputValue])

  const handleActivate = () => setIsActive(true)

  const handleSend = () => {
    if (!inputValue.trim() && files.length === 0) return
    voice.stopListening()
    onSubmit?.(inputValue.trim(), files, null)
    setInputValue("")
    setFiles([])
    setIsActive(false)
    lastConsumedTranscriptRef.current = ""
    voice.reset()
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFiles((prev) => [...prev, ...Array.from(e.target.files!)])
    }
    e.target.value = ""
  }

  /* Programmatic file picker — creates a fresh <input> each click to avoid
     Chromium bug where zero-dimension hidden inputs silently refuse .click() */
  const openFilePicker = () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.multiple = true
    input.accept = 'image/*,.tif,.tiff,.png,.jpg,.jpeg,.webp,.npz,.nc,.geojson'
    input.style.display = 'none'
    document.body.appendChild(input)
    input.addEventListener('change', (ev) => {
      const target = ev.target as HTMLInputElement
      if (target.files && target.files.length > 0) {
        setFiles(prev => [...prev, ...Array.from(target.files!)])
      }
      document.body.removeChild(input)
    })
    input.addEventListener('cancel', () => {
      document.body.removeChild(input)
    })
    input.click()
  }

  const containerVariants = {
    collapsed: {
      boxShadow: "0 2px 16px 0 rgba(0,0,0,0.35)",
      transition: { type: "spring" as const, stiffness: 120, damping: 18 },
    },
    expanded: {
      boxShadow: "0 2px 16px 0 rgba(0,0,0,0.35)",
      transition: { type: "spring" as const, stiffness: 120, damping: 18 },
    },
  }

  const placeholderContainerVariants = {
    initial: {},
    animate: { transition: { staggerChildren: 0.022 } },
    exit: { transition: { staggerChildren: 0.012, staggerDirection: -1 as const } },
  }

  const letterVariants = {
    initial: { opacity: 0, filter: "blur(10px)", y: 8 },
    animate: {
      opacity: 1,
      filter: "blur(0px)",
      y: 0,
      transition: {
        opacity: { duration: 0.22 },
        filter: { duration: 0.35 },
        y: { type: "spring" as const, stiffness: 80, damping: 20 },
      },
    },
    exit: {
      opacity: 0,
      filter: "blur(10px)",
      y: -8,
      transition: {
        opacity: { duration: 0.18 },
        filter: { duration: 0.28 },
        y: { type: "spring" as const, stiffness: 80, damping: 20 },
      },
    },
  }

  const canSend = inputValue.trim().length > 0 || files.length > 0

  /* ── Sync voice transcript into input ── */
  useEffect(() => {
    if (voice.transcript && voice.transcript !== lastConsumedTranscriptRef.current) {
      console.log('[AIChatInput] Consuming transcript:', voice.transcript)
      lastConsumedTranscriptRef.current = voice.transcript
      setInputValue(prev => {
        const separator = prev && !prev.endsWith(' ') ? ' ' : ''
        return prev + separator + voice.transcript
      })
      // Clear consumed transcript without stopping the listening session.
      // voice.reset() would kill the session — we only want to clear the
      // accumulated text so it doesn't get re-appended.
      voice.clearTranscript()
      setIsActive(true)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [voice.transcript])

  return (
    <div className="w-full">


      <motion.div
        ref={wrapperRef}
        className="w-full"
        variants={containerVariants}
        animate={isActive || inputValue ? "expanded" : "collapsed"}
        initial="collapsed"
        style={{
          borderRadius: 24,
          background: "rgba(20, 20, 20, 0.92)",
          border: "1px solid rgba(255,255,255,0.10)",
          backdropFilter: "blur(24px)",
          WebkitBackdropFilter: "blur(24px)",
          /* inner padding keeps buttons away from the border */
          padding: 6,
          minHeight: 68,
        }}
        onClick={handleActivate}
      >
        <div className="flex flex-col items-stretch w-full h-full">

          {/* ── Input row ── */}
          <div className="flex items-center gap-3 px-2 py-1 w-full">

            {/* Attach */}
            <button
              className="flex items-center justify-center rounded-full hover:bg-white/[0.09] transition-colors flex-shrink-0"
              style={{ width: 44, height: 44, minWidth: 44 }}
              title="Attach satellite imagery or data files"
              type="button"
              tabIndex={-1}
              onClick={(e) => { e.stopPropagation(); openFilePicker() }}
            >
              <Paperclip size={22} style={{ color: "rgba(255,255,255,0.60)" }} />
            </button>

            {/* Text input + animated placeholder */}
            <div className="relative flex-1 min-w-0">
              <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                onFocus={handleActivate}
                className="w-full border-0 outline-none bg-transparent py-2 text-base leading-relaxed"
                style={{
                  color: "rgba(255,255,255,0.90)",
                  fontFamily: "Inter, system-ui, sans-serif",
                  fontSize: 15,
                  position: "relative",
                  zIndex: 1,
                }}
              />

              {/* Animated placeholder */}
              <div className="absolute left-0 top-0 w-full h-full pointer-events-none flex items-center">
                <AnimatePresence mode="wait">
                  {showPlaceholder && !isActive && !inputValue && (
                    <motion.span
                      key={placeholderIndex}
                      className="absolute left-0 top-1/2 -translate-y-1/2 select-none pointer-events-none overflow-hidden"
                      style={{
                        whiteSpace: "nowrap",
                        textOverflow: "ellipsis",
                        color: "rgba(255,255,255,0.28)",
                        fontSize: 15,
                        fontFamily: "Inter, system-ui, sans-serif",
                        zIndex: 0,
                      }}
                      variants={placeholderContainerVariants}
                      initial="initial"
                      animate="animate"
                      exit="exit"
                    >
                      {PLACEHOLDERS[placeholderIndex].split("").map((char, i) => (
                        <motion.span
                          key={i}
                          variants={letterVariants}
                          style={{ display: "inline-block" }}
                        >
                          {char === " " ? "\u00A0" : char}
                        </motion.span>
                      ))}
                    </motion.span>
                  )}
                </AnimatePresence>
              </div>
            </div>

            {/* Voice input */}
            <VoiceInputButton
              isSupported={voice.isSupported}
              isListening={voice.isListening}
              error={voice.error}
              onToggle={voice.toggleListening}
            />

            {/* Interim voice transcript indicator */}
            {voice.isListening && voice.interimTranscript && (
              <div
                style={{
                  position: 'absolute',
                  bottom: -28,
                  left: 60,
                  right: 100,
                  fontSize: 12,
                  color: 'rgba(255,255,255,0.35)',
                  fontStyle: 'italic',
                  fontFamily: 'Inter, system-ui, sans-serif',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                  pointerEvents: 'none',
                }}
              >
                {voice.interimTranscript}
              </div>
            )}

            {/* Send */}
            <button
              onClick={(e) => { e.stopPropagation(); handleSend() }}
              disabled={!canSend}
              className="flex items-center justify-center gap-2 rounded-full transition-all duration-200 flex-shrink-0 font-medium"
              title="Send"
              type="button"
              style={{
                height: 44,
                minWidth: 44,
                paddingLeft: canSend ? 20 : 0,
                paddingRight: canSend ? 20 : 0,
                width: canSend ? "auto" : 44,
                background: canSend ? "#ffffff" : "rgba(255,255,255,0.08)",
                border: canSend ? "none" : "1px solid rgba(255,255,255,0.12)",
                cursor: canSend ? "pointer" : "not-allowed",
                fontSize: 14,
                color: canSend ? "#0a0a0a" : "rgba(255,255,255,0.25)",
              }}
            >
              <Send size={18} />
              {canSend && <span>Send</span>}
            </button>
          </div>

          {/* Files badge — only shown when files are attached */}
          {files.length > 0 && (
            <div className="px-4 pb-2">
              <span
                className="text-[12px] px-3 py-1.5 rounded-full"
                style={{
                  background: "rgba(255,255,255,0.06)",
                  border: "1px solid rgba(255,255,255,0.09)",
                  color: "rgba(255,255,255,0.40)",
                }}
              >
                {files.length} file{files.length > 1 ? "s" : ""} attached
              </span>
            </div>
          )}

        </div>
      </motion.div>
    </div>
  )
}

export { AIChatInput }
export default AIChatInput
