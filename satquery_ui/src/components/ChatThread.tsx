import React, { useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Download, AlertCircle, ChevronDown, CheckCircle2, Loader2, RefreshCw } from 'lucide-react'
import { useState } from 'react'
import { exportMessageToPdf } from '@/lib/pdf-export'

/* ─────────────────────────────────────────────────────────
   Types
───────────────────────────────────────────────────────── */
interface BoundingBox {
  label: string
  x1: number
  y1: number
  x2: number
  y2: number
  confidence?: number
}

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  taskType?: string
  inputType?: string
  modelUsed?: string
  confidence?: number
  previewUrls?: string[]
  uploadedFileNames?: string[]   // parallel array to previewUrls
  boundingBoxes?: BoundingBox[]
  masksB64?: string[]             // SAM segmentation masks (base64 PNG)
  opticalPct?: number
  sarPct?: number
  isStreaming?: boolean
}

interface ChatThreadProps {
  messages: ChatMessage[]
  isLoading: boolean
  onRetry?: () => void
}

/* ─────────────────────────────────────────────────────────
   Constants
───────────────────────────────────────────────────────── */
const TASK_LABELS: Record<string, string> = {
  vqa: 'VQA',
  caption: 'Caption',
  change: 'Change Detection',
  fusion: 'Cross-Modal Fusion',
  grounding: 'Object Grounding',
  agentic: 'Agentic',
}

/* ─────────────────────────────────────────────────────────
   SatQuery orbital avatar
───────────────────────────────────────────────────────── */
const SatMark = () => (
  <div
    className="flex-shrink-0 flex items-center justify-center rounded-full overflow-hidden"
    style={{
      width: 32, height: 32,
      background: 'rgba(255,255,255,0.06)',
      border: '1px solid rgba(255,255,255,0.12)',
    }}
  >
    <img src="/ai-avatar.png" alt="SatQuery AI" className="w-full h-full object-cover" />
  </div>
)

/* ─────────────────────────────────────────────────────────
   Bounding-box canvas overlay — draws coloured boxes on
   top of the uploaded image for grounding results
───────────────────────────────────────────────────────── */
const BBOX_COLORS = ['#22d3ee', '#a78bfa', '#34d399', '#f59e0b', '#f87171']

const BBoxCanvas: React.FC<{ previewUrl: string; boxes: BoundingBox[]; masksB64?: string[] }> = ({ previewUrl, boxes, masksB64 }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const [loadedMasks, setLoadedMasks] = useState<HTMLImageElement[]>([])

  // Load mask images asynchronously when masksB64 changes
  React.useEffect(() => {
    if (!masksB64 || masksB64.length === 0) {
      setLoadedMasks([])
      return
    }
    let active = true
    const images: HTMLImageElement[] = []
    let loaded = 0

    masksB64.forEach((b64, idx) => {
      const img = new Image()
      const src = b64.startsWith('data:') ? b64 : `data:image/png;base64,${b64}`
      img.onload = () => {
        if (!active) return
        images[idx] = img
        loaded++
        if (loaded === masksB64.length) setLoadedMasks([...images])
      }
      img.onerror = () => {
        if (!active) return
        loaded++
        if (loaded === masksB64.length) setLoadedMasks([...images.filter(Boolean)])
      }
      img.src = src
    })

    return () => { active = false }
  }, [masksB64])

  const draw = React.useCallback(() => {
    const canvas = canvasRef.current
    const img = imgRef.current
    if (!canvas || !img) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = img.naturalWidth || img.clientWidth
    canvas.height = img.naturalHeight || img.clientHeight
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // 1. Draw SAM segmentation mask overlays if present
    loadedMasks.forEach((maskImg) => {
      if (maskImg) {
        ctx.globalAlpha = 0.5
        ctx.drawImage(maskImg, 0, 0, canvas.width, canvas.height)
        ctx.globalAlpha = 1.0
      }
    })

    // 2. Draw bounding boxes & labels
    boxes.forEach((box, i) => {
      const color = BBOX_COLORS[i % BBOX_COLORS.length]
      const x = box.x1 * canvas.width
      const y = box.y1 * canvas.height
      const w = (box.x2 - box.x1) * canvas.width
      const h = (box.y2 - box.y1) * canvas.height

      // Box fill (semi-transparent)
      ctx.fillStyle = color + '22'
      ctx.fillRect(x, y, w, h)

      // Box border
      ctx.strokeStyle = color
      ctx.lineWidth = Math.max(2, canvas.width * 0.003)
      ctx.strokeRect(x, y, w, h)

      // Label background
      const label = box.label || `Object ${i + 1}`
      ctx.font = `bold ${Math.max(12, canvas.width * 0.025)}px Inter, sans-serif`
      const textW = ctx.measureText(label).width
      const labelH = Math.max(20, canvas.width * 0.04)
      ctx.fillStyle = color
      ctx.fillRect(x, Math.max(0, y - labelH), textW + 10, labelH)

      // Label text
      ctx.fillStyle = '#000000'
      ctx.fillText(label, x + 5, Math.max(labelH - 5, y - 5))
    })
  }, [boxes, loadedMasks])

  React.useEffect(() => {
    draw()
  }, [draw])

  return (
    <div style={{ position: 'relative', width: '100%', maxWidth: 520, borderRadius: 12, overflow: 'hidden' }}>
      <img
        ref={imgRef}
        src={previewUrl}
        onLoad={draw}
        style={{ width: '100%', display: 'block', borderRadius: 12 }}
        alt="Satellite image with detections"
      />
      <canvas
        ref={canvasRef}
        style={{
          position: 'absolute', top: 0, left: 0,
          width: '100%', height: '100%',
          pointerEvents: 'none',
        }}
      />
      {/* Legend */}
      {boxes.length > 0 && (
        <div
          style={{
            position: 'absolute', bottom: 8, left: 8, right: 8,
            display: 'flex', flexWrap: 'wrap', gap: 6,
            maxHeight: 75, overflowY: 'auto',
            pointerEvents: 'auto',
          }}
        >
          {boxes.slice(0, 8).map((box, i) => (
            <div
              key={i}
              style={{
                display: 'flex', alignItems: 'center', gap: 6,
                padding: '3px 8px', borderRadius: 6,
                background: 'rgba(0,0,0,0.75)',
                backdropFilter: 'blur(8px)',
                border: `1px solid ${BBOX_COLORS[i % BBOX_COLORS.length]}44`,
              }}
            >
              <span style={{ width: 8, height: 8, borderRadius: 2, flexShrink: 0, background: BBOX_COLORS[i % BBOX_COLORS.length] }} />
              <span style={{ fontSize: 11, color: '#fff', fontFamily: 'Inter, sans-serif' }}>
                {box.label}
                {box.confidence !== undefined ? ` · ${Math.round(box.confidence * 100)}%` : ''}
              </span>
            </div>
          ))}
          {boxes.length > 8 && (
            <div
              style={{
                display: 'flex', alignItems: 'center',
                padding: '3px 8px', borderRadius: 6,
                background: 'rgba(0,0,0,0.75)',
                fontSize: 11, color: 'rgba(255,255,255,0.7)',
                backdropFilter: 'blur(8px)',
              }}
            >
              +{boxes.length - 8} more
            </div>
          )}
        </div>
      )}
    </div>
  )
}

/* ─────────────────────────────────────────────────────────
   Image thumbnail strip — shown in user message bubble.
   Handles browser-renderable formats (JPG/PNG/WEBP) AND
   non-renderable satellite formats (TIFF/GeoTIFF) gracefully.
───────────────────────────────────────────────────────── */
const BROWSER_RENDERABLE = /\.(png|jpg|jpeg|gif|webp|bmp|avif)$/i
const SATELLITE_ICON = (
  <svg viewBox="0 0 24 24" fill="none" style={{ width: 28, height: 28, color: 'rgba(255,255,255,0.4)' }}>
    <rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" strokeWidth="1.4" />
    <path d="M3 9h18M3 15h18M9 3v18M15 3v18" stroke="currentColor" strokeWidth="1" opacity="0.4" />
    <circle cx="12" cy="12" r="2.5" fill="currentColor" opacity="0.5" />
  </svg>
)

const FileBadge: React.FC<{ name: string }> = ({ name }) => (
  <div style={{
    width: 120, height: 80, borderRadius: 10, flexShrink: 0,
    border: '1px solid rgba(255,255,255,0.12)',
    background: 'rgba(255,255,255,0.04)',
    display: 'flex', flexDirection: 'column',
    alignItems: 'center', justifyContent: 'center', gap: 6,
    padding: '8px 6px',
  }}>
    {SATELLITE_ICON}
    <span style={{
      fontSize: 10, color: 'rgba(255,255,255,0.40)',
      fontFamily: 'Inter, sans-serif', textAlign: 'center',
      wordBreak: 'break-all', lineHeight: 1.3,
      maxWidth: '100%', overflow: 'hidden',
    }}>
      {name.length > 16 ? name.slice(0, 14) + '…' : name}
    </span>
  </div>
)

const ImageThumbnailRow: React.FC<{ urls: string[]; fileNames?: string[] }> = ({ urls, fileNames }) => {
  const [failedUrls, setFailedUrls] = React.useState<Set<string>>(new Set())

  return (
    <div className="flex gap-2 mt-2 flex-wrap">
      {urls.map((url, i) => {
        const name = fileNames?.[i] ?? `image_${i + 1}`
        const isDataOrBlob = typeof url === 'string' && (url.startsWith('data:image/') || url.startsWith('blob:'))
        const isRenderable = (isDataOrBlob || BROWSER_RENDERABLE.test(name) || BROWSER_RENDERABLE.test(url)) && !failedUrls.has(url)

        if (!isRenderable) {
          return <FileBadge key={i} name={name} />
        }

        return (
          <div
            key={i}
            style={{
              width: 120, height: 80, borderRadius: 10,
              overflow: 'hidden', border: '1px solid rgba(255,255,255,0.12)',
              background: 'rgba(0,0,0,0.3)', flexShrink: 0, cursor: 'pointer',
            }}
            title={name}
          >
            <img
              src={url}
              alt={name}
              onError={() => setFailedUrls(prev => new Set(prev).add(url))}
              style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
            />
          </div>
        )
      })}
    </div>
  )
}

/* ─────────────────────────────────────────────────────────
   Assistant image preview — prominent visual container for
   converted PNGs, cyan water overlays, change masks, etc.
───────────────────────────────────────────────────────── */
const AssistantImagePreview: React.FC<{ url: string; label?: string }> = ({ url, label }) => {
  const [zoom, setZoom] = useState(false)
  const [failed, setFailed] = useState(false)

  if (failed) {
    return (
      <div className="mb-4">
        <FileBadge name={label || 'Output image'} />
      </div>
    )
  }

  const handleDownload = (e: React.MouseEvent) => {
    e.stopPropagation()
    const a = document.createElement('a')
    a.href = url
    a.download = `satquery_analysis_${Date.now()}.png`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  return (
    <>
      <div
        className="relative group overflow-hidden mb-4 cursor-pointer"
        style={{
          maxWidth: 520,
          borderRadius: 12,
          border: '1px solid rgba(255,255,255,0.15)',
          background: 'rgba(0,0,0,0.4)',
          boxShadow: '0 4px 20px rgba(0,0,0,0.3)',
        }}
        onClick={() => setZoom(true)}
      >
        <img
          src={url}
          alt={label || 'SatQuery Analysis Output'}
          onError={() => setFailed(true)}
          style={{ width: '100%', height: 'auto', display: 'block', borderRadius: 12 }}
        />
        {/* Hover overlay bar */}
        <div
          className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-end justify-between p-3"
          style={{ backdropFilter: 'blur(2px)' }}
        >
          <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.9)', fontWeight: 500, fontFamily: 'Inter, sans-serif' }}>
            Click to expand
          </span>
          <button
            onClick={handleDownload}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-white/10 hover:bg-white/20 text-white transition-colors"
            style={{ border: '1px solid rgba(255,255,255,0.2)' }}
          >
            <Download className="w-3.5 h-3.5" />
            Download
          </button>
        </div>
      </div>

      {/* Lightbox Modal */}
      {zoom && (
        <div
          className="fixed inset-0 z-50 bg-black/85 backdrop-blur-md flex items-center justify-center p-4"
          onClick={() => setZoom(false)}
        >
          <div className="relative max-w-4xl max-h-[90vh] overflow-hidden rounded-xl border border-white/20">
            <img src={url} alt="Expanded Analysis" className="max-w-full max-h-[85vh] object-contain" />
            <button
              onClick={() => setZoom(false)}
              className="absolute top-3 right-3 p-2 rounded-full bg-black/60 text-white hover:bg-black/80 transition-colors"
            >
              ✕
            </button>
          </div>
        </div>
      )}
    </>
  )
}

/* ─────────────────────────────────────────────────────────
   Meta row — task · model · confidence
───────────────────────────────────────────────────────── */
const TASK_BADGE_STYLES: Record<string, { bg: string; border: string; color: string; dot: string }> = {
  grounding: {
    bg: 'linear-gradient(135deg, rgba(34,211,238,0.16) 0%, rgba(59,130,246,0.16) 100%)',
    border: '1px solid rgba(34,211,238,0.40)',
    color: 'rgba(165,243,252,0.95)',
    dot: '#22d3ee',
  },
  vqa: {
    bg: 'linear-gradient(135deg, rgba(167,139,250,0.16) 0%, rgba(139,92,246,0.16) 100%)',
    border: '1px solid rgba(167,139,250,0.40)',
    color: 'rgba(221,214,254,0.95)',
    dot: '#a78bfa',
  },
  change: {
    bg: 'linear-gradient(135deg, rgba(251,191,36,0.16) 0%, rgba(245,158,11,0.16) 100%)',
    border: '1px solid rgba(251,191,36,0.40)',
    color: 'rgba(254,240,138,0.95)',
    dot: '#fbbf24',
  },
  fusion: {
    bg: 'linear-gradient(135deg, rgba(52,211,153,0.16) 0%, rgba(16,185,129,0.16) 100%)',
    border: '1px solid rgba(52,211,153,0.40)',
    color: 'rgba(167,243,208,0.95)',
    dot: '#34d399',
  },
  caption: {
    bg: 'linear-gradient(135deg, rgba(56,189,248,0.16) 0%, rgba(99,102,241,0.16) 100%)',
    border: '1px solid rgba(56,189,248,0.40)',
    color: 'rgba(186,230,253,0.95)',
    dot: '#38bdf8',
  },
  agentic: {
    bg: 'linear-gradient(135deg, rgba(236,72,153,0.16) 0%, rgba(168,85,247,0.16) 100%)',
    border: '1px solid rgba(236,72,153,0.40)',
    color: 'rgba(251,207,232,0.95)',
    dot: '#ec4899',
  },
}

const MetaRow = ({
  taskType, modelUsed, confidence,
}: { taskType?: string; modelUsed?: string; confidence?: number }) => {
  if (!taskType && !modelUsed && confidence === undefined) return null

  const badgeStyle = TASK_BADGE_STYLES[taskType || 'vqa'] ?? TASK_BADGE_STYLES.vqa

  const confColor = confidence !== undefined
    ? confidence > 0.7 ? 'rgba(134,239,172,0.70)'
    : confidence > 0.4 ? 'rgba(251,191,36,0.70)'
    : 'rgba(248,113,113,0.70)'
    : undefined

  return (
    <div className="flex items-center gap-3 mb-4 flex-wrap">
      {taskType && TASK_LABELS[taskType] && (
        <span
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 8,
            padding: '6px 22px',
            borderRadius: 24,
            minWidth: 130,
            justifyContent: 'center',
            fontSize: 12,
            fontWeight: 600,
            letterSpacing: '0.04em',
            whiteSpace: 'nowrap',
            background: badgeStyle.bg,
            border: badgeStyle.border,
            color: badgeStyle.color,
            boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
            backdropFilter: 'blur(8px)',
          }}
        >
          <span
            style={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              background: badgeStyle.dot,
              boxShadow: `0 0 6px ${badgeStyle.dot}`,
              flexShrink: 0,
            }}
          />
          {TASK_LABELS[taskType]}
        </span>
      )}
      {modelUsed && (
        <>
          <span style={{ color: 'rgba(255,255,255,0.15)', fontSize: 10 }}>·</span>
          <span className="text-[12px] font-medium tracking-wide" style={{ color: 'rgba(255,255,255,0.40)' }}>
            {modelUsed}
          </span>
        </>
      )}
      {confidence !== undefined && (
        <>
          <span style={{ color: 'rgba(255,255,255,0.15)', fontSize: 10 }}>·</span>
          <span className="text-[12px] font-medium tracking-wide flex items-center gap-1.5" style={{ color: confColor }}>
            <span style={{
              width: 6, height: 6, borderRadius: '50%',
              background: confColor, flexShrink: 0,
            }} />
            {(confidence * 100).toFixed(0)}% confidence
          </span>
        </>
      )}
    </div>
  )
}

/* ─────────────────────────────────────────────────────────
   Formatted text — handles line breaks and basic structure
───────────────────────────────────────────────────────── */
const FormattedText: React.FC<{ text: string }> = ({ text }) => {
  const lines = text.split(/\n+/).filter(l => l.trim().length > 0)

  // Render inline markdown (bold)
  const renderInline = (str: string) => {
    const parts: React.ReactNode[] = []
    const regex = /\*\*(.+?)\*\*/g
    let lastIndex = 0
    let match: RegExpExecArray | null

    while ((match = regex.exec(str)) !== null) {
      if (match.index > lastIndex) {
        parts.push(str.slice(lastIndex, match.index))
      }
      parts.push(
        <strong key={match.index} style={{ color: 'rgba(255,255,255,0.95)', fontWeight: 600 }}>
          {match[1]}
        </strong>
      )
      lastIndex = regex.lastIndex
    }
    if (lastIndex < str.length) {
      parts.push(str.slice(lastIndex))
    }
    return parts.length > 0 ? parts : [str]
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {lines.map((line, i) => {
        const trimmed = line.trim()

        // Headers (### or ##)
        if (/^#{1,3}\s/.test(trimmed)) {
          const headerText = trimmed.replace(/^#{1,3}\s*/, '')
          return (
            <p key={i} style={{
              color: 'rgba(255,255,255,0.92)', fontSize: 16,
              fontFamily: 'Inter, system-ui, sans-serif', lineHeight: 1.6,
              fontWeight: 600, marginTop: i > 0 ? 8 : 0,
            }}>
              {renderInline(headerText)}
            </p>
          )
        }

        // Numbered list items (1. or 1)
        const numberedMatch = trimmed.match(/^(\d+)[.)]\s+(.*)/)
        if (numberedMatch) {
          return (
            <div key={i} style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
              <span style={{
                color: 'rgba(255,255,255,0.40)', marginTop: 1, flexShrink: 0,
                fontSize: 14, fontWeight: 600, minWidth: 18, textAlign: 'right',
              }}>
                {numberedMatch[1]}.
              </span>
              <span style={{
                color: 'rgba(255,255,255,0.82)', fontSize: 15,
                fontFamily: 'Inter, system-ui, sans-serif', lineHeight: 1.7,
              }}>
                {renderInline(numberedMatch[2])}
              </span>
            </div>
          )
        }

        // Bullet points
        const isBullet = /^[-•*]\s/.test(trimmed)
        if (isBullet) {
          const cleaned = trimmed.replace(/^[-•*]\s*/, '')
          return (
            <div key={i} style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
              <span style={{ color: 'rgba(255,255,255,0.35)', marginTop: 2, flexShrink: 0 }}>•</span>
              <span style={{
                color: 'rgba(255,255,255,0.82)', fontSize: 15,
                fontFamily: 'Inter, system-ui, sans-serif', lineHeight: 1.7,
              }}>
                {renderInline(cleaned)}
              </span>
            </div>
          )
        }

        // Regular paragraph
        return (
          <p key={i} style={{
            color: 'rgba(255,255,255,0.85)', fontSize: 15,
            fontFamily: 'Inter, system-ui, sans-serif',
            lineHeight: 1.8, fontWeight: 380, margin: 0,
          }}>
            {renderInline(line)}
          </p>
        )
      })}
    </div>
  )
}

/* ─────────────────────────────────────────────────────────
   Collapsible execution trace
───────────────────────────────────────────────────────── */
const ExecutionTrace = ({
  taskType, inputType, modelUsed, confidence, isStreaming,
}: {
  taskType: string; inputType: string; modelUsed: string
  confidence: number; isStreaming?: boolean
}) => {
  const [open, setOpen] = useState(false)
  const pct = (confidence * 100).toFixed(0)

  const steps = [
    { label: 'Query classified', detail: TASK_LABELS[taskType] ?? taskType },
    { label: 'Input validated', detail: inputType },
    { label: 'Model selected', detail: modelUsed },
    { label: 'Analysis complete', detail: isStreaming ? 'In progress…' : `${pct}% confidence` },
  ]

  return (
    <div className="mt-5">
      <button
        onClick={() => setOpen(v => !v)}
        className="flex items-center gap-2"
        style={{ cursor: 'pointer' }}
      >
        {isStreaming
          ? <Loader2 className="w-3 h-3 animate-spin" style={{ color: 'rgba(255,255,255,0.30)' }} />
          : <CheckCircle2 className="w-3 h-3" style={{ color: 'rgba(255,255,255,0.30)' }} />
        }
        <span className="text-[11px] uppercase tracking-widest font-medium" style={{ color: 'rgba(255,255,255,0.28)' }}>
          Execution trace
        </span>
        <motion.span animate={{ rotate: open ? 180 : 0 }} transition={{ duration: 0.2 }}>
          <ChevronDown className="w-3 h-3" style={{ color: 'rgba(255,255,255,0.22)' }} />
        </motion.span>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
            className="overflow-hidden"
          >
            <div className="mt-3 flex flex-col gap-2 pl-1">
              {steps.map((step, i) => (
                <div key={i} className="flex items-baseline gap-3">
                  <span className="text-[12px] font-medium" style={{ color: 'rgba(255,255,255,0.50)', minWidth: 140 }}>
                    {step.label}
                  </span>
                  <span className="text-[12px]" style={{ color: 'rgba(255,255,255,0.28)' }}>
                    {step.detail}
                  </span>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

/* ─────────────────────────────────────────────────────────
   Error card — friendly message + retry button
───────────────────────────────────────────────────────── */
const ErrorCard: React.FC<{ content: string; onRetry?: () => void }> = ({ content, onRetry }) => {
  const isOffline = /backend unavailable|fetch|network|econnrefused/i.test(content)
  const isChangeCount = /bi_temporal|change.*incompatible/i.test(content)
  const isFusionScope = /fusion.*scope|cross_modal.*incompatible/i.test(content)

  const msg =
    isChangeCount ? 'Change detection requires exactly two images: a Before and an After image.'
    : isFusionScope ? 'Cross-modal fusion requires an Optical image and a SAR image pair.'
    : isOffline ? 'SatQuery services are currently unavailable. Please ensure all backend services are running on ports 8000, 8100, and 8200.'
    : content

  return (
    <div style={{
      borderRadius: 14, padding: '16px 20px',
      background: 'rgba(248,113,113,0.05)',
      border: '1px solid rgba(248,113,113,0.18)',
    }}>
      <div className="flex items-start gap-3">
        <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" style={{ color: 'rgba(248,113,113,0.70)' }} />
        <div className="flex flex-col gap-3 flex-1 min-w-0">
          <p style={{ fontSize: 14, color: 'rgba(248,113,113,0.80)', lineHeight: 1.65, margin: 0 }}>
            {msg}
          </p>
          {onRetry && (
            <button
              onClick={onRetry}
              style={{
                display: 'flex', alignItems: 'center', gap: 6,
                width: 'fit-content', padding: '6px 14px', borderRadius: 8,
                fontSize: 12, cursor: 'pointer',
                background: 'rgba(248,113,113,0.10)',
                border: '1px solid rgba(248,113,113,0.20)',
                color: 'rgba(248,113,113,0.70)',
              }}
            >
              <RefreshCw style={{ width: 12, height: 12 }} />
              Try again
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

/* ─────────────────────────────────────────────────────────
   Loading indicator
───────────────────────────────────────────────────────── */
const LoadingRow = () => (
  <motion.div
    initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
    className="flex items-start gap-4 py-2"
  >
    <SatMark />
    <div className="flex items-center gap-1.5 pt-1.5">
      {[0, 1, 2].map(i => (
        <motion.div
          key={i} className="rounded-full"
          style={{ width: 5, height: 5, background: 'rgba(255,255,255,0.30)' }}
          animate={{ scale: [1, 1.5, 1], opacity: [0.4, 0.9, 0.4] }}
          transition={{ duration: 1.1, delay: i * 0.18, repeat: Infinity, ease: 'easeInOut' }}
        />
      ))}
    </div>
  </motion.div>
)

/* ─────────────────────────────────────────────────────────
   ChatThread
───────────────────────────────────────────────────────── */
export const ChatThread: React.FC<ChatThreadProps> = ({ messages, isLoading, onRetry }) => {
  return (
    <div className="flex flex-col gap-8 py-6">
      <AnimatePresence initial={false}>
        {messages.map(msg => (
          <motion.div
            key={msg.id}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.30, ease: [0.16, 1, 0.3, 1] }}
          >
            {msg.role === 'user' ? (
              /* ── User message — right-aligned with image thumbnails ── */
              <div className="flex justify-end">
                <div style={{ maxWidth: 560 }}>
                  {/* Image thumbnails above the pill */}
                  {msg.previewUrls && msg.previewUrls.length > 0 && (
                    <div className="flex justify-end">
                      <ImageThumbnailRow urls={msg.previewUrls} fileNames={msg.uploadedFileNames} />
                    </div>
                  )}
                  {/* Text pill */}
                  {msg.content && (
                    <div
                      className="rounded-2xl rounded-br-md text-[15px] leading-relaxed mt-2"
                      style={{
                        padding: '14px 20px',
                        background: 'rgba(255,255,255,0.07)',
                        border: '1px solid rgba(255,255,255,0.09)',
                        color: 'rgba(255,255,255,0.88)',
                        fontFamily: 'Inter, system-ui, sans-serif',
                      }}
                    >
                      {msg.content}
                    </div>
                  )}
                </div>
              </div>
            ) : (
              /* ── AI response — left-aligned with avatar ── */
              <div className="flex items-start gap-4">
                <SatMark />
                <div className="flex-1 min-w-0 pt-1">

                  {/* Error state */}
                  {msg.taskType === 'error' ? (
                    <ErrorCard content={msg.content} onRetry={onRetry} />
                  ) : (
                    <>
                      {/* Meta — task · model · confidence */}
                      {msg.taskType && (
                        <MetaRow
                          taskType={msg.taskType}
                          modelUsed={msg.modelUsed}
                          confidence={msg.confidence}
                        />
                      )}

                      {/* Grounding: draw bounding boxes and SAM masks on the image canvas */}
                      {msg.boundingBoxes && msg.boundingBoxes.length > 0 && msg.previewUrls && msg.previewUrls[0] ? (
                        <div className="mb-5">
                          <BBoxCanvas
                            previewUrl={msg.previewUrls[0]}
                            boxes={msg.boundingBoxes}
                            masksB64={msg.masksB64}
                          />
                        </div>
                      ) : (
                        /* Non-grounding assistant output: render prominent image preview if available */
                        msg.previewUrls && msg.previewUrls.length > 0 && (
                          <div className="mb-4 flex flex-col gap-3">
                            {msg.previewUrls.map((url, idx) => (
                              <AssistantImagePreview key={idx} url={url} label={msg.uploadedFileNames?.[idx]} />
                            ))}
                          </div>
                        )
                      )}

                      {/* Main answer text — formatted */}
                      <FormattedText text={msg.content} />

                      {/* Fusion modality split */}
                      {msg.taskType === 'fusion' && msg.opticalPct !== undefined && (
                        <div className="flex gap-6 mt-4">
                          {[
                            { label: 'Optical', val: msg.opticalPct },
                            { label: 'SAR', val: msg.sarPct ?? 0 },
                          ].map(m => (
                            <div key={m.label} className="flex flex-col gap-0.5">
                              <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.30)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                                {m.label}
                              </span>
                              <span style={{ fontSize: 18, fontWeight: 600, color: 'rgba(255,255,255,0.80)' }}>
                                {Math.round(m.val * 100)}%
                              </span>
                            </div>
                          ))}
                        </div>
                      )}

                      {/* Execution trace */}
                      {msg.taskType && (
                        <ExecutionTrace
                          taskType={msg.taskType}
                          inputType={msg.inputType ?? 'Single Image'}
                          modelUsed={msg.modelUsed ?? 'InternVL2'}
                          confidence={msg.confidence ?? 0}
                          isStreaming={msg.isStreaming}
                        />
                      )}

                      {/* Export */}
                      {!msg.isStreaming && msg.taskType && (
                        <button
                          className="flex items-center gap-2 mt-4 text-[12px] font-medium transition-opacity hover:opacity-80"
                          style={{ color: 'rgba(255,255,255,0.30)', cursor: 'pointer', background: 'none', border: 'none', padding: 0 }}
                          onClick={() => {
                            // Find the user message that precedes this assistant message
                            const msgIdx = messages.indexOf(msg)
                            const userMsg = messages.slice(0, msgIdx).reverse().find(m => m.role === 'user')
                            exportMessageToPdf({
                              query: userMsg?.content ?? '',
                              answer: msg.content,
                              taskType: msg.taskType ?? 'vqa',
                              modelUsed: msg.modelUsed,
                              confidence: msg.confidence,
                              inputType: msg.inputType,
                              previewUrls: userMsg?.previewUrls,
                              resultImageUrl: msg.previewUrls?.[0],
                              boundingBoxes: msg.boundingBoxes,
                              opticalPct: msg.opticalPct,
                              sarPct: msg.sarPct,
                              timestamp: Date.now(),
                            })
                          }}
                        >
                          <Download className="w-3.5 h-3.5" />
                          Export report
                        </button>
                      )}
                    </>
                  )}
                </div>
              </div>
            )}
          </motion.div>
        ))}

        {isLoading && <LoadingRow key="loading" />}
      </AnimatePresence>
    </div>
  )
}

export default ChatThread
