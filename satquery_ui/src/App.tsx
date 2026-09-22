import { useState, useEffect, useCallback, useRef } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import HeroSection from './components/HeroSection'
import WorkspaceView from './components/WorkspaceView'
import AlertsPanel from './components/AlertsPanel'
import InvestigationModal from './components/InvestigationModal'
import './index.css'

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

interface WorkspaceMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  taskType?: string
  inputType?: string
  modelUsed?: string
  confidence?: number
  previewUrls?: string[]          // blob: URLs — session-only, not persisted
  uploadedFileNames?: string[]    // parallel array, persisted for display
  boundingBoxes?: BoundingBox[]
  masksB64?: string[]             // SAM segmentation masks (base64 PNG)
  opticalPct?: number
  sarPct?: number
  isStreaming?: boolean
}

interface ChatHistoryItem {
  id: string
  title: string
  messages?: WorkspaceMessage[] // stored without previewUrls (those are ephemeral)
  timestamp?: number
}

/* ─────────────────────────────────────────────────────────
   Helpers
───────────────────────────────────────────────────────── */
const uid = () => Math.random().toString(36).slice(2)

const LS_KEY = 'satquery_conversations'

function loadConversations(): ChatHistoryItem[] {
  try {
    const raw = localStorage.getItem(LS_KEY)
    return raw ? JSON.parse(raw) : []
  } catch { return [] }
}

function saveConversations(items: ChatHistoryItem[]) {
  try {
    // Keep data: and http: URLs (they survive refresh), strip only ephemeral blob: URLs
    const sanitized = items.slice(0, 50).map(conv => ({
      ...conv,
      messages: conv.messages?.map(m => {
        const persistable = m.previewUrls?.filter(u => !u.startsWith('blob:'))
        return { ...m, previewUrls: persistable && persistable.length > 0 ? persistable : undefined }
      }),
    }))
    localStorage.setItem(LS_KEY, JSON.stringify(sanitized))
  } catch { /* storage full */ }
}

/* ─────────────────────────────────────────────────────────
   Task detection — from query keywords and file count
───────────────────────────────────────────────────────── */
function detectTaskType(
  query: string,
  files: File[],
  selectedTask: string | null,
): 'vqa' | 'caption' | 'change' | 'fusion' | 'grounding' {
  if (selectedTask === 'vqa') return 'vqa'
  if (selectedTask === 'caption') return 'caption'
  if (selectedTask === 'change') return 'change'
  if (selectedTask === 'fusion') return 'fusion'
  if (selectedTask === 'grounding') return 'grounding'

  const q = query.toLowerCase()
  const fileNames = files.map(f => f.name.toLowerCase()).join(' ')

  // Grounding & object detection keywords
  if (/\b(high?ligh?t|hightlight|hilight|heighlight|highlite|locate|mark|find|where|show|draw|outline|bbox|bounding|detect|detection|spot|search|point|identify|grounding|segment|segmentation|mask|water body|water bodies|water detection|building detection|house detection|structure detection|how many|count|amount|number of|how much|quantity)\b/i.test(q)) return 'grounding'

  // Fusion / SAR — check query & file pair names (S1 + S2 or SAR + Optical)
  if (
    /\b(sar|radar|fusion|optical.*sar|sar.*optical|sentinel-1|s1)\b/.test(q) ||
    ((/\b(sar|s1|radar)\b/.test(fileNames) || files.length >= 2) && /\b(fusion|optical.*sar|sar.*optical|s1.*s2|s2.*s1)\b/.test(q)) ||
    (/\b(sar|s1|radar)\b/.test(fileNames) && /\b(optical|s2|rgb|hand)\b/.test(fileNames))
  ) {
    return 'fusion'
  }

  // Change detection — 2 bi-temporal images or temporal change keywords
  if (files.length >= 2 || /\b(change|before|after|compare|difference|temporal|between)\b/.test(q)) return 'change'

  // Caption: only trigger when the user explicitly asks for a caption/summary.
  if (/\b(caption|summarize the image|give me a caption)\b/.test(q)) return 'caption'

  // Default: VQA
  return 'vqa'
}

function getInputType(files: File[], task: string): string {
  if (task === 'change') return 'Bi-temporal Optical'
  if (task === 'fusion') return 'Optical + SAR Pair'
  return files.length > 0 ? 'Uploaded Image' : 'Single Image'
}

function getModelName(task: string): string {
  const m: Record<string, string> = {
    vqa: 'Prithvi-100M + TagHead + Mistral',
    caption: 'Prithvi-100M + TagHead + Mistral',
    change: 'Prithvi-100M Siamese + ChangeHead',
    fusion: 'Prithvi-100M + SAR Encoder + FusionHead',
    grounding: 'Grounding DINO SwinT-OGC',
  }
  return m[task] ?? 'Prithvi-100M'
}


function makeChatTitle(query: string, files: File[]): string {
  if (query.trim().length > 0) return query.trim().slice(0, 48) + (query.length > 48 ? '…' : '')
  if (files.length > 0) return `Image analysis — ${files[0].name.split('.')[0]}`
  return 'New analysis'
}

function groupChatHistory(history: ChatHistoryItem[]): Record<string, ChatHistoryItem[]> {
  if (history.length === 0) return {}
  const now = Date.now(), ONE_DAY = 86_400_000
  const today: ChatHistoryItem[] = [], yesterday: ChatHistoryItem[] = [], older: ChatHistoryItem[] = []
  for (const item of history) {
    const age = now - (item.timestamp ?? now)
    if (age < ONE_DAY) today.push(item)
    else if (age < 2 * ONE_DAY) yesterday.push(item)
    else older.push(item)
  }
  const result: Record<string, ChatHistoryItem[]> = {}
  if (today.length) result['Today'] = today
  if (yesterday.length) result['Yesterday'] = yesterday
  if (older.length) result['Older'] = older
  return result
}

/* ─────────────────────────────────────────────────────────
   Query: pass raw user question to the model.
   Anti-hallucination constraints live in the inference
   layer (vqa.py / caption.py), not here.
───────────────────────────────────────────────────────── */
function augmentQuery(query: string, _task: string): string {
  return query.trim()
}

/* ─────────────────────────────────────────────────────────
   Parse grounding bounding boxes from controller response
───────────────────────────────────────────────────────── */
function parseGroundingBboxes(data: any): BoundingBox[] {
  const facts = data.facts ?? {}
  const boxes: BoundingBox[] = []

  // Image dimensions for normalizing pixel coordinates → 0-1
  const imgW = facts.image_width ?? 1
  const imgH = facts.image_height ?? 1

  // Primary: ML server grounding → facts.bounding_boxes = [{x_min, y_min, x_max, y_max, label, confidence}]
  if (Array.isArray(facts.bounding_boxes)) {
    for (const obj of facts.bounding_boxes) {
      if (obj.x_min !== undefined) {
        // Normalize pixel coordinates to 0-1 range for canvas rendering
        boxes.push({
          label: obj.label ?? 'Object',
          x1: obj.x_min / imgW,
          y1: obj.y_min / imgH,
          x2: obj.x_max / imgW,
          y2: obj.y_max / imgH,
          confidence: obj.confidence,
        })
      }
    }
  }

  // Legacy: facts.objects with x1/y1/x2/y2 (already normalized)
  if (boxes.length === 0 && Array.isArray(facts.objects)) {
    for (const obj of facts.objects) {
      if (obj.x1 !== undefined) {
        boxes.push({
          label: obj.label ?? 'Object',
          x1: obj.x1, y1: obj.y1,
          x2: obj.x2, y2: obj.y2,
          confidence: obj.confidence,
        })
      }
    }
  }

  // Fallback: evidence array
  if (boxes.length === 0 && Array.isArray(data.evidence)) {
    for (const ev of data.evidence) {
      if (ev.type === 'bounding_box' && ev.data) {
        const d = ev.data
        boxes.push({
          label: d.label ?? 'Detection',
          x1: d.x1 ?? 0, y1: d.y1 ?? 0,
          x2: d.x2 ?? 1, y2: d.y2 ?? 1,
          confidence: d.confidence,
        })
      }
    }
  }

  return boxes
}

/* ─────────────────────────────────────────────────────────
   Build human-readable grounding summary text
───────────────────────────────────────────────────────── */
function groundingToText(boxes: BoundingBox[], query: string): string {
  if (boxes.length === 0) {
    return `No specific objects were confidently detected for "${query}". The model could not localise the requested feature with sufficient precision in this image.`
  }

  const lines = boxes.map((b, _i) => {
    const pct = b.confidence !== undefined ? ` (${Math.round(b.confidence * 100)}% confidence)` : ''
    const loc = `top-left (${Math.round(b.x1 * 100)}%, ${Math.round(b.y1 * 100)}%) → bottom-right (${Math.round(b.x2 * 100)}%, ${Math.round(b.y2 * 100)}%)`
    return `- ${b.label}${pct}: located at ${loc}`
  })

  return (
    `Detected ${boxes.length} instance${boxes.length > 1 ? 's' : ''} in the image:\n\n` +
    lines.join('\n') +
    `\n\nThe bounding box${boxes.length > 1 ? 'es are' : ' is'} shown overlaid on the image above.`
  )
}

/* ─────────────────────────────────────────────────────────
   App Component
───────────────────────────────────────────────────────── */
export function App() {
  const [view, setView] = useState<'hero' | 'workspace'>('hero')
  const [messages, setMessages] = useState<WorkspaceMessage[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null)
  const [chatHistory, setChatHistory] = useState<ChatHistoryItem[]>(() => loadConversations())
  const [activeConversationId, setActiveConversationId] = useState<string | undefined>()

  /* ── PGIL Spatial Intelligence state ── */
  const [isAlertsOpen, setIsAlertsOpen] = useState(false)
  const [pendingAlertCount, setPendingAlertCount] = useState(0)
  const [selectedAlertId, setSelectedAlertId] = useState<string | null>(null)
  const [selectedChangeEventId, setSelectedChangeEventId] = useState<string | null>(null)
  const [isInvestigationOpen, setIsInvestigationOpen] = useState(false)

  const handleSelectAlert = (alertId: string, changeEventId?: string) => {
    setSelectedAlertId(alertId)
    setSelectedChangeEventId(changeEventId || null)
    setIsInvestigationOpen(true)
  }

  // Track the last submitted query for retry
  const lastSubmitRef = useRef<{ query: string; files: File[]; task: string | null } | null>(null)
  // Track all blob URLs for cleanup
  const blobUrlsRef = useRef<string[]>([])
  // Track the last uploaded image IDs so follow-up queries can reuse them
  const lastImageIdsRef = useRef<string[]>([])
  // ★ Synchronous refs for conversation isolation — prevents async responses
  //   from leaking into the wrong conversation when the user switches mid-inference
  const messagesRef = useRef<WorkspaceMessage[]>([])
  const activeConvIdRef = useRef<string | undefined>()

  /* ── Health check — poll every 30s ── */
  useEffect(() => {
    const check = () =>
      fetch('http://localhost:8200/health').then(r => setBackendOnline(r.ok)).catch(() => setBackendOnline(false))
    check()
    const t = setInterval(check, 30_000)
    return () => clearInterval(t)
  }, [])

  /* ── Persist conversations (without blob URLs) ── */
  useEffect(() => { saveConversations(chatHistory) }, [chatHistory])

  /* ── Keep synchronous refs in sync with React state ── */
  useEffect(() => { messagesRef.current = messages }, [messages])
  useEffect(() => { activeConvIdRef.current = activeConversationId }, [activeConversationId])

  /* ── Revoke blob URLs when clearing messages ── */
  const revokeBlobUrls = useCallback(() => {
    blobUrlsRef.current.forEach(url => URL.revokeObjectURL(url))
    blobUrlsRef.current = []
  }, [])

  /* ── Upload files to ML server, get absolute-path image_ids back ── */
  const uploadFiles = useCallback(async (files: File[]): Promise<string[]> => {
    if (files.length === 0) {
      // No file uploaded — use default demo patch ID
      return ['S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_55_90']
    }
    const ids: string[] = []
    for (const file of files) {
      const form = new FormData()
      form.append('file', file, file.name)
      const res = await fetch('http://localhost:8200/upload', { method: 'POST', body: form })
      if (!res.ok) {
        const err = await res.text().catch(() => res.statusText)
        throw new Error(`Image upload failed (${res.status}): ${err}`)
      }
      const data = await res.json()
      ids.push(data.image_id)  // absolute path on ML server disk
    }
    return ids
  }, [])

  /* ── Core inference — routes directly to ML server endpoints ── */
  const runInference = useCallback(async (
    query: string,
    files: File[],
    task: 'vqa' | 'caption' | 'change' | 'fusion' | 'grounding',
    previewUrls: string[],
    existingImageIds?: string[],
  ): Promise<WorkspaceMessage> => {
    // Step 1: Upload real files → get absolute paths as image_ids
    // If no new files but we have previously uploaded IDs, reuse them
    let imageIds: string[]
    if (files.length > 0) {
      imageIds = await uploadFiles(files)
    } else if (existingImageIds && existingImageIds.length > 0) {
      imageIds = existingImageIds
    } else {
      imageIds = await uploadFiles(files) // falls back to demo ID
    }

    // Store for follow-up queries
    lastImageIdsRef.current = imageIds

    const augmentedQuery = augmentQuery(query, task)

    // Route to the correct ML server endpoint based on task
    const ML_BASE = 'http://localhost:8200'
    const endpoint = `${ML_BASE}/${task}`

    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        request_id: crypto.randomUUID(),
        query: augmentedQuery,
        image_ids: imageIds,
        task_type: task,
      }),
    })

    if (!res.ok) {
      const errBody = await res.text().catch(() => res.statusText)
      throw new Error(`HTTP ${res.status}: ${errBody}`)
    }
    const data = await res.json()

    if (data.status === 'failed' || data.status === 'error') {
      throw new Error(data.error || `Server returned ${data.status}`)
    }

    // MLResponse schema: { facts, confidence, model_version, evidence }
    const facts = data.facts ?? {}
    const confidenceScore = typeof data.confidence === 'number'
      ? data.confidence
      : (typeof data.confidence === 'object' ? data.confidence?.score : 0)

    // Priority chain: annotated > change_map > converted > user preview
    const resultImageUrl =
      facts.annotated_image_b64 ||
      facts.change_map_b64 ||
      facts.converted_image_b64 ||
      (previewUrls.length > 0 ? previewUrls[0] : undefined)

    // ── Grounding: extract bboxes, build human-readable text
    if (task === 'grounding') {
      const bboxes = parseGroundingBboxes(data)
      const text = facts.answer || groundingToText(bboxes, query)
      const masksB64 = Array.isArray(facts.masks_b64) ? facts.masks_b64 : undefined
      return {
        id: uid(), role: 'assistant',
        content: text,
        taskType: 'grounding',
        inputType: getInputType(files, task),
        modelUsed: data.model_version || getModelName(task),
        confidence: confidenceScore,
        boundingBoxes: bboxes,
        masksB64,
        previewUrls: resultImageUrl ? [resultImageUrl] : undefined,
      }
    }

    // ── All other tasks: extract answer text from facts
    const answer =
      facts.answer ||
      facts.fused_answer ||
      facts.caption ||
      facts.description ||
      (typeof facts === 'string' ? facts : null) ||
      JSON.stringify(facts)

    return {
      id: uid(), role: 'assistant',
      content: answer,
      taskType: task,
      inputType: getInputType(files, task),
      modelUsed: data.model_version || getModelName(task),
      confidence: confidenceScore,
      opticalPct: facts.optical_contribution,
      sarPct: facts.sar_contribution,
      previewUrls: resultImageUrl ? [resultImageUrl] : undefined,
    }
  }, [uploadFiles])

  /* ── Core submit / execute ── */
  const executeSubmit = useCallback(async (
    query: string,
    files: File[],
    selectedTask: string | null,
    existingConvId?: string,
  ) => {
    // Detect task locally FIRST (instant, no network)
    let task: 'vqa' | 'caption' | 'change' | 'fusion' | 'grounding' = detectTaskType(query, files, selectedTask)

    // Pre-flight: change detection needs 2 images
    if (task === 'change' && files.length > 0 && files.length < 2) {
      const userMsg: WorkspaceMessage = { id: uid(), role: 'user', content: query }
      const errMsg: WorkspaceMessage = {
        id: uid(), role: 'assistant',
        content: 'Change detection requires two images: Before and After.',
        taskType: 'error',
      }
      setMessages(prev => [...prev, userMsg, errMsg])
      setView('workspace')
      return
    }

    // One blob URL per file — ChatThread shows <img> for web formats, FileBadge for TIFF etc.
    const allPreviewUrls = files.map(f => URL.createObjectURL(f))
    const allFileNames    = files.map(f => f.name)
    blobUrlsRef.current.push(...allPreviewUrls)

    const userMsg: WorkspaceMessage = {
      id: uid(), role: 'user',
      content: query || `[Uploaded ${files.length} image${files.length > 1 ? 's' : ''}]`,
      previewUrls:       allPreviewUrls.length > 0 ? allPreviewUrls : undefined,
      uploadedFileNames: allFileNames.length   > 0 ? allFileNames   : undefined,
    }

    // Create or reuse conversation
    const convId = existingConvId ?? uid()
    if (!existingConvId) {
      const newEntry: ChatHistoryItem = {
        id: convId,
        title: makeChatTitle(query, files),
        timestamp: Date.now(),
        messages: [userMsg],
      }
      setChatHistory(prev => [newEntry, ...prev])
      setActiveConversationId(convId)
      activeConvIdRef.current = convId  // sync immediately, don't wait for useEffect
    }

    // ★ Capture a SNAPSHOT of this conversation's messages at submit time.
    //   This snapshot is used later so the async response always goes to the
    //   correct conversation, even if the user switches away mid-inference.
    const convMessages = [...messagesRef.current, userMsg]

    // ★ INSTANT transition — move to workspace and show thinking animation
    setMessages(convMessages)
    messagesRef.current = convMessages
    setView('workspace')
    setIsLoading(true)

    // Task routing: detectTaskType() provides instant, reliable routing via regex.
    // The /route LLM endpoint is NOT called here — it monopolises the GPU for ~15s
    // and always falls back to VQA anyway (the EOV2B model returns prose, not JSON).
    // LLM-based routing lives server-side in /analyze for non-streaming tasks.

    // Reuse previously uploaded image IDs for follow-up queries in the same conversation
    const inheritedImageIds = files.length === 0 ? lastImageIdsRef.current : undefined

    try {
      const aiMsg = await runInference(query, files, task as any, allPreviewUrls, inheritedImageIds)

      // ★ Build final messages from the SNAPSHOT (not from current state)
      const finalMessages = [...convMessages, aiMsg]

      // ★ Always update chatHistory for the correct conversation
      setChatHistory(hist =>
        hist.map(h => h.id === convId ? { ...h, messages: finalMessages } : h)
      )

      // ★ Only update displayed messages if user is still viewing THIS conversation
      if (activeConvIdRef.current === convId) {
        setMessages(finalMessages)
        messagesRef.current = finalMessages
      }
    } catch (err: any) {
      const errMsg: WorkspaceMessage = {
        id: uid(), role: 'assistant',
        content: err.message ?? String(err),
        taskType: 'error',
      }
      const finalMessages = [...convMessages, errMsg]

      // Only update display if still on this conversation
      if (activeConvIdRef.current === convId) {
        setMessages(finalMessages)
        messagesRef.current = finalMessages
      }
    } finally {
      // Only clear loading if still on this conversation
      if (activeConvIdRef.current === convId) {
        setIsLoading(false)
      }
    }
  }, [runInference])

  /* ── Public submit handler ── */
  const handleSubmit = useCallback((
    query: string,
    files: File[],
    selectedTask: string | null,
  ) => {
    lastSubmitRef.current = { query, files, task: selectedTask }
    executeSubmit(query, files, selectedTask, view === 'workspace' ? activeConversationId : undefined)
  }, [view, activeConversationId, executeSubmit])

  /* ── Retry ── */
  const handleRetry = useCallback(() => {
    if (!lastSubmitRef.current) return
    const { query, files, task } = lastSubmitRef.current
    setMessages(prev => prev.slice(0, -1)) // remove last error
    setTimeout(() => executeSubmit(query, files, task, activeConversationId), 100)
  }, [activeConversationId, executeSubmit])

  /* ── New analysis — revoke blob URLs ── */
  const handleNewAnalysis = useCallback(() => {
    revokeBlobUrls()
    setMessages([])
    messagesRef.current = []
    setActiveConversationId(undefined)
    activeConvIdRef.current = undefined
    lastSubmitRef.current = null
    setView('hero')
  }, [revokeBlobUrls])

  /* ── Restore past conversation ── */
  const handleSelectConversation = useCallback((id: string) => {
    const conv = chatHistory.find(h => h.id === id)
    if (!conv) return
    const restored = conv.messages ?? []
    setMessages(restored)
    messagesRef.current = restored
    setActiveConversationId(id)
    activeConvIdRef.current = id
    setIsLoading(false)   // cancel any loading spinner from previous conversation
    setView('workspace')
  }, [chatHistory])

  const groupedHistory = groupChatHistory(chatHistory)

  return (
    <>
      {/* ── Global video background — never unmounted ── */}
      <video
        autoPlay loop muted playsInline
        className={`video-bg ${view === 'workspace' ? 'workspace-mode' : ''}`}
      >
        <source
          src="https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260315_073750_51473149-4350-4920-ae24-c8214286f323.mp4"
          type="video/mp4"
        />
      </video>

      {/* ── PGIL Intelligence Alerts & GPU Status pill header bar ── */}
      <div style={{
        position: 'fixed', top: 12, right: 16, zIndex: 9999,
        display: 'flex', alignItems: 'center', gap: 10,
      }}>
        {/* Spatial Intelligence Bell Button */}
        <button
          onClick={() => setIsAlertsOpen(true)}
          style={{
            display: 'flex', alignItems: 'center', gap: 6,
            padding: '5px 12px', borderRadius: 9999,
            background: 'rgba(12,12,12,0.85)',
            border: '1px solid rgba(16,185,129,0.3)',
            backdropFilter: 'blur(12px)',
            WebkitBackdropFilter: 'blur(12px)',
            color: '#e2e8f0', cursor: 'pointer', fontSize: 11,
            fontFamily: 'Inter, system-ui, sans-serif', fontWeight: 600,
            transition: 'all 0.2s ease',
          }}
          title="PGIL Spatial Intelligence Alerts"
        >
          <svg style={{ width: 14, height: 14, color: '#10b981' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
          </svg>
          <span>PGIL Alerts</span>
          {pendingAlertCount > 0 && (
            <span style={{
              background: '#ef4444', color: '#ffffff',
              fontSize: 10, fontWeight: 700, padding: '1px 6px',
              borderRadius: 9999, lineHeight: 1,
            }}>
              {pendingAlertCount}
            </span>
          )}
        </button>

        {/* Backend / GPU status pill */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 6,
          padding: '5px 12px', borderRadius: 9999,
          background: 'rgba(12,12,12,0.80)',
          border: '1px solid rgba(255,255,255,0.08)',
          backdropFilter: 'blur(12px)',
          WebkitBackdropFilter: 'blur(12px)',
        }}>
          <span style={{
            width: 6, height: 6, borderRadius: '50%', flexShrink: 0,
            background: backendOnline === true ? '#86efac' : backendOnline === false ? '#f87171' : '#fb923c',
          }} />
          <span style={{
            color: 'rgba(255,255,255,0.40)', fontSize: 11,
            fontFamily: 'Inter, system-ui, sans-serif', fontWeight: 500, letterSpacing: '0.02em',
          }}>
            GPU {backendOnline === true ? 'Online' : backendOnline === false ? 'Offline' : 'Checking…'}
          </span>
        </div>
      </div>

      {/* ── View layer ── */}
      <div style={{ position: 'relative', zIndex: 10, width: '100vw', height: '100vh', overflow: 'hidden' }}>
        <AnimatePresence mode="wait">
          {view === 'hero' ? (
            <motion.div
              key="hero"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0, scale: 0.99, filter: 'blur(6px)' }}
              transition={{ duration: 0.5, ease: 'easeInOut' }}
              style={{ width: '100%', height: '100%' }}
            >
              <HeroSection onSubmit={handleSubmit} githubUrl="https://github.com" />
            </motion.div>
          ) : (
            <motion.div
              key="workspace"
              initial={{ opacity: 0, filter: 'blur(6px)' }}
              animate={{ opacity: 1, filter: 'blur(0px)' }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
              style={{ width: '100%', height: '100%', overflow: 'hidden' }}
            >
              <WorkspaceView
                messages={messages}
                isLoading={isLoading}
                onNewMessage={handleSubmit}
                onNewAnalysis={handleNewAnalysis}
                chatHistory={groupedHistory}
                activeConversationId={activeConversationId}
                onSelectConversation={handleSelectConversation}
                onRetry={handleRetry}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* ── PGIL Spatial Intelligence Slide-Out & Investigation Modal ── */}
      <AlertsPanel
        isOpen={isAlertsOpen}
        onClose={() => setIsAlertsOpen(false)}
        onSelectAlert={handleSelectAlert}
        onUpdateCount={setPendingAlertCount}
      />

      <InvestigationModal
        isOpen={isInvestigationOpen}
        onClose={() => setIsInvestigationOpen(false)}
        alertId={selectedAlertId}
        changeEventId={selectedChangeEventId}
      />
    </>
  )
}

export default App

