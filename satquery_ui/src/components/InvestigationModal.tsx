import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  X,
  AlertTriangle,
  Clock,
  ShieldAlert,
  MessageSquarePlus
} from 'lucide-react'

interface InvestigationModalProps {
  alertId: string | null
  changeEventId?: string | null
  isOpen: boolean
  onClose: () => void
  onStartChat?: (imageIds: string[], query: string) => void
}

const CONTROLLER_URL = 'http://localhost:8000'

export default function InvestigationModal({
  alertId,
  changeEventId,
  isOpen,
  onClose,
  onStartChat,
}: InvestigationModalProps) {
  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const [activeView, setActiveView] = useState<'comparison' | 'objects' | 'mask'>('comparison')

  useEffect(() => {
    if (!isOpen) return
    const loadInvestigation = async () => {
      setLoading(true)
      setError(null)
      try {
        let endpoint = ''
        if (alertId) {
          endpoint = `${CONTROLLER_URL}/alerts/${alertId}`
        } else if (changeEventId) {
          endpoint = `${CONTROLLER_URL}/investigation/${changeEventId}`
        }

        if (!endpoint) return
        const res = await fetch(endpoint)
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const json = await res.json()
        setData(json)
      } catch (err: any) {
        console.error('Failed to load investigation payload:', err)
        setError('Failed to fetch full intelligence investigation details.')
      } finally {
        setLoading(false)
      }
    }
    loadInvestigation()
  }, [isOpen, alertId, changeEventId])

  if (!isOpen) return null

  const alert = data?.alert
  const changeEvent = data?.change_event
  const area = data?.area
  const oldObs = data?.old_observation
  const newObs = data?.new_observation
  const oldObjects = data?.old_objects || []
  const newObjects = data?.new_objects || []

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
        {/* Backdrop */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
          className="fixed inset-0 bg-black/20 z-50"
        />

        {/* Modal Window — ONE single seamless translucent glass box */}
        <motion.div
          initial={{ scale: 0.96, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.96, opacity: 0 }}
          style={{
            background: 'rgba(31, 31, 31, 0.38)',
            backdropFilter: 'blur(16px)',
            WebkitBackdropFilter: 'blur(16px)',
            border: '1px solid #333333',
            boxShadow: '0 12px 48px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06)',
          }}
          className="relative w-full max-w-5xl max-h-[85vh] rounded-3xl overflow-hidden flex flex-col z-50 text-slate-100 font-sans"
        >
          {/* Header */}
          <div
            style={{ padding: '28px 56px 14px 56px' }}
            className="flex items-center justify-between shrink-0"
          >
            <div className="flex items-center gap-3.5">
              <div
                style={{
                  background: 'rgba(255, 255, 255, 0.06)',
                  padding: '10px',
                  borderRadius: '14px',
                }}
                className="text-slate-200"
              >
                <ShieldAlert className="w-6 h-6 text-amber-400" />
              </div>
              <div>
                <h2 className="text-lg font-bold text-white flex items-center gap-2">
                  {alert?.title || changeEvent?.summary || 'Geospatial Change Investigation'}
                </h2>
                <div className="flex items-center gap-2.5 text-xs text-slate-300 mt-0.5 font-mono">
                  <span>Area: <span className="text-slate-100">{area?.area_id || 'Unknown'}</span></span>
                  <span>•</span>
                  <span>Sensor: <span className="text-slate-100">{newObs?.sensor || 'Optical'}</span></span>
                  <span>•</span>
                  <span>Resolution: <span className="text-slate-100">{newObs?.resolution_m ? `${newObs.resolution_m}m` : 'N/A'}</span></span>
                </div>
              </div>
            </div>
            <button
              onClick={onClose}
              style={{
                padding: '8px 12px',
                borderRadius: '12px',
                background: 'rgba(255, 255, 255, 0.08)',
                color: '#d1d5db',
              }}
              className="hover:bg-white/15 hover:text-white transition-all cursor-pointer border-none"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Navigation View Tabs */}
          <div
            style={{ padding: '8px 56px' }}
            className="flex items-center justify-between text-xs shrink-0"
          >
            <div className="flex items-center gap-2">
              <button
                onClick={() => setActiveView('comparison')}
                style={{
                  padding: '7px 18px',
                  fontSize: 13,
                  borderRadius: 9999,
                  background: activeView === 'comparison' ? 'rgba(255,255,255,0.14)' : 'transparent',
                  color: activeView === 'comparison' ? '#ffffff' : '#9ca3af',
                }}
                className="font-medium transition-all cursor-pointer border-none"
              >
                Bi-Temporal Observation Comparison
              </button>
              <button
                onClick={() => setActiveView('objects')}
                style={{
                  padding: '7px 18px',
                  fontSize: 13,
                  borderRadius: 9999,
                  background: activeView === 'objects' ? 'rgba(255,255,255,0.14)' : 'transparent',
                  color: activeView === 'objects' ? '#ffffff' : '#9ca3af',
                }}
                className="font-medium transition-all cursor-pointer border-none"
              >
                Object Delta Analysis ({oldObjects.length} → {newObjects.length})
              </button>
            </div>

            {changeEvent?.confidence && (
              <div className="text-slate-300 font-mono text-xs">
                Change Confidence:{' '}
                <strong className="text-white font-bold text-sm">
                  {Math.round(changeEvent.confidence * 100)}%
                </strong>
              </div>
            )}
          </div>

          {/* Modal Body */}
          <div
            style={{ padding: '20px 56px' }}
            className="flex-1 overflow-y-auto space-y-6"
          >
            {loading && (
              <div className="py-20 text-center text-slate-300 flex flex-col items-center gap-3">
                <div className="w-8 h-8 border-2 border-white border-t-transparent rounded-full animate-spin" />
                <p className="text-sm">Loading intelligence evidence payload...</p>
              </div>
            )}

            {error && (
              <div
                style={{
                  background: 'rgba(239, 68, 68, 0.12)',
                  borderRadius: '14px',
                  padding: '16px 20px',
                }}
                className="text-sm text-red-300 flex items-center gap-3"
              >
                <AlertTriangle className="w-5 h-5 text-red-400 shrink-0" />
                <span>{error}</span>
              </div>
            )}

            {!loading && !error && data && (
              <>
                {/* Comparison View */}
                {activeView === 'comparison' && (
                  <div className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {/* Old Observation (t-1) */}
                      <div className="space-y-2.5">
                        <div className="flex items-center justify-between text-xs px-1">
                          <span className="font-semibold text-white flex items-center gap-1.5">
                            <Clock className="w-3.5 h-3.5 text-slate-400" />
                            Previous Observation (t-1)
                          </span>
                          <span className="text-slate-300 font-mono">
                            {oldObs?.created_at?.slice(0, 10) || 'Previous Date'}
                          </span>
                        </div>
                        <div className="h-60 bg-black/40 rounded-xl overflow-hidden flex items-center justify-center relative">
                          {oldObs?.image_path ? (
                            <img
                              src={`http://localhost:8200/file?path=${encodeURIComponent(oldObs.image_path)}`}
                              onError={(e: any) => {
                                e.target.onerror = null
                                e.target.src = 'https://placehold.co/600x400/0f172a/94a3b8?text=Observation+Image+1'
                              }}
                              alt="Observation t-1"
                              className="w-full h-full object-cover"
                            />
                          ) : (
                            <span className="text-xs text-slate-400">No Image Available</span>
                          )}
                          <div
                            style={{ background: 'rgba(0,0,0,0.6)' }}
                            className="absolute bottom-2 left-2 px-2.5 py-0.5 rounded-md text-[11px] text-slate-300 font-mono"
                          >
                            ID: {oldObs?.observation_id?.slice(0, 8)}
                          </div>
                        </div>
                      </div>

                      {/* New Observation (t) */}
                      <div className="space-y-2.5">
                        <div className="flex items-center justify-between text-xs px-1">
                          <span className="font-semibold text-white flex items-center gap-1.5">
                            <Clock className="w-3.5 h-3.5 text-slate-400" />
                            Latest Observation (t)
                          </span>
                          <span className="text-slate-300 font-mono">
                            {newObs?.created_at?.slice(0, 10) || 'Latest Date'}
                          </span>
                        </div>
                        <div className="h-60 bg-black/40 rounded-xl overflow-hidden flex items-center justify-center relative">
                          {newObs?.image_path ? (
                            <img
                              src={`http://localhost:8200/file?path=${encodeURIComponent(newObs.image_path)}`}
                              onError={(e: any) => {
                                e.target.onerror = null
                                e.target.src = 'https://placehold.co/600x400/0f172a/94a3b8?text=Observation+Image+2'
                              }}
                              alt="Observation t"
                              className="w-full h-full object-cover"
                            />
                          ) : (
                            <span className="text-xs text-slate-400">No Image Available</span>
                          )}
                          <div
                            style={{ background: 'rgba(0,0,0,0.6)' }}
                            className="absolute bottom-2 left-2 px-2.5 py-0.5 rounded-md text-[11px] text-slate-300 font-mono"
                          >
                            ID: {newObs?.observation_id?.slice(0, 8)}
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Change Assessment Details */}
                    <div className="space-y-2 text-xs pt-2">
                      <h4 className="font-bold text-sm text-white">Intelligence Finding Summary</h4>
                      <p className="text-slate-200 leading-relaxed font-normal">
                        {changeEvent?.summary || alert?.description || 'No detailed change summary recorded.'}
                      </p>
                      {changeEvent?.pixel_change_ratio !== undefined && (
                        <div className="pt-2 flex items-center gap-5 text-slate-300 font-mono text-xs">
                          <span>
                            Pixel Change Ratio:{' '}
                            <strong className="text-amber-300">
                              {(changeEvent.pixel_change_ratio * 100).toFixed(2)}%
                            </strong>
                          </span>
                          <span>
                            Change Category:{' '}
                            <strong className="text-white capitalize">
                              {changeEvent.change_type}
                            </strong>
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Objects View */}
                {activeView === 'objects' && (
                  <div className="space-y-4 text-xs">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {/* Old Objects */}
                      <div>
                        <h4 className="font-bold text-sm text-white mb-3 flex items-center justify-between">
                          <span>Previous Objects ({oldObjects.length})</span>
                        </h4>
                        <div className="space-y-2 max-h-56 overflow-y-auto pr-1">
                          {oldObjects.map((obj: any, idx: number) => (
                            <div
                              key={idx}
                              style={{ background: 'rgba(255,255,255,0.05)' }}
                              className="p-2.5 rounded-lg flex justify-between items-center"
                            >
                              <span className="font-medium text-slate-100 text-xs">{obj.category}</span>
                              <span className="text-slate-400 font-mono text-[11px]">
                                Conf: {Math.round(obj.confidence * 100)}%
                              </span>
                            </div>
                          ))}
                          {oldObjects.length === 0 && (
                            <p className="text-slate-400 italic">No objects extracted from t-1.</p>
                          )}
                        </div>
                      </div>

                      {/* New Objects */}
                      <div>
                        <h4 className="font-bold text-sm text-white mb-3 flex items-center justify-between">
                          <span>Latest Objects ({newObjects.length})</span>
                        </h4>
                        <div className="space-y-2 max-h-56 overflow-y-auto pr-1">
                          {newObjects.map((obj: any, idx: number) => (
                            <div
                              key={idx}
                              style={{ background: 'rgba(255,255,255,0.05)' }}
                              className="p-2.5 rounded-lg flex justify-between items-center"
                            >
                              <span className="font-medium text-slate-100 text-xs">{obj.category}</span>
                              <span className="text-slate-400 font-mono text-[11px]">
                                Conf: {Math.round(obj.confidence * 100)}%
                              </span>
                            </div>
                          ))}
                          {newObjects.length === 0 && (
                            <p className="text-slate-400 italic">No objects extracted from t.</p>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>

          {/* Footer Actions */}
          <div
            style={{ padding: '20px 56px 28px 56px' }}
            className="flex items-center justify-between text-xs shrink-0"
          >
            <span className="text-slate-300 font-mono text-xs">
              Status:{' '}
              <strong className="text-white capitalize font-bold text-xs">
                {alert?.status || 'Active'}
              </strong>
            </span>

            <div className="flex items-center gap-3">
              {onStartChat && (oldObs?.image_path || newObs?.image_path) && (
                <button
                  onClick={() => {
                    const imageIds: string[] = []
                    if (oldObs?.image_path) imageIds.push(oldObs.image_path)
                    if (newObs?.image_path) imageIds.push(newObs.image_path)
                    const query = `Analyze the changes detected in area ${area?.area_id || 'this region'}. ${changeEvent?.summary || alert?.description || ''}`
                    onStartChat(imageIds, query)
                    onClose()
                  }}
                  style={{
                    padding: '8px 22px',
                    fontSize: 13,
                    borderRadius: 9999,
                    background: 'rgba(16, 185, 129, 0.2)',
                    border: '1px solid rgba(16, 185, 129, 0.4)',
                    color: '#6ee7b7',
                  }}
                  className="font-medium hover:bg-emerald-500/30 transition-all cursor-pointer flex items-center gap-2"
                >
                  <MessageSquarePlus className="w-4 h-4" />
                  Investigate in Chat
                </button>
              )}
              <button
                onClick={onClose}
                style={{
                  padding: '8px 22px',
                  fontSize: 13,
                  borderRadius: 9999,
                  background: 'rgba(255, 255, 255, 0.12)',
                  color: '#ffffff',
                }}
                className="font-medium hover:bg-white/20 transition-all cursor-pointer border-none"
              >
                Close Window
              </button>
            </div>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  )
}
