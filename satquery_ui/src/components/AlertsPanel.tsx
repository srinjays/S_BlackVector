import React, { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Bell,
  X,
  AlertTriangle,
  Filter,
  ArrowRight,
  RotateCw,
  ShieldAlert
} from 'lucide-react'

export interface PGILAlert {
  alert_id: string
  area_id: string
  alert_type: string
  title: string
  description: string
  severity: 'critical' | 'high' | 'medium' | 'low' | string
  status: 'pending' | 'confirmed' | 'dismissed' | string
  confidence: number
  observation_interval?: string
  affected_count?: number
  change_event_id?: string
  created_at?: string
  // Image paths from associated observations (enriched by controller)
  old_image_path?: string
  new_image_path?: string
  old_image_url?: string
  new_image_url?: string
}

interface AlertsPanelProps {
  isOpen: boolean
  onClose: () => void
  onSelectAlert: (alertId: string, changeEventId?: string) => void
  onUpdateCount?: (count: number) => void
}

const CONTROLLER_URL = 'http://localhost:8000'

export default function AlertsPanel({
  isOpen,
  onClose,
  onSelectAlert,
  onUpdateCount,
}: AlertsPanelProps) {
  const [alerts, setAlerts] = useState<PGILAlert[]>([])
  const [filter, setFilter] = useState<'all' | 'pending' | 'critical'>('pending')
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)

  const fetchAlerts = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${CONTROLLER_URL}/alerts/all`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      const list: PGILAlert[] = data.alerts || []
      setAlerts(list)
      const pendingCount = list.filter(a => a.status === 'pending').length
      if (onUpdateCount) onUpdateCount(pendingCount)
    } catch (err: any) {
      console.warn('Failed to fetch alerts:', err)
      setError('Could not connect to PGIL intelligence engine.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (isOpen) {
      fetchAlerts()
    }
  }, [isOpen])

  // Periodic polling every 10 seconds for new background alerts
  useEffect(() => {
    fetchAlerts()
    const timer = setInterval(fetchAlerts, 10000)
    return () => clearInterval(timer)
  }, [])

  const handleStatusChange = async (alertId: string, newStatus: 'confirmed' | 'dismissed', e: React.MouseEvent) => {
    e.stopPropagation()
    try {
      await fetch(`${CONTROLLER_URL}/alerts/${alertId}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: newStatus, feedback: `User marked as ${newStatus}` }),
      })
      setAlerts(prev => prev.map(a => a.alert_id === alertId ? { ...a, status: newStatus } : a))
    } catch (err) {
      console.error('Feedback error:', err)
    }
  }

  const filteredAlerts = alerts.filter(a => {
    if (filter === 'pending') return a.status === 'pending'
    if (filter === 'critical') return a.severity === 'critical' || a.severity === 'high'
    return true
  })

  const getSeverityBadge = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'critical':
        return 'bg-red-500/15 text-red-400 border-red-500/30'
      case 'high':
        return 'bg-amber-500/15 text-amber-400 border-amber-500/30'
      case 'medium':
        return 'bg-yellow-500/15 text-yellow-400 border-yellow-500/30'
      default:
        return 'bg-blue-500/15 text-blue-400 border-blue-500/30'
    }
  }

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/20 z-[10000]"
          />

          {/* Slide-out Drawer — translucent see-through blur matching navbar & chatbox */}
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 28, stiffness: 220 }}
            style={{
              background: 'rgba(31, 31, 31, 0.34)',
              backdropFilter: 'blur(12px)',
              WebkitBackdropFilter: 'blur(12px)',
              borderLeft: '1px solid #333333',
              boxShadow: '-8px 0 32px rgba(0, 0, 0, 0.3), inset 1px 0 0 rgba(255, 255, 255, 0.05)',
            }}
            className="fixed top-0 right-0 h-full w-full sm:w-[680px] z-[10001] flex flex-col text-slate-100 font-sans"
          >
            {/* Header */}
            <div
              style={{
                background: 'rgba(31, 31, 31, 0.25)',
                borderBottom: '1px solid #333333',
                padding: '24px 32px',
              }}
              className="flex items-center justify-between shrink-0"
            >
              <div className="flex items-center gap-4">
                <div
                  style={{
                    background: 'rgba(255, 255, 255, 0.06)',
                    border: '1px solid #333333',
                    padding: '12px',
                    borderRadius: '16px',
                  }}
                  className="text-slate-200"
                >
                  <Bell className="w-6 h-6" />
                </div>
                <div>
                  <h2 className="font-bold text-xl tracking-tight text-white">
                    Spatial Intelligence
                  </h2>
                  <p className="text-sm text-slate-300 mt-1 font-medium">
                    Persistent Geospatial Memory Engine
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <button
                  onClick={fetchAlerts}
                  disabled={loading}
                  style={{
                    padding: '10px 14px',
                    borderRadius: '14px',
                    border: '1px solid #333333',
                    background: 'rgba(31, 31, 31, 0.62)',
                    color: '#d1d5db',
                  }}
                  className="hover:border-slate-500 hover:text-white transition-all cursor-pointer"
                  title="Refresh Alerts"
                >
                  <RotateCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
                </button>
                <button
                  onClick={onClose}
                  style={{
                    padding: '10px 14px',
                    borderRadius: '14px',
                    border: '1px solid #333333',
                    background: 'rgba(31, 31, 31, 0.62)',
                    color: '#d1d5db',
                  }}
                  className="hover:border-slate-500 hover:text-white transition-all cursor-pointer"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            </div>

            {/* Filter Bar */}
            <div
              style={{
                background: 'rgba(31, 31, 31, 0.18)',
                borderBottom: '1px solid #333333',
                padding: '16px 32px',
              }}
              className="flex items-center justify-between gap-4 shrink-0"
            >
              <div className="flex items-center gap-2.5 text-slate-300 text-sm font-medium">
                <Filter className="w-4 h-4 text-slate-400" />
                <span>Filter:</span>
              </div>
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setFilter('pending')}
                  style={{
                    padding: '8px 18px',
                    fontSize: 13,
                    borderRadius: 9999,
                    border: filter === 'pending' ? '1px solid rgba(255,255,255,0.4)' : '1px solid #333',
                    background: filter === 'pending' ? 'rgba(255,255,255,0.14)' : 'rgba(31,31,31,0.4)',
                    color: filter === 'pending' ? '#ffffff' : '#9ca3af',
                  }}
                  className="font-medium transition-all cursor-pointer"
                >
                  Pending ({alerts.filter(a => a.status === 'pending').length})
                </button>
                <button
                  onClick={() => setFilter('critical')}
                  style={{
                    padding: '8px 18px',
                    fontSize: 13,
                    borderRadius: 9999,
                    border: filter === 'critical' ? '1px solid rgba(239,68,68,0.5)' : '1px solid #333',
                    background: filter === 'critical' ? 'rgba(239,68,68,0.18)' : 'rgba(31,31,31,0.4)',
                    color: filter === 'critical' ? '#fca5a5' : '#9ca3af',
                  }}
                  className="font-medium transition-all cursor-pointer"
                >
                  High / Critical
                </button>
                <button
                  onClick={() => setFilter('all')}
                  style={{
                    padding: '8px 18px',
                    fontSize: 13,
                    borderRadius: 9999,
                    border: filter === 'all' ? '1px solid rgba(255,255,255,0.4)' : '1px solid #333',
                    background: filter === 'all' ? 'rgba(255,255,255,0.14)' : 'rgba(31,31,31,0.4)',
                    color: filter === 'all' ? '#ffffff' : '#9ca3af',
                  }}
                  className="font-medium transition-all cursor-pointer"
                >
                  All ({alerts.length})
                </button>
              </div>
            </div>

            {/* Alerts Scrollable Area */}
            <div className="flex-1 overflow-y-auto p-8 space-y-9">
              {error && (
                <div
                  style={{
                    background: 'rgba(239, 68, 68, 0.12)',
                    border: '1px solid rgba(239, 68, 68, 0.3)',
                    borderRadius: '16px',
                    padding: '18px 24px',
                  }}
                  className="text-sm text-red-300 flex items-center gap-4"
                >
                  <AlertTriangle className="w-6 h-6 text-red-400 shrink-0" />
                  <span>{error}</span>
                </div>
              )}

              {filteredAlerts.length === 0 && !loading && (
                <div className="py-28 text-center space-y-5">
                  <div
                    style={{
                      background: 'rgba(31, 31, 31, 0.4)',
                      border: '1px solid #333333',
                      borderRadius: '20px',
                    }}
                    className="w-18 h-18 mx-auto flex items-center justify-center text-slate-300"
                  >
                    <ShieldAlert className="w-9 h-9 stroke-[1.5]" />
                  </div>
                  <div className="space-y-2">
                    <p className="text-lg font-semibold text-white">No spatial alerts found</p>
                    <p className="text-sm text-slate-300 max-w-sm mx-auto leading-relaxed">
                      PGIL spatial memory engine runs background temporal checks automatically when new imagery is ingested.
                    </p>
                  </div>
                </div>
              )}

              {filteredAlerts.map(alert => (
                <div
                  key={alert.alert_id}
                  onClick={() => onSelectAlert(alert.alert_id, alert.change_event_id)}
                  style={{
                    background: 'rgba(31, 31, 31, 0.28)',
                    backdropFilter: 'blur(12px)',
                    WebkitBackdropFilter: 'blur(12px)',
                    border: '1px solid #333333',
                    borderRadius: '22px',
                    padding: '28px',
                    boxShadow: '0 8px 32px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.04)',
                  }}
                  className={`group transition-all cursor-pointer hover:border-slate-400 hover:bg-[#1f1f1f50] ${
                    alert.status === 'pending' ? 'opacity-100' : 'opacity-65 hover:opacity-100'
                  }`}
                >
                  <div className="flex items-center justify-between gap-4 mb-5">
                    <span
                      className={`px-5 py-2 rounded-xl text-xs uppercase font-bold tracking-wider border ${getSeverityBadge(
                        alert.severity
                      )}`}
                    >
                      {alert.severity}
                    </span>
                    <span className="text-sm text-slate-300 font-mono">
                      Conf: <strong className="text-white text-sm">{Math.round(alert.confidence * 100)}%</strong>
                    </span>
                  </div>

                  <h3 className="font-semibold text-lg text-white group-hover:text-emerald-300 transition-colors flex items-center justify-between gap-3 leading-snug">
                    <span>{alert.title}</span>
                    <ArrowRight className="w-5 h-5 text-slate-400 group-hover:text-white group-hover:translate-x-1 transition-all shrink-0" />
                  </h3>

                  <p className="text-sm text-slate-200 mt-3.5 leading-relaxed font-normal">
                    {alert.description}
                  </p>

                  {/* Before/After Image Thumbnails */}
                  {(alert.old_image_url || alert.new_image_url) && (
                    <div className="mt-4 grid grid-cols-2 gap-3">
                      {alert.old_image_url && (
                        <div className="relative">
                          <div
                            style={{
                              background: 'rgba(0,0,0,0.4)',
                              borderRadius: '12px',
                              overflow: 'hidden',
                              height: 90,
                            }}
                            className="flex items-center justify-center"
                          >
                            <img
                              src={alert.old_image_url}
                              onError={(e: any) => {
                                e.target.style.display = 'none'
                              }}
                              alt="Previous observation"
                              className="w-full h-full object-cover"
                            />
                          </div>
                          <span
                            style={{
                              background: 'rgba(0,0,0,0.7)',
                              fontSize: 10,
                              padding: '2px 8px',
                              borderRadius: 6,
                            }}
                            className="absolute bottom-1.5 left-1.5 text-slate-300 font-mono"
                          >
                            Before
                          </span>
                        </div>
                      )}
                      {alert.new_image_url && (
                        <div className="relative">
                          <div
                            style={{
                              background: 'rgba(0,0,0,0.4)',
                              borderRadius: '12px',
                              overflow: 'hidden',
                              height: 90,
                            }}
                            className="flex items-center justify-center"
                          >
                            <img
                              src={alert.new_image_url}
                              onError={(e: any) => {
                                e.target.style.display = 'none'
                              }}
                              alt="Latest observation"
                              className="w-full h-full object-cover"
                            />
                          </div>
                          <span
                            style={{
                              background: 'rgba(0,0,0,0.7)',
                              fontSize: 10,
                              padding: '2px 8px',
                              borderRadius: 6,
                            }}
                            className="absolute bottom-1.5 left-1.5 text-slate-300 font-mono"
                          >
                            After
                          </span>
                        </div>
                      )}
                    </div>
                  )}

                  {alert.observation_interval && (
                    <div
                      style={{
                        background: 'rgba(15, 17, 16, 0.6)',
                        border: '1px solid #333333',
                        borderRadius: '12px',
                        padding: '12px 16px',
                      }}
                      className="mt-5 text-xs text-slate-300 font-mono flex items-center justify-between"
                    >
                      <span>Interval:</span>
                      <span className="text-white font-medium">{alert.observation_interval}</span>
                    </div>
                  )}

                  {/* Actions & Footer */}
                  <div
                    style={{ borderTop: '1px solid #333333' }}
                    className="mt-6 pt-5 flex items-center justify-between text-xs"
                  >
                    <span className="text-xs text-slate-400 font-mono">
                      Area: <span className="text-slate-200">{alert.area_id.slice(0, 12)}...</span>
                    </span>

                    <div className="flex items-center gap-3">
                      {alert.status === 'pending' ? (
                        <>
                          <button
                            onClick={e => handleStatusChange(alert.alert_id, 'confirmed', e)}
                            style={{
                              padding: '10px 20px',
                              fontSize: 13,
                              borderRadius: 9999,
                              border: '1px solid rgba(255, 255, 255, 0.3)',
                              background: 'rgba(255, 255, 255, 0.12)',
                              color: '#ffffff',
                            }}
                            className="font-medium hover:bg-white/20 transition-all cursor-pointer"
                          >
                            Confirm Threat
                          </button>
                          <button
                            onClick={e => handleStatusChange(alert.alert_id, 'dismissed', e)}
                            style={{
                              padding: '10px 20px',
                              fontSize: 13,
                              borderRadius: 9999,
                              border: '1px solid #333333',
                              background: 'rgba(31, 31, 31, 0.6)',
                              color: '#9ca3af',
                            }}
                            className="font-medium hover:border-slate-500 hover:text-white transition-all cursor-pointer"
                          >
                            Dismiss
                          </button>
                        </>
                      ) : (
                        <span className="text-xs text-slate-400 italic capitalize font-medium">
                          {alert.status}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}
