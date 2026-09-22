/**
 * ChatSidebar — ChatGPT-style collapsible sidebar
 *
 * Two states:
 *   Expanded  (~260 px): logo + name, nav items, recents list, user footer
 *   Collapsed (~56 px):  icon rail, centered icons, avatar footer
 *
 * Toggle: clicking the logo icon expands / collapses (no chevron button).
 * State is persisted in localStorage so it survives re-renders.
 */

import React, { useState, useEffect, useCallback, useRef } from 'react'
import {
  Plus,
  Search,
  BookOpen,
  MoreHorizontal,
  MessageSquare,
  SquarePen,
  LogOut,
  LogIn,
} from 'lucide-react'

/* ─────────────────────────────────────────────────────────
   Public Types
───────────────────────────────────────────────────────── */
export interface SidebarNavItem {
  id: string
  label: string
  icon: React.ReactNode
  onClick?: () => void
}

export interface SidebarRecentChat {
  id: string
  title: string
}

export interface SidebarUserInfo {
  name: string
  initials: string
  subtitle?: string
  avatarUrl?: string
  email?: string
}

export interface ChatSidebarProps {
  /** Primary nav rows (icon + label). Defaults to New chat / Library / More */
  navItems?: SidebarNavItem[]
  /** Flat list of recent chats, newest first */
  recentChats?: SidebarRecentChat[]
  /** Highlighted chat id */
  activeConversationId?: string
  /** Footer user info */
  userInfo?: SidebarUserInfo | null
  /** Fires when user clicks "New chat" (nav item or header icon) */
  onNewChat?: () => void
  /** Fires when user clicks a recent chat row */
  onSelectConversation?: (id: string) => void
  /** Fires when user clicks the search icon */
  onSearch?: () => void
  /** Fires when user clicks sign out */
  onSignOut?: () => void
  /** Fires when user clicks sign in */
  onSignIn?: () => void
}

/* ─────────────────────────────────────────────────────────
   Constants
───────────────────────────────────────────────────────── */
const EXPANDED_W = 260
const COLLAPSED_W = 56
const ICON_SIZE = 20
const LS_KEY = 'satquery_sidebar_expanded'
const EASE = 'cubic-bezier(0.16, 1, 0.3, 1)'
const DURATION = '230ms'

/* ─────────────────────────────────────────────────────────
   Defaults
───────────────────────────────────────────────────────── */
const DEFAULT_NAV: SidebarNavItem[] = [
  { id: 'new-chat', label: 'New chat', icon: <Plus size={ICON_SIZE} /> },
  { id: 'library', label: 'Library', icon: <BookOpen size={ICON_SIZE} /> },
  { id: 'more', label: 'More', icon: <MoreHorizontal size={ICON_SIZE} /> },
]

const DEFAULT_USER: SidebarUserInfo = {
  name: 'Analyst',
  initials: 'AN',
  subtitle: 'Pro Workspace',
}

/* ─────────────────────────────────────────────────────────
   Tooltip — appears to the right of an icon when collapsed
───────────────────────────────────────────────────────── */
const Tooltip: React.FC<{
  text: string
  enabled: boolean
  children: React.ReactNode
}> = ({ text, enabled, children }) => {
  const [show, setShow] = useState(false)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)

  const enter = () => {
    if (!enabled) return
    timer.current = setTimeout(() => setShow(true), 350)
  }
  const leave = () => {
    clearTimeout(timer.current)
    setShow(false)
  }

  return (
    <div onMouseEnter={enter} onMouseLeave={leave} style={{ position: 'relative' }}>
      {children}
      {show && (
        <div
          style={{
            position: 'absolute',
            left: '100%',
            top: '50%',
            transform: 'translateY(-50%)',
            marginLeft: 10,
            padding: '5px 12px',
            borderRadius: 7,
            background: 'rgba(28, 28, 30, 0.97)',
            border: '1px solid rgba(255,255,255,0.10)',
            color: 'rgba(255,255,255,0.88)',
            fontSize: 12,
            fontWeight: 500,
            fontFamily: 'Inter, system-ui, sans-serif',
            whiteSpace: 'nowrap',
            pointerEvents: 'none',
            zIndex: 999,
            boxShadow: '0 4px 20px rgba(0,0,0,0.45)',
          }}
        >
          {text}
        </div>
      )}
    </div>
  )
}

/* ─────────────────────────────────────────────────────────
   Header icon button (search / compose)
───────────────────────────────────────────────────────── */
const HeaderIconBtn: React.FC<{
  icon: React.ReactNode
  title: string
  onClick?: () => void
}> = ({ icon, title, onClick }) => {
  const [h, setH] = useState(false)
  return (
    <button
      onClick={onClick}
      aria-label={title}
      onMouseEnter={() => setH(true)}
      onMouseLeave={() => setH(false)}
      style={{
        width: 32,
        height: 32,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: 8,
        border: 'none',
        background: h ? 'rgba(255,255,255,0.08)' : 'transparent',
        color: h ? 'rgba(255,255,255,0.85)' : 'rgba(255,255,255,0.45)',
        cursor: 'pointer',
        transition: 'background 150ms, color 150ms',
        flexShrink: 0,
      }}
    >
      {icon}
    </button>
  )
}

/* ─────────────────────────────────────────────────────────
   NavRow — one row in the primary nav list
   Expanded: icon + label, left-aligned
   Collapsed: icon only, centered
───────────────────────────────────────────────────────── */
const NavRow: React.FC<{
  item: SidebarNavItem
  expanded: boolean
  onClick?: () => void
}> = ({ item, expanded, onClick }) => {
  const [h, setH] = useState(false)

  return (
    <Tooltip text={item.label} enabled={!expanded}>
      <button
        onClick={onClick ?? item.onClick}
        aria-label={item.label}
        onMouseEnter={() => setH(true)}
        onMouseLeave={() => setH(false)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          width: '100%',
          padding: expanded ? '10px 12px' : '10px 0',
          justifyContent: expanded ? 'flex-start' : 'center',
          borderRadius: 8,
          border: 'none',
          background: h ? 'rgba(255,255,255,0.07)' : 'transparent',
          color: h ? 'rgba(255,255,255,0.92)' : 'rgba(255,255,255,0.6)',
          cursor: 'pointer',
          transition: `background 150ms, color 150ms, padding ${DURATION} ${EASE}`,
          fontFamily: 'Inter, system-ui, sans-serif',
          fontSize: 14,
          fontWeight: 500,
          flexShrink: 0,
        }}
      >
        <span style={{ flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', width: ICON_SIZE, height: ICON_SIZE }}>
          {item.icon}
        </span>
        <span
          style={{
            overflow: 'hidden',
            whiteSpace: 'nowrap',
            opacity: expanded ? 1 : 0,
            maxWidth: expanded ? 200 : 0,
            transition: `opacity 180ms ease, max-width ${DURATION} ${EASE}`,
          }}
        >
          {item.label}
        </span>
      </button>
    </Tooltip>
  )
}

/* ─────────────────────────────────────────────────────────
   RecentRow — one chat title in the Recents list
───────────────────────────────────────────────────────── */
const RecentRow: React.FC<{
  chat: SidebarRecentChat
  isActive: boolean
  onClick: () => void
}> = ({ chat, isActive, onClick }) => {
  const [h, setH] = useState(false)

  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setH(true)}
      onMouseLeave={() => setH(false)}
      style={{
        display: 'flex',
        alignItems: 'center',
        width: '100%',
        padding: '9px 12px',
        borderRadius: 8,
        border: 'none',
        background: isActive
          ? 'rgba(255,255,255,0.10)'
          : h
            ? 'rgba(255,255,255,0.06)'
            : 'transparent',
        color: isActive ? 'rgba(255,255,255,0.95)' : 'rgba(255,255,255,0.50)',
        cursor: 'pointer',
        transition: 'background 150ms, color 150ms',
        fontFamily: 'Inter, system-ui, sans-serif',
        fontSize: 13.5,
        fontWeight: isActive ? 500 : 400,
        textAlign: 'left',
      }}
    >
      <span
        style={{
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
          minWidth: 0,
        }}
      >
        {chat.title}
      </span>
    </button>
  )
}

/* ─────────────────────────────────────────────────────────
   Logo toggle — the icon is the sidebar collapse/expand trigger
───────────────────────────────────────────────────────── */
const LogoToggle: React.FC<{
  expanded: boolean
  onToggle: () => void
}> = ({ expanded, onToggle }) => {
  const [h, setH] = useState(false)

  return (
    <Tooltip text="Toggle sidebar" enabled={!expanded}>
      <button
        onClick={onToggle}
        aria-label={expanded ? 'Collapse sidebar' : 'Expand sidebar'}
        onMouseEnter={() => setH(true)}
        onMouseLeave={() => setH(false)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          background: h ? 'rgba(255,255,255,0.06)' : 'transparent',
          border: 'none',
          cursor: 'pointer',
          padding: '6px 8px',
          borderRadius: 8,
          transition: 'background 150ms',
          flexShrink: 0,
          overflow: 'hidden',
        }}
      >
        <img
          src="/satquery-icon.png"
          alt="SatQuery AI"
          style={{
            height: 24,
            width: 24,
            objectFit: 'contain',
            flexShrink: 0,
            filter: 'drop-shadow(0 1px 3px rgba(0,0,0,0.5))',
            imageRendering: 'auto' as const,
          }}
          draggable={false}
        />
        <span
          style={{
            color: 'rgba(255,255,255,0.88)',
            fontSize: 15,
            fontWeight: 600,
            fontFamily: 'Inter, system-ui, sans-serif',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            opacity: expanded ? 1 : 0,
            maxWidth: expanded ? 160 : 0,
            transition: `opacity 180ms ease, max-width ${DURATION} ${EASE}`,
            letterSpacing: '-0.01em',
          }}
        >
          SatQuery AI
        </span>
      </button>
    </Tooltip>
  )
}

/* ─────────────────────────────────────────────────────────
   UserFooter — pinned at bottom, avatar + name + subtitle
───────────────────────────────────────────────────────── */
const UserFooter: React.FC<{
  userInfo?: SidebarUserInfo | null
  expanded: boolean
  onSignOut?: () => void
  onSignIn?: () => void
}> = ({ userInfo, expanded, onSignOut, onSignIn }) => {
  const [h, setH] = useState(false)

  if (!userInfo) {
    return (
      <Tooltip text="Sign In" enabled={!expanded}>
        <button
          onClick={onSignIn}
          onMouseEnter={() => setH(true)}
          onMouseLeave={() => setH(false)}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            width: '100%',
            padding: expanded ? '8px 10px' : '8px 0',
            justifyContent: expanded ? 'flex-start' : 'center',
            borderRadius: 8,
            border: '1px solid rgba(255,255,255,0.12)',
            background: h ? 'rgba(255,255,255,0.08)' : 'rgba(255,255,255,0.03)',
            cursor: 'pointer',
            transition: `background 150ms, border-color 150ms, padding ${DURATION} ${EASE}`,
          }}
        >
          <div
            style={{
              width: 32,
              height: 32,
              borderRadius: '50%',
              background: 'rgba(99,102,241,0.15)',
              border: '1px solid rgba(99,102,241,0.3)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0,
              color: '#a5b4fc',
            }}
          >
            <LogIn size={15} />
          </div>
          {expanded && (
            <div style={{ display: 'flex', flexDirection: 'column', textAlign: 'left', overflow: 'hidden' }}>
              <span style={{ fontSize: 13, fontWeight: 600, color: 'rgba(255,255,255,0.9)', fontFamily: 'Inter, system-ui, sans-serif' }}>
                Sign In
              </span>
              <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.38)', fontFamily: 'Inter, system-ui, sans-serif' }}>
                Sync chat history
              </span>
            </div>
          )}
        </button>
      </Tooltip>
    )
  }

  return (
    <Tooltip text={onSignOut ? `${userInfo.name} (Click to Sign Out)` : userInfo.name} enabled={!expanded}>
      <div
        onMouseEnter={() => setH(true)}
        onMouseLeave={() => setH(false)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          width: '100%',
          padding: expanded ? '8px 10px' : '8px 0',
          justifyContent: expanded ? 'flex-start' : 'center',
          borderRadius: 8,
          background: h ? 'rgba(255,255,255,0.06)' : 'transparent',
          transition: `background 150ms, padding ${DURATION} ${EASE}`,
        }}
      >
        {/* Avatar circle or image */}
        {userInfo.avatarUrl ? (
          <img
            src={userInfo.avatarUrl}
            alt={userInfo.name}
            style={{
              width: 32,
              height: 32,
              borderRadius: '50%',
              objectFit: 'cover',
              border: '1px solid rgba(255,255,255,0.18)',
              flexShrink: 0,
            }}
          />
        ) : (
          <div
            style={{
              width: 32,
              height: 32,
              borderRadius: '50%',
              background: 'linear-gradient(135deg, rgba(124,58,237,0.5), rgba(59,130,246,0.5))',
              border: '1px solid rgba(255,255,255,0.14)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0,
              color: 'rgba(255,255,255,0.88)',
              fontSize: 12,
              fontWeight: 700,
              fontFamily: 'Inter, system-ui, sans-serif',
              letterSpacing: '0.02em',
            }}
          >
            {userInfo.initials}
          </div>
        )}

        {/* Name + subtitle — fade out when collapsed */}
        <div
          style={{
            overflow: 'hidden',
            opacity: expanded ? 1 : 0,
            maxWidth: expanded ? 140 : 0,
            transition: `opacity 180ms ease, max-width ${DURATION} ${EASE}`,
            display: 'flex',
            flexDirection: 'column',
            textAlign: 'left',
            flex: 1,
            minWidth: 0,
          }}
        >
          <span
            style={{
              color: 'rgba(255,255,255,0.85)',
              fontSize: 13,
              fontWeight: 600,
              fontFamily: 'Inter, system-ui, sans-serif',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              lineHeight: 1.3,
            }}
          >
            {userInfo.name}
          </span>
          {userInfo.subtitle && (
            <span
              style={{
                color: 'rgba(255,255,255,0.32)',
                fontSize: 11,
                fontFamily: 'Inter, system-ui, sans-serif',
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                lineHeight: 1.3,
              }}
            >
              {userInfo.subtitle}
            </span>
          )}
        </div>

        {/* Sign out action button */}
        {expanded && onSignOut && (
          <button
            onClick={(e) => {
              e.stopPropagation()
              onSignOut()
            }}
            title="Sign Out"
            style={{
              background: 'none',
              border: 'none',
              padding: '6px',
              borderRadius: 6,
              cursor: 'pointer',
              color: 'rgba(255,255,255,0.35)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'color 150ms, background 150ms',
              flexShrink: 0,
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.color = '#f87171'
              e.currentTarget.style.background = 'rgba(239,68,68,0.12)'
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.color = 'rgba(255,255,255,0.35)'
              e.currentTarget.style.background = 'none'
            }}
          >
            <LogOut size={15} />
          </button>
        )}
      </div>
    </Tooltip>
  )
}

/* ─────────────────────────────────────────────────────────
   ChatSidebar — main exported component
───────────────────────────────────────────────────────── */
export const ChatSidebar: React.FC<ChatSidebarProps> = ({
  navItems = DEFAULT_NAV,
  recentChats = [],
  userInfo = DEFAULT_USER,
  activeConversationId,
  onSelectConversation,
  onNewChat,
  onSearch,
  onSignOut,
  onSignIn,
}) => {
  /* ── Persisted expanded state ── */
  const [expanded, setExpanded] = useState(() => {
    try {
      const saved = localStorage.getItem(LS_KEY)
      return saved !== null ? saved === 'true' : true
    } catch {
      return true
    }
  })

  useEffect(() => {
    try { localStorage.setItem(LS_KEY, String(expanded)) } catch { /* noop */ }
  }, [expanded])

  const toggle = useCallback(() => setExpanded(prev => !prev), [])

  return (
    <aside
      style={{
        width: expanded ? EXPANDED_W : COLLAPSED_W,
        minWidth: expanded ? EXPANDED_W : COLLAPSED_W,
        transition: `width ${DURATION} ${EASE}, min-width ${DURATION} ${EASE}`,
        background: 'rgb(14, 14, 16)',
        borderRight: '1px solid rgba(255, 255, 255, 0.07)',
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        overflow: 'hidden',
        flexShrink: 0,
        zIndex: 20,
      }}
    >
      {/* ════════════════════════════════════════════
          1. FIXED HEADER — logo toggle + action icons
         ════════════════════════════════════════════ */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          padding: expanded ? '14px 10px 8px' : '14px 0 8px',
          justifyContent: expanded ? 'space-between' : 'center',
          flexShrink: 0,
          transition: `padding ${DURATION} ${EASE}`,
        }}
      >
        <LogoToggle expanded={expanded} onToggle={toggle} />

        {/* Search + Compose — visible in expanded only */}
        <div
          style={{
            display: 'flex',
            gap: 2,
            flexShrink: 0,
            overflow: 'hidden',
            opacity: expanded ? 1 : 0,
            maxWidth: expanded ? 80 : 0,
            transition: `opacity 180ms ease, max-width ${DURATION} ${EASE}`,
            pointerEvents: expanded ? 'auto' : 'none',
          }}
        >
          <HeaderIconBtn icon={<Search size={18} />} title="Search" onClick={onSearch} />
          <HeaderIconBtn icon={<SquarePen size={18} />} title="New chat" onClick={onNewChat} />
        </div>
      </div>

      {/* ════════════════════════════════════════════
          2. SCROLLABLE MIDDLE — nav + recents
         ════════════════════════════════════════════ */}
      <div
        style={{
          flex: 1,
          minHeight: 0,
          overflowY: 'auto',
          overflowX: 'hidden',
          display: 'flex',
          flexDirection: 'column',
          padding: expanded ? '4px 8px' : '4px 4px',
          transition: `padding ${DURATION} ${EASE}`,
          scrollbarWidth: 'thin',
          scrollbarColor: 'rgba(255,255,255,0.08) transparent',
        }}
      >
        {/* ── Primary nav items ── */}
        <nav style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
          {navItems.map(item => (
            <NavRow
              key={item.id}
              item={item}
              expanded={expanded}
              onClick={() => {
                if (item.id === 'new-chat') onNewChat?.()
                else item.onClick?.()
              }}
            />
          ))}
        </nav>

        {/* ── Extra collapsed-only icons (search + chats) ── */}
        {!expanded && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 1, marginTop: 2 }}>
            <NavRow
              item={{ id: '_search', label: 'Search', icon: <Search size={ICON_SIZE} /> }}
              expanded={false}
              onClick={onSearch}
            />
            <NavRow
              item={{ id: '_chats', label: 'Chats', icon: <MessageSquare size={ICON_SIZE} /> }}
              expanded={false}
            />
          </div>
        )}

        {/* ── Recents section (expanded only) ── */}
        {expanded && recentChats.length > 0 && (
          <div style={{ marginTop: 20 }}>
            <p
              style={{
                padding: '4px 12px 8px',
                fontSize: 11,
                fontWeight: 600,
                textTransform: 'uppercase',
                letterSpacing: '0.1em',
                color: 'rgba(255,255,255,0.26)',
                fontFamily: 'Inter, system-ui, sans-serif',
                userSelect: 'none',
              }}
            >
              Recents
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              {recentChats.map(chat => (
                <RecentRow
                  key={chat.id}
                  chat={chat}
                  isActive={activeConversationId === chat.id}
                  onClick={() => onSelectConversation?.(chat.id)}
                />
              ))}
            </div>
          </div>
        )}
      </div>

      {/* ════════════════════════════════════════════
          3. FIXED FOOTER — user avatar + info
         ════════════════════════════════════════════ */}
      <div
        style={{
          flexShrink: 0,
          borderTop: '1px solid rgba(255,255,255,0.07)',
          padding: expanded ? '10px 8px 14px' : '10px 4px 14px',
          transition: `padding ${DURATION} ${EASE}`,
        }}
      >
        <UserFooter userInfo={userInfo} expanded={expanded} onSignOut={onSignOut} onSignIn={onSignIn} />
      </div>
    </aside>
  )
}

export default ChatSidebar
