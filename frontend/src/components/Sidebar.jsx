import React from 'react';
import { 
  ShieldAlert, Workflow as BeltIcon, Upload, Layers, 
  BarChart3, Settings, HelpCircle, User, CheckCircle2, Info 
} from 'lucide-react';

export default function Sidebar({ activeTab, setActiveTab, modelStatus, scanResult }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <ShieldAlert size={28} color="var(--accent-primary)" />
        <div className="sidebar-brand-text">
          <h1 className="sidebar-brand-title">X-RAY <span style={{ color: 'var(--accent-primary)' }}>SENTRY</span></h1>
          <span className="sidebar-brand-subtitle">Dual-Model Edition</span>
        </div>
      </div>

      <nav className="sidebar-nav">
        <div className="nav-group">
          <span className="nav-group-title">Inspection</span>
          <button className={`nav-btn ${activeTab === 'conveyor' ? 'active' : ''}`} onClick={() => setActiveTab('conveyor')}>
            <BeltIcon size={18} /> <span>Conveyor Feed</span>
          </button>
          <button className={`nav-btn ${activeTab === 'manual' ? 'active' : ''}`} onClick={() => setActiveTab('manual')}>
            <Upload size={18} /> <span>Deep Property Analyzer</span>
          </button>
        </div>

        <div className="nav-group">
          <span className="nav-group-title">Simulation</span>
          <button className={`nav-btn ${activeTab === 'tip' ? 'active' : ''}`} onClick={() => setActiveTab('tip')}>
            <Layers size={18} /> <span>TIP Sandbox</span>
          </button>
        </div>

        <div className="nav-group">
          <span className="nav-group-title">System</span>
          <button className={`nav-btn ${activeTab === 'analytics' ? 'active' : ''}`} onClick={() => setActiveTab('analytics')}>
            <BarChart3 size={18} /> <span>Model Analytics</span>
          </button>
        </div>
      </nav>

      <div className="system-status threat-composition-widget" style={{ marginTop: 'auto', marginBottom: '1rem', border: '1px solid rgba(255,255,255,0.08)' }}>
        <div className="system-status-title" style={{ marginBottom: '0.8rem' }}>
          <span>Threat Composition</span>
          <CheckCircle2 size={14} color="var(--color-safe)" />
        </div>
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
          {[
            { label: 'Heavy Metal', key: 'Heavy/Dense Metal', color: '#ef4444' },
            { label: 'Light Metal', key: 'Light Metal', color: '#3b82f6' },
            { label: 'Fabric/Plastic', key: 'Fabric/Plastic', color: '#22c55e' },
            { label: 'Organic', key: 'Organic', color: '#f59e0b' }
          ].map(mat => {
            const valStr = scanResult?.diagnostics?.composition_breakdown?.[mat.key] || '0%';
            const pct = parseFloat(valStr) || 0;
            return (
              <div key={mat.label} className="composition-row" style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div className="comp-label" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <div className="comp-dot" style={{ width: '8px', height: '8px', borderRadius: '50%', background: mat.color }}></div>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>{mat.label}</span>
                  </div>
                  <span className="comp-value" style={{ fontSize: '0.75rem', fontWeight: 700, color: mat.color }}>{valStr}</span>
                </div>
                <div style={{ height: '4px', background: 'rgba(255,255,255,0.08)', borderRadius: '2px', overflow: 'hidden' }}>
                  <div style={{ width: `${pct}%`, height: '100%', background: mat.color, transition: 'width 0.5s ease' }}></div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="sidebar-footer">
        <button className="icon-btn"><Settings size={18} /></button>
        <button className="icon-btn"><HelpCircle size={18} /></button>
        <button className="icon-btn"><User size={18} /></button>
      </div>
    </aside>
  );
}
