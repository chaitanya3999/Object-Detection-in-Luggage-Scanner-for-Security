import React from 'react';
import { 
  ShieldAlert, ShieldCheck, Activity, Workflow as BeltIcon, 
  Upload, Layers, BarChart3, Settings, HelpCircle, User, Info, CheckCircle2 
} from 'lucide-react';

export default function Sidebar({ activeTab, setActiveTab, modelStatus }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <ShieldAlert size={28} color="var(--accent-cyan)" />
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          <h1 style={{ fontSize: '1.1rem', fontWeight: '800', letterSpacing: '0.5px' }}>X-RAY <span style={{ color: 'var(--accent-cyan)' }}>SENTRY</span></h1>
          <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px' }}>Dual-Model Edition</span>
        </div>
      </div>

      <nav className="sidebar-nav">
        <div className="nav-group">
          <span className="nav-group-title">Inspection</span>
          <button className={`nav-btn ${activeTab === 'conveyor' ? 'active' : ''}`} onClick={() => setActiveTab('conveyor')}>
            <BeltIcon size={18} /> Conveyor Feed
          </button>
          <button className={`nav-btn ${activeTab === 'manual' ? 'active' : ''}`} onClick={() => setActiveTab('manual')}>
            <Upload size={18} /> Deep Property Analyzer
          </button>
        </div>

        <div className="nav-group">
          <span className="nav-group-title">Simulation</span>
          <button className={`nav-btn ${activeTab === 'tip' ? 'active' : ''}`} onClick={() => setActiveTab('tip')}>
            <Layers size={18} /> TIP Sandbox
          </button>
        </div>

        <div className="nav-group">
          <span className="nav-group-title">System</span>
          <button className={`nav-btn ${activeTab === 'analytics' ? 'active' : ''}`} onClick={() => setActiveTab('analytics')}>
            <BarChart3 size={18} /> Model Analytics
          </button>
        </div>
      </nav>

      {modelStatus && (
        <div className="system-status">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.7rem', textTransform: 'uppercase', color: 'var(--text-secondary)', fontWeight: '700' }}>Engine Status</span>
            {modelStatus.checkpoint_found ? <CheckCircle2 size={14} color="var(--color-safe)" /> : <Info size={14} color="var(--color-warning)" />}
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.2rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            <span style={{ display: 'flex', justifyContent: 'space-between' }}>
              Mode: <span style={{ color: modelStatus.is_dl_mode ? 'var(--accent-cyan)' : 'var(--text-secondary)' }}>
                {modelStatus.is_dl_mode ? 'DEEP LEARNING' : 'CV FALLBACK'}
              </span>
            </span>
            <span style={{ display: 'flex', justifyContent: 'space-between' }}>
              Model 2: <span style={{ color: modelStatus.model2_found ? 'var(--accent-cyan)' : 'var(--text-secondary)' }}>
                {modelStatus.model2_found ? 'LOADED (RF)' : 'HEURISTICS'}
              </span>
            </span>
            <span style={{ display: 'flex', justifyContent: 'space-between' }}>
              Device: <span>{modelStatus.device.toUpperCase()}</span>
            </span>
          </div>
        </div>
      )}

      <div className="sidebar-footer">
        <button className="icon-btn"><Settings size={18} /></button>
        <button className="icon-btn"><HelpCircle size={18} /></button>
        <button className="icon-btn"><User size={18} /></button>
      </div>
    </aside>
  );
}
