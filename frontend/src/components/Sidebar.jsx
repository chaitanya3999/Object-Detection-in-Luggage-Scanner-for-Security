import React from 'react';
import { 
  ShieldAlert, Workflow as BeltIcon, Upload, Layers, 
  BarChart3, Settings, HelpCircle, User, CheckCircle2, Info 
} from 'lucide-react';

export default function Sidebar({ activeTab, setActiveTab, modelStatus }) {
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

      {modelStatus && (
        <div className="system-status">
          <div className="system-status-title">
            <span>Engine Status</span>
            {modelStatus.checkpoint_found ? <CheckCircle2 size={14} color="var(--color-safe)" /> : <Info size={14} color="var(--color-warning)" />}
          </div>
          <div className="status-row">
            <span>Mode:</span>
            <span className={`status-value ${modelStatus.is_dl_mode ? 'active' : 'fallback'}`}>
              {modelStatus.is_dl_mode ? 'DEEP LEARNING' : 'CV FALLBACK'}
            </span>
          </div>
          <div className="status-row">
            <span>Model 2:</span>
            <span className={`status-value ${modelStatus.model2_found ? 'active' : 'fallback'}`}>
              {modelStatus.model2_found ? 'LOADED (RF)' : 'HEURISTICS'}
            </span>
          </div>
          <div className="status-row">
            <span>Device:</span>
            <span className="status-value">{modelStatus.device?.toUpperCase()}</span>
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
