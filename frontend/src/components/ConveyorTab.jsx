import React from 'react';
import { Play, Pause, ShieldCheck, AlertTriangle, RefreshCw } from 'lucide-react';

export default function ConveyorTab({
  isPlaying, setIsPlaying, luggageQueue, selectedBag,
  scanResult, scanLoading, selectedBoxId, setSelectedBoxId, handleSelectBag
}) {
  const getThreatColor = (level) => {
    if (level === 'CRITICAL') return 'var(--color-critical)';
    if (level === 'WARNING') return 'var(--color-warning)';
    return 'var(--color-safe)';
  };

  const getMaterialColor = (mat) => {
    if (mat === 'organic') return 'var(--color-organic)';
    if (mat === 'metallic') return 'var(--color-metallic)';
    if (mat === 'mixed') return 'var(--color-mixed)';
    return 'var(--color-opaque)';
  };

  return (
    <div className="animate-fadeIn" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', height: '100%' }}>
      
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 className="section-title">
          <ShieldCheck size={22} color="var(--accent-primary)" />
          Operator Live Feed
        </h2>
        <button onClick={() => setIsPlaying(!isPlaying)} className="btn-primary">
          {isPlaying ? <><Pause size={16} /> Pause Conveyor</> : <><Play size={16} /> Start Conveyor</>}
        </button>
      </div>

      {/* Conveyor Strip */}
      <div className={`bag-queue-belt ${isPlaying ? 'conveyor-animation' : ''}`}>
        {luggageQueue.map((bag) => (
          <div 
            key={bag.id}
            className={`queue-item ${bag.id === selectedBag?.id ? 'active' : ''} ${bag.result?.overall?.level || ''}`}
            onClick={() => handleSelectBag(bag)}
          >
            <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              Bag #{bag.id.substring(0, 5)}
            </div>
            {bag.result ? (
              <div style={{ fontSize: '0.65rem', color: getThreatColor(bag.result.overall.level), fontWeight: 700 }}>
                {bag.result.overall.level}
              </div>
            ) : (
              <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>Pending...</div>
            )}
            {bag.result && bag.result.bboxes.length > 0 && (
              <div className={`mat-chip ${bag.result.bboxes[0].material}`}>
                {bag.result.bboxes[0].material}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Selected Bag Detail */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 350px', gap: '1.25rem', flex: 1, minHeight: 0 }}>
        
        {/* Left: Image Viewer */}
        <div className="glass-panel" style={{ padding: '1rem', display: 'flex', flexDirection: 'column', position: 'relative' }}>
          {scanLoading ? (
            <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.5)', zIndex: 10 }}>
              <RefreshCw size={32} className="animate-spin" color="var(--accent-primary)" />
            </div>
          ) : null}
          
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.3)', borderRadius: 'var(--radius-sm)', overflow: 'hidden', position: 'relative' }}>
            {scanResult ? (
              <img src={scanResult.annotated_image} alt="Scan Result" style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }} />
            ) : (
              <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Select a bag to view analysis</div>
            )}
          </div>
          
          {scanResult && scanResult.bboxes.length > 0 && (
            <div style={{ display: 'flex', gap: '0.5rem', marginTop: '1rem', flexWrap: 'wrap' }}>
              {scanResult.bboxes.map((box, i) => (
                <button
                  key={i}
                  onClick={() => setSelectedBoxId(i)}
                  className={selectedBoxId === i ? 'btn-primary' : 'btn-ghost'}
                  style={{
                    fontSize: '0.72rem', padding: '0.3rem 0.6rem',
                    border: selectedBoxId === i ? 'none' : `1px solid ${getThreatColor(box.threat_level)}33`,
                    color: selectedBoxId === i ? 'white' : getThreatColor(box.threat_level)
                  }}
                >
                  #{i + 1} {box.material}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Right: Threat Info */}
        <div className="glass-panel" style={{ padding: '1.25rem', overflowY: 'auto' }}>
          <h3 className="section-subtitle">Bag Diagnostics</h3>
          {scanResult ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div className="glass-inset" style={{ padding: '1rem', borderLeft: `4px solid ${getThreatColor(scanResult.overall.level)}` }}>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>Overall Verdict</div>
                <div style={{ fontSize: '1.1rem', fontWeight: 800, color: getThreatColor(scanResult.overall.level) }}>{scanResult.overall.level}</div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-primary)', marginTop: '0.5rem' }}>{scanResult.overall.explanation}</div>
              </div>
              
              {scanResult.bboxes[selectedBoxId] && (
                <>
                  <h4 className="section-subtitle" style={{ marginTop: '0.5rem' }}>Selected Object Properties</h4>
                  <div className="props-grid">
                    {Object.entries(scanResult.properties?.[selectedBoxId] || {}).slice(0, 8).map(([key, value]) => {
                      const norm = Math.min((value / (key === 'length_to_width_ratio' ? 10 : 1)) * 100, 100);
                      return (
                        <div key={key} className="prop-bar-container">
                          <div className="prop-label-row">
                            <span className="label">{key.replace(/_/g, ' ')}</span>
                            <span className="value">{value.toFixed(2)}</span>
                          </div>
                          <div className="prop-bar-outer">
                            <div className="prop-bar-inner" style={{ width: `${norm}%` }} />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </>
              )}
            </div>
          ) : (
            <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Waiting for data...</div>
          )}
        </div>

      </div>
    </div>
  );
}
