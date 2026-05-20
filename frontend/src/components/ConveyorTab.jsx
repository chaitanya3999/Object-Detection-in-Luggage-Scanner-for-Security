import React from 'react';
import { Play, Pause, AlertTriangle, ShieldCheck } from 'lucide-react';

export default function ConveyorTab({
  isPlaying, setIsPlaying, luggageQueue, selectedBag, scanResult, scanLoading, selectedBoxId, setSelectedBoxId, handleSelectBag
}) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '250px 1fr', gap: '1.5rem', height: '100%' }}>
      {/* Simulation Control Panel */}
      <div className="glass-panel" style={{ padding: '1rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <h3 style={{ fontSize: '0.85rem', fontWeight: '800', textTransform: 'uppercase', color: 'var(--text-secondary)', letterSpacing: '0.5px' }}>Conveyor Control</h3>
        <button 
          className={isPlaying ? "btn-danger" : "btn-primary"}
          style={{ width: '100%', display: 'flex', justifyContent: 'center' }}
          onClick={() => setIsPlaying(!isPlaying)}
        >
          {isPlaying ? <><Pause size={16} /> Halt Belt</> : <><Play size={16} /> Start Belt</>}
        </button>
        
        <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.5rem' }}>
          {luggageQueue.map((bag) => (
            <div 
              key={bag.id}
              onClick={() => handleSelectBag(bag)}
              style={{
                padding: '0.75rem',
                borderRadius: '8px',
                background: selectedBag?.id === bag.id ? 'rgba(56, 189, 248, 0.1)' : 'rgba(255,255,255,0.02)',
                border: selectedBag?.id === bag.id ? '1px solid var(--accent-cyan)' : '1px solid rgba(255,255,255,0.05)',
                cursor: 'pointer',
                transition: 'all 0.2s ease'
              }}
            >
              <div style={{ fontSize: '0.8rem', fontWeight: '700', color: 'var(--text-main)', marginBottom: '0.2rem' }}>{bag.name}</div>
              <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>Expected: <span style={{ color: bag.expected_level === 'CRITICAL' ? 'var(--color-critical)' : (bag.expected_level === 'WARNING' ? 'var(--color-warning)' : 'var(--color-safe)') }}>{bag.expected_level}</span></div>
            </div>
          ))}
        </div>
      </div>

      {/* Main Inspection View */}
      <div className="glass-panel" style={{ padding: '0', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div style={{ padding: '1.25rem', borderBottom: '1px solid rgba(255,255,255,0.05)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h2 style={{ fontSize: '1.1rem', fontWeight: '800' }}>Live Scanner Feed</h2>
          {scanResult && (
            <div className="mat-chip" style={{ 
              background: scanResult.overall.level === 'CRITICAL' ? 'var(--color-critical-glow)' : 'var(--color-safe-glow)', 
              color: scanResult.overall.level === 'CRITICAL' ? 'var(--color-critical)' : 'var(--color-safe)',
              border: '1px solid currentColor'
            }}>
              {scanResult.overall.level === 'CRITICAL' ? <AlertTriangle size={14} /> : <ShieldCheck size={14} />}
              Verdict: {scanResult.overall.level}
            </div>
          )}
        </div>
        
        <div style={{ flex: 1, padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.5rem', overflowY: 'auto' }}>
          {scanLoading ? (
            <div style={{ flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'center', color: 'var(--accent-cyan)' }}>
              Scanning...
            </div>
          ) : scanResult ? (
            <>
              {/* Dual Image Display */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', height: '350px' }}>
                <div style={{ background: '#0e111a', borderRadius: '8px', padding: '0.5rem', border: '1px solid rgba(255,255,255,0.05)', display: 'flex', flexDirection: 'column' }}>
                  <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', marginBottom: '0.5rem', textAlign: 'center' }}>Raw Dual-Energy X-Ray</span>
                  <div style={{ flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'center', overflow: 'hidden' }}>
                    <img src={scanResult.original_image} alt="Raw" style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }} />
                  </div>
                </div>
                <div style={{ background: '#0e111a', borderRadius: '8px', padding: '0.5rem', border: '1px solid rgba(255,255,255,0.05)', display: 'flex', flexDirection: 'column' }}>
                  <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', marginBottom: '0.5rem', textAlign: 'center' }}>AI Detected Properties</span>
                  <div style={{ flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'center', overflow: 'hidden' }}>
                    <img src={scanResult.annotated_image} alt="Annotated" style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }} />
                  </div>
                </div>
              </div>

              {/* Detected Objects List */}
              {scanResult.bboxes.length > 0 && (
                <div>
                  <h3 style={{ fontSize: '0.85rem', fontWeight: '700', marginBottom: '0.75rem', color: 'var(--text-secondary)' }}>Detected Elements ({scanResult.bboxes.length})</h3>
                  <div style={{ display: 'flex', gap: '0.75rem', overflowX: 'auto', paddingBottom: '0.5rem' }}>
                    {scanResult.bboxes.map((box, idx) => (
                      <button 
                        key={idx}
                        className={`mat-chip ${selectedBoxId === idx ? 'active' : ''}`}
                        onClick={() => setSelectedBoxId(idx)}
                        style={{
                          background: selectedBoxId === idx ? 'rgba(255,255,255,0.1)' : 'rgba(255,255,255,0.02)',
                          borderColor: box.threat_level === 'CRITICAL' ? 'var(--color-critical)' : 'rgba(255,255,255,0.1)'
                        }}
                      >
                        Item #{idx + 1} - {box.material}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Inspector Panel */}
              {selectedBoxId !== null && scanResult.bboxes[selectedBoxId] && (
                <div style={{ background: 'rgba(255,255,255,0.02)', borderRadius: '8px', padding: '1.25rem', border: '1px solid rgba(255,255,255,0.05)' }}>
                  <h4 style={{ fontSize: '0.9rem', fontWeight: '800', marginBottom: '1rem', color: 'var(--accent-cyan)' }}>Deep Properties Extracted (Model 1)</h4>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: '1rem' }}>
                    {Object.entries(scanResult.bboxes[selectedBoxId].properties || {}).map(([key, value]) => {
                      const formattedKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                      const formattedVal = typeof value === 'number' ? value.toFixed(3) : value;
                      return (
                        <div key={key} style={{ display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
                          <span style={{ fontSize: '0.65rem', color: 'var(--text-secondary)' }}>{formattedKey}</span>
                          <span style={{ fontSize: '0.85rem', fontWeight: '600', color: 'var(--text-main)' }}>{formattedVal}</span>
                        </div>
                      );
                    })}
                  </div>
                  <div style={{ marginTop: '1.5rem', paddingTop: '1rem', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                    <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '0.25rem' }}>Model 2 Classification Reason:</span>
                    <p style={{ fontSize: '0.85rem', color: 'var(--text-main)', lineHeight: '1.4' }}>{scanResult.bboxes[selectedBoxId].explanation}</p>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div style={{ flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'center', color: 'var(--text-muted)' }}>
              Select a bag to view scan results.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
