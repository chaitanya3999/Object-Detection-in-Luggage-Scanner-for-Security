import React from 'react';
import { Play, Pause, ShieldCheck, ShieldAlert, AlertTriangle, RefreshCw, Clock, Fingerprint, Cpu, Crosshair } from 'lucide-react';

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

  const ensureDataUri = (src) => {
    if (!src) return null;
    if (src.startsWith('data:')) return src;
    return `data:image/jpeg;base64,${src}`;
  };

  // Compute material composition breakdown from bboxes
  const getComposition = () => {
    if (!scanResult?.bboxes || scanResult.bboxes.length === 0) return [];
    const counts = {};
    scanResult.bboxes.forEach(b => {
      counts[b.material] = (counts[b.material] || 0) + 1;
    });
    const total = scanResult.bboxes.length;
    return Object.entries(counts).map(([mat, count]) => ({
      material: mat,
      percentage: Math.round((count / total) * 100),
      count
    }));
  };

  // Get the max AI confidence from all detected boxes
  const getMaxConfidence = () => {
    if (!scanResult?.bboxes || scanResult.bboxes.length === 0) return null;
    return Math.max(...scanResult.bboxes.map(b => b.confidence || 0));
  };

  // Get flagged threat names for CRITICAL bags
  const getFlaggedThreats = () => {
    if (!scanResult?.bboxes) return [];
    return scanResult.bboxes
      .filter(b => b.threat_level === 'CRITICAL')
      .map(b => ({
        id: b.id,
        label: b.label || b.classification || `Object #${b.id + 1}`,
        material: b.material,
        explanation: b.explanation
      }));
  };

  const composition = getComposition();
  const maxConf = getMaxConfidence();
  const flaggedThreats = getFlaggedThreats();

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

      <div className={`bag-queue-belt ${isPlaying ? 'conveyor-animation' : ''}`}>
        {luggageQueue.map((bag) => (
          <div 
            key={bag.id}
            className={`queue-item ${bag.id === selectedBag?.id ? 'active' : ''} ${bag.result?.overall?.level || ''}`}
            onClick={() => handleSelectBag(bag)}
          >
            <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              Bag #{String(bag.id).substring(0, 5)}
            </div>
            {bag.result ? (
              <div style={{ fontSize: '0.65rem', color: getThreatColor(bag.result.overall?.level), fontWeight: 700 }}>
                {bag.result.overall?.level || 'UNKNOWN'}
              </div>
            ) : (
              <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>Pending...</div>
            )}
            {bag.result && bag.result.bboxes?.length > 0 && (
              <div className={`mat-chip ${bag.result.bboxes[0].material}`}>
                {bag.result.bboxes[0].material}
              </div>
            )}
          </div>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '1.25rem', flex: 1, minHeight: 0 }}>
        
        {/* Left: Image Viewer */}
        <div className="glass-panel" style={{ padding: '1rem', display: 'flex', flexDirection: 'column', position: 'relative' }}>
          {scanLoading ? (
            <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.5)', zIndex: 10, borderRadius: 'var(--radius-md)' }}>
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem' }}>
                <RefreshCw size={32} className="animate-spin" color="var(--accent-primary)" />
                <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Extracting 11D attenuations...</span>
              </div>
            </div>
          ) : null}
          
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.2)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--glass-border)', overflow: 'hidden', position: 'relative' }}>
            {scanResult?.annotated_image ? (
              <img src={ensureDataUri(scanResult.annotated_image)} alt="Scan Result" style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }} />
            ) : (
              <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Select a bag to view analysis</div>
            )}
          </div>

          {scanResult && scanResult.bboxes?.length > 0 && (
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

        {/* Right: Bag Diagnostics */}
        <div className="glass-panel" style={{ padding: '1.25rem', overflowY: 'auto' }}>
          <h3 className="section-subtitle">Bag Diagnostics</h3>
          {scanResult ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', marginTop: '1rem' }}>

              {/* Overall Verdict */}
              <div className="glass-inset" style={{ padding: '1rem', borderLeft: `4px solid ${getThreatColor(scanResult.overall?.level || 'SAFE')}` }}>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>Overall Verdict</div>
                <div style={{ fontSize: '1.1rem', fontWeight: 800, color: getThreatColor(scanResult.overall?.level || 'SAFE') }}>{scanResult.overall?.level || 'SAFE'}</div>
                <div style={{ fontSize: '0.8rem', marginTop: '0.3rem' }}>{scanResult.overall?.explanation || 'Cleared.'}</div>
              </div>

              {/* Professional Metadata Strip */}
              <div className="glass-inset" style={{ padding: '0.85rem', display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
                <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 700, letterSpacing: '0.5px' }}>Scan Metadata</div>
                
                {/* Scan ID */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Fingerprint size={14} color="var(--accent-primary)" style={{ flexShrink: 0 }} />
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>Scan ID</div>
                    <div style={{ fontSize: '0.8rem', fontWeight: 600, fontFamily: 'monospace', color: 'var(--text-primary)' }}>
                      {scanResult.scan_id != null ? `SCN-${String(scanResult.scan_id).padStart(6, '0')}` : `SCN-${String(selectedBag?.id || 0).substring(0, 6).padStart(6, '0')}`}
                    </div>
                  </div>
                </div>

                {/* Timestamp */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Clock size={14} color="var(--accent-secondary)" style={{ flexShrink: 0 }} />
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>Timestamp</div>
                    <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                      {new Date().toLocaleString('en-US', { year: 'numeric', month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                    </div>
                  </div>
                </div>

                {/* AI Confidence */}
                {maxConf != null && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Cpu size={14} color="var(--accent-primary)" style={{ flexShrink: 0 }} />
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>AI Confidence (Max)</div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <div style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                          {(maxConf * 100).toFixed(1)}%
                        </div>
                        <div style={{ flex: 1, height: '5px', background: 'rgba(255,255,255,0.04)', borderRadius: 'var(--radius-full)', overflow: 'hidden' }}>
                          <div style={{ width: `${maxConf * 100}%`, height: '100%', borderRadius: 'var(--radius-full)', background: 'var(--accent-gradient)', transition: 'width 0.4s var(--ease-smooth)' }} />
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Inference Mode */}
                {scanResult.mode && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Crosshair size={14} color="var(--text-muted)" style={{ flexShrink: 0 }} />
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>Engine</div>
                      <div style={{ fontSize: '0.75rem', fontWeight: 500, color: 'var(--text-secondary)' }}>{scanResult.mode}</div>
                    </div>
                  </div>
                )}
              </div>

              {/* Material Composition Breakdown */}
              {composition.length > 0 && (
                <div className="glass-inset" style={{ padding: '0.85rem' }}>
                  <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 700, letterSpacing: '0.5px', marginBottom: '0.6rem' }}>Material Composition</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    {composition.map(({ material, percentage }) => (
                      <div key={material} style={{ display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                            <div style={{ width: '7px', height: '7px', borderRadius: '50%', background: getMaterialColor(material), flexShrink: 0 }} />
                            <span style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', fontWeight: 500, textTransform: 'capitalize' }}>{material}</span>
                          </div>
                          <span style={{ fontSize: '0.72rem', fontWeight: 700, color: getMaterialColor(material), fontVariantNumeric: 'tabular-nums' }}>{percentage}%</span>
                        </div>
                        <div style={{ height: '5px', background: 'rgba(255,255,255,0.04)', borderRadius: 'var(--radius-full)', overflow: 'hidden' }}>
                          <div style={{
                            width: `${percentage}%`, height: '100%', borderRadius: 'var(--radius-full)',
                            background: getMaterialColor(material),
                            transition: 'width 0.6s var(--ease-smooth)'
                          }} />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Flagged Threats (CRITICAL only) */}
              {flaggedThreats.length > 0 && (
                <div className="glass-inset" style={{ padding: '0.85rem', borderLeft: '3px solid var(--color-critical)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.5rem' }}>
                    <ShieldAlert size={14} color="var(--color-critical)" />
                    <span style={{ fontSize: '0.65rem', color: 'var(--color-critical)', textTransform: 'uppercase', fontWeight: 700, letterSpacing: '0.5px' }}>Flagged Threats</span>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem' }}>
                    {flaggedThreats.map((t, idx) => (
                      <div key={idx} style={{ display: 'flex', flexDirection: 'column', gap: '0.15rem', padding: '0.4rem 0.6rem', background: 'rgba(239, 68, 68, 0.06)', borderRadius: 'var(--radius-xs)', border: '1px solid rgba(239, 68, 68, 0.12)' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--color-critical)' }}>{t.label}</span>
                          <span className={`mat-chip ${t.material}`} style={{ fontSize: '0.6rem', padding: '1px 6px' }}>{t.material}</span>
                        </div>
                        <span style={{ fontSize: '0.68rem', color: 'var(--text-secondary)', lineHeight: 1.35 }}>{t.explanation}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Selected Object Properties - Safely handles Array and Object dictionary properties */}
              {(scanResult.properties?.[selectedBoxId] || scanResult.properties?.[String(selectedBoxId)]) && (
                <>
                  <h4 className="section-subtitle" style={{ marginTop: '0.5rem' }}>Selected Object Properties</h4>
                  <div className="props-list">
                    {Object.entries(
                      scanResult.properties?.[selectedBoxId] || scanResult.properties?.[String(selectedBoxId)] || {}
                    ).slice(0, 8).map(([key, value]) => {
                      const norm = Math.min((value / (key === 'length_to_width_ratio' ? 10 : 1)) * 100, 100);
                      return (
                        <div key={key} className="prop-bar-container">
                          <div className="prop-label-row">
                            <span className="label">{key.replace(/_/g, ' ')}</span>
                            <span className="value">{typeof value === 'number' ? value.toFixed(2) : value}</span>
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
            <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Waiting for scan data...</div>
          )}
        </div>
      </div>
    </div>
  );
}
