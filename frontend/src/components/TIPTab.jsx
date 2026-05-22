import React from 'react';
import { Layers, RefreshCw, Play, Compass } from 'lucide-react';

export default function TIPTab({
  tipBgType, setTipBgType, tipThreatType, setTipThreatType, 
  tipScale, setTipScale, tipAngle, setTipAngle, 
  tipPosX, setTipPosX, tipPosY, setTipPosY, 
  tipThickness, setTipThickness, tipResult, tipLoading,
  tipHoveredBoxId, setTipHoveredBoxId, tipSelectedBoxId, setTipSelectedBoxId,
  threatPresets, handleRunTIP, tipImgRef
}) {
  const ensureDataUri = (src) => {
    if (!src) return null;
    if (src.startsWith('data:')) return src;
    return `data:image/jpeg;base64,${src}`;
  };

  return (
    <div className="animate-fadeIn" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', height: '100%' }}>
      <h2 className="section-title">
        <Layers size={22} color="var(--accent-primary)" />
        Threat Image Projection Sandbox
      </h2>

      <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: '1.25rem', flex: 1, minHeight: 0 }}>
        
        {/* Controls Panel */}
        <div className="glass-panel" style={{ padding: '1.25rem', display: 'flex', flexDirection: 'column', gap: '1.25rem', overflowY: 'auto' }}>
          <div>
            <label style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.5rem', display: 'block' }}>Background Image</label>
            <div style={{ width: '100%', padding: '0.6rem', borderRadius: 'var(--radius-sm)', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', color: 'var(--text-primary)', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <Layers size={14} color="var(--text-secondary)" /> Random Safe X-Ray (Dataset)
            </div>
          </div>

          <div>
            <label style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.5rem', display: 'block' }}>Threat Preset</label>
            <div style={{ width: '100%', padding: '0.6rem', borderRadius: 'var(--radius-sm)', background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.2)', color: 'var(--color-critical)', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <Compass size={14} color="var(--color-critical)" /> Random Threat X-Ray (Dataset)
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.3rem' }}>
                <span>Scale</span><span>{tipScale}x</span>
              </div>
              <input type="range" min="0.3" max="3.0" step="0.1" value={tipScale} onChange={e => setTipScale(parseFloat(e.target.value))} className="slider-control" />
            </div>
            
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.3rem' }}>
                <span>Angle</span><span>{tipAngle}°</span>
              </div>
              <input type="range" min="0" max="360" step="5" value={tipAngle} onChange={e => setTipAngle(parseInt(e.target.value))} className="slider-control" />
            </div>
            
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.3rem' }}>
                <span>Position X</span><span>{tipPosX}%</span>
              </div>
              <input type="range" min="0" max="100" step="5" value={tipPosX} onChange={e => setTipPosX(parseInt(e.target.value))} className="slider-control" />
            </div>
            
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.3rem' }}>
                <span>Position Y</span><span>{tipPosY}%</span>
              </div>
              <input type="range" min="0" max="100" step="5" value={tipPosY} onChange={e => setTipPosY(parseInt(e.target.value))} className="slider-control" />
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.3rem' }}>
                <span>Material Thickness (Beer-Lambert)</span><span>{tipThickness}cm</span>
              </div>
              <input type="range" min="0.1" max="3.0" step="0.1" value={tipThickness} onChange={e => setTipThickness(parseFloat(e.target.value))} className="slider-control" />
            </div>
          </div>

          <button onClick={handleRunTIP} className="btn-primary" style={{ marginTop: 'auto' }} disabled={tipLoading}>
            {tipLoading ? <RefreshCw size={16} className="animate-spin" /> : <Play size={16} />}
            Run TIP Projection
          </button>
        </div>

        {/* Result Viewer */}
        <div className="glass-panel" style={{ padding: '1.25rem', display: 'flex', flexDirection: 'column', position: 'relative' }}>
          {tipLoading ? (
            <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.5)', zIndex: 10 }}>
              <RefreshCw size={32} className="animate-spin" color="var(--accent-primary)" />
            </div>
          ) : null}
          
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.3)', borderRadius: 'var(--radius-sm)', overflow: 'hidden', position: 'relative' }}>
            {tipResult ? (
              <>
                <img 
                  ref={tipImgRef}
                  src={ensureDataUri(tipResult.composite_image)} 
                  alt="TIP Result" 
                  style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }} 
                />
                {tipResult.scan_results?.bboxes?.map((box, index) => {
                  if (!tipImgRef.current) return null;
                  const img = tipImgRef.current;
                  const origW = img.naturalWidth || 640;
                  const origH = img.naturalHeight || 480;
                  const l = (box.x1 / origW) * 100, t = (box.y1 / origH) * 100;
                  const w = ((box.x2 - box.x1) / origW) * 100, h = ((box.y2 - box.y1) / origH) * 100;
                  const isSelected = tipSelectedBoxId === index;
                  const color = box.threat_level === 'CRITICAL' ? 'var(--color-critical)' : box.threat_level === 'WARNING' ? 'var(--color-warning)' : 'var(--color-safe)';
                  return (
                    <button
                      key={index}
                      style={{
                        position: 'absolute', left: `${l}%`, top: `${t}%`, width: `${w}%`, height: `${h}%`,
                        border: isSelected ? `2px solid ${color}` : `1px dashed ${color}`,
                        background: isSelected ? `${color}22` : 'transparent',
                        cursor: 'pointer'
                      }}
                      onClick={() => setTipSelectedBoxId(index)}
                    />
                  );
                })}
              </>
            ) : (
              <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Compass size={18} /> Configure parameters and run projection
              </div>
            )}
          </div>

          {tipResult && tipResult.scan_results && (
            <div style={{ marginTop: '1.25rem' }}>
              <div className="glass-inset" style={{ padding: '1rem', borderLeft: `4px solid ${tipResult.scan_results.overall.level === 'CRITICAL' ? 'var(--color-critical)' : 'var(--color-safe)'}` }}>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>AI Verification</div>
                <div style={{ fontSize: '1rem', fontWeight: 800, color: tipResult.scan_results.overall.level === 'CRITICAL' ? 'var(--color-critical)' : 'var(--color-safe)' }}>
                  {tipResult.scan_results.overall.level}
                </div>
                <div style={{ fontSize: '0.8rem', marginTop: '0.3rem' }}>{tipResult.scan_results.overall.explanation}</div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
