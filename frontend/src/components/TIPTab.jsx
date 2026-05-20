import React, { useRef } from 'react';
import { RefreshCw, Compass } from 'lucide-react';

export default function TIPTab({
  tipBgType, setTipBgType, tipThreatType, setTipThreatType,
  tipScale, setTipScale, tipAngle, setTipAngle,
  tipPosX, setTipPosX, tipPosY, setTipPosY,
  tipThickness, setTipThickness, tipResult, tipLoading,
  tipHoveredBoxId, setTipHoveredBoxId, tipSelectedBoxId, setTipSelectedBoxId,
  threatPresets, handleRunTIP, tipImgRef
}) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: '1.5rem', height: '100%' }}>
      {/* Sandbox Controls */}
      <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.5rem', overflowY: 'auto' }}>
        <h2 style={{ fontSize: '1.1rem', fontWeight: '800' }}>Attenuation Physics Config</h2>
        
        <div>
          <label className="form-label">Background Carrier (Safe Bag)</label>
          <select className="form-select" value={tipBgType} onChange={(e) => setTipBgType(e.target.value)}>
            <option value="safe_luggage">Standard Carry-on</option>
            <option value="toolbox">Heavy Toolbox</option>
          </select>
        </div>

        <div>
          <label className="form-label">Threat Object Injection</label>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {threatPresets.map(t => (
              <label key={t.id} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', fontSize: '0.8rem' }}>
                <input 
                  type="radio" 
                  name="threat-preset" 
                  value={t.id} 
                  checked={tipThreatType === t.id}
                  onChange={() => setTipThreatType(t.id)} 
                />
                <span style={{ color: 'var(--text-main)' }}>{t.name}</span>
                <span style={{ marginLeft: 'auto', fontSize: '0.65rem', color: t.complexity === 'Critical' ? 'var(--color-critical)' : 'var(--text-muted)' }}>{t.complexity}</span>
              </label>
            ))}
          </div>
        </div>

        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.2rem' }}>
            <span>Scale</span>
            <span style={{ color: '#fff', fontWeight: '700' }}>{tipScale.toFixed(2)}x</span>
          </div>
          <input type="range" min="0.5" max="2.0" step="0.1" className="slider-control" value={tipScale} onChange={(e) => setTipScale(parseFloat(e.target.value))} />
        </div>

        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.2rem' }}>
            <span>Rotation Angle</span>
            <span style={{ color: '#fff', fontWeight: '700' }}>{tipAngle}°</span>
          </div>
          <input type="range" min="0" max="360" step="1" className="slider-control" value={tipAngle} onChange={(e) => setTipAngle(parseInt(e.target.value))} />
        </div>

        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.2rem' }}>
            <span>Attenuation Thickness</span>
            <span style={{ color: '#fff', fontWeight: '700' }}>{tipThickness.toFixed(2)}</span>
          </div>
          <input type="range" min="0.2" max="3.0" step="0.1" className="slider-control" value={tipThickness} onChange={(e) => setTipThickness(parseFloat(e.target.value))} />
        </div>

        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.2rem' }}>
            <span>X Position</span>
            <span style={{ color: '#fff', fontWeight: '700' }}>{tipPosX}%</span>
          </div>
          <input type="range" min="10" max="90" step="1" className="slider-control" value={tipPosX} onChange={(e) => setTipPosX(parseInt(e.target.value))} />
        </div>

        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.2rem' }}>
            <span>Y Position</span>
            <span style={{ color: '#fff', fontWeight: '700' }}>{tipPosY}%</span>
          </div>
          <input type="range" min="10" max="90" step="1" className="slider-control" value={tipPosY} onChange={(e) => setTipPosY(parseInt(e.target.value))} />
        </div>

        <button 
          className="btn-primary" 
          onClick={handleRunTIP} 
          style={{ width: '100%', marginTop: '0.5rem', display: 'flex', justifyContent: 'center' }}
        >
          <RefreshCw size={16} /> Run Attenuated Projection
        </button>
      </div>

      {/* Sandbox Projection Output */}
      <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: '800', textTransform: 'uppercase', color: 'var(--text-secondary)' }}>Attenuated Synthesis Output</h3>
          {tipResult?.overall && (
            <div className="mat-chip" style={{ background: tipResult.overall.level === 'CRITICAL' ? 'var(--color-critical-glow)' : 'var(--color-safe-glow)', color: tipResult.overall.level === 'CRITICAL' ? 'var(--color-critical)' : 'var(--color-safe)', border: '1px solid currentColor' }}>
              Model 2: {tipResult.overall.level}
            </div>
          )}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '1rem' }}>
          {/* Image Output Display */}
          <div style={{ position: 'relative', width: '100%', height: '380px', background: '#0e111a', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            {tipLoading ? (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem' }}>
                <RefreshCw size={24} className="animate-spin" color="var(--accent-cyan)" />
                <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Computing absorption overlays...</span>
              </div>
            ) : tipResult ? (
              <div style={{ position: 'relative', maxWidth: '100%', maxHeight: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <img 
                  ref={tipImgRef}
                  src={tipResult.projected_image} 
                  alt="Projected Attenuation Case" 
                  style={{ maxWidth: '100%', maxHeight: '360px', objectFit: 'contain', display: 'block' }}
                />
                
                {/* Projects Overlay Boxes */}
                {tipResult.bboxes.map((box, index) => {
                  const origW = 600;
                  const origH = 400;
                  
                  const leftPct = (box.x1 / origW) * 100;
                  const topPct = (box.y1 / origH) * 100;
                  const widthPct = ((box.x2 - box.x1) / origW) * 100;
                  const heightPct = ((box.y2 - box.y1) / origH) * 100;
                  
                  const isSelected = tipSelectedBoxId === index;
                  const isHovered = tipHoveredBoxId === index;
                  
                  let strokeColor = 'var(--color-mixed)';
                  if (box.material === 'organic') strokeColor = 'var(--color-organic)';
                  if (box.material === 'metallic') strokeColor = 'var(--color-metallic)';
                  if (box.material === 'opaque') strokeColor = 'var(--color-opaque)';
                  if (box.threat_level === 'CRITICAL') strokeColor = 'var(--color-critical)';

                  return (
                    <button
                      key={box.id}
                      style={{
                        position: 'absolute',
                        left: `${leftPct}%`,
                        top: `${topPct}%`,
                        width: `${widthPct}%`,
                        height: `${heightPct}%`,
                        border: isSelected ? `3px solid ${strokeColor}` : (isHovered ? `2px dashed ${strokeColor}` : `1px solid ${strokeColor}`),
                        background: isSelected ? 'rgba(56, 189, 248, 0.05)' : (isHovered ? 'rgba(255,255,255,0.02)' : 'transparent'),
                        boxShadow: isSelected ? `0 0 10px ${strokeColor}` : 'none',
                        cursor: 'pointer',
                        outline: 'none',
                        padding: 0
                      }}
                      onMouseEnter={() => setTipHoveredBoxId(index)}
                      onMouseLeave={() => setTipHoveredBoxId(null)}
                      onClick={() => setTipSelectedBoxId(index)}
                    />
                  );
                })}
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', color: 'var(--text-muted)', gap: '0.5rem' }}>
                <Compass size={32} />
                <span style={{ fontSize: '0.85rem' }}>Adjust sliders and click "Run Attenuated Projection".</span>
              </div>
            )}
          </div>
          
          {/* Resulting Properties Inspector */}
          {tipResult?.bboxes[tipSelectedBoxId] && (
            <div style={{ background: 'rgba(255,255,255,0.02)', padding: '1rem', border: '1px solid rgba(255,255,255,0.05)', borderRadius: 'var(--radius-md)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                <h4 style={{ fontSize: '0.85rem', fontWeight: '700' }}>
                  Detected Element: {tipResult.bboxes[tipSelectedBoxId].material}
                </h4>
                <span style={{ fontSize: '0.75rem', fontWeight: '800', color: 'var(--color-critical)' }}>
                  Risk Score: {Math.round(tipResult.bboxes[tipSelectedBoxId].confidence * 100)}%
                </span>
              </div>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
                <strong>AI Verdict:</strong> {tipResult.bboxes[tipSelectedBoxId].explanation}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
