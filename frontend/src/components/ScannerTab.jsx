import React, { useRef, useState, useCallback } from 'react';
import {
  Upload, RefreshCw, FileText, Printer, Download,
  ShieldAlert, ShieldCheck, BadgeInfo, CheckCircle,
  AlertTriangle, Crosshair, Zap, Eye, Target, Scan
} from 'lucide-react';
import { useReactToPrint } from 'react-to-print';
import {
  Radar, RadarChart, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, ResponsiveContainer, Tooltip
} from 'recharts';

// ─── Property Display Names & Icons ──────────────────────
const PROPERTY_META = {
  edge_sharpness:           { label: 'Edge Sharpness',       unit: '', icon: '🔪', max: 1 },
  length_to_width_ratio:    { label: 'Length/Width Ratio',    unit: '', icon: '📏', max: 10 },
  symmetry_score:           { label: 'Symmetry Score',        unit: '', icon: '⚖️',  max: 1 },
  curvature_index:          { label: 'Curvature Index',       unit: '', icon: '🔄', max: 1 },
  approximate_volume:       { label: 'Approx. Volume',        unit: '', icon: '📦', max: 1 },
  material_category:        { label: 'Material Category',     unit: '', icon: '🧱', max: 3 },
  avg_absorption_intensity: { label: 'Absorption Intensity',  unit: '', icon: '☢️',  max: 1 },
  material_homogeneity:     { label: 'Material Homogeneity',  unit: '', icon: '🧬', max: 1 },
  density_level:            { label: 'Density Level',         unit: '', icon: '⚫', max: 1 },
  sharp_edge_count:         { label: 'Sharp Edge Count',      unit: '', icon: '✂️',  max: 20 },
  occlusion_score:          { label: 'Occlusion Score',       unit: '', icon: '👁️',  max: 1 },
};

const MATERIAL_LABELS = { 0: 'Organic', 1: 'Metallic', 2: 'Mixed', 3: 'Opaque' };

// ─── Printable Incident Report Template ─────────────────
class IncidentReportTemplate extends React.Component {
  render() {
    const { scanResult, selectedBoxId } = this.props;
    if (!scanResult) return null;

    const threats = scanResult.bboxes.filter(
      b => b.threat_level === 'CRITICAL' || b.threat_level === 'WARNING'
    );
    const allBoxes = scanResult.bboxes;
    const overall = scanResult.overall;
    const incidentId = `INC-${Date.now().toString(36).toUpperCase()}`;
    const now = new Date();

    return (
      <div style={{
        padding: '48px 56px', fontFamily: "'Inter', 'Helvetica Neue', Arial, sans-serif",
        color: '#0f172a', background: '#fff', lineHeight: 1.6, fontSize: '13px'
      }}>
        {/* ── Cover Header ────────────────────────── */}
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
          borderBottom: '3px solid #0f172a', paddingBottom: '20px', marginBottom: '24px'
        }}>
          <div>
            <h1 style={{ fontSize: '22px', fontWeight: 800, margin: 0, letterSpacing: '0.5px' }}>
              VIT SECURITY SYSTEMS
            </h1>
            <h2 style={{ fontSize: '14px', color: '#64748b', margin: '4px 0 0', fontWeight: 600 }}>
              OFFICIAL X-RAY SCAN INCIDENT REPORT
            </h2>
          </div>
          <div style={{ textAlign: 'right', fontSize: '11px', color: '#475569' }}>
            <p style={{ margin: '0 0 3px' }}><strong>Incident ID:</strong> {incidentId}</p>
            <p style={{ margin: '0 0 3px' }}><strong>Date:</strong> {now.toLocaleDateString('en-IN', { year: 'numeric', month: 'long', day: 'numeric' })}</p>
            <p style={{ margin: '0 0 3px' }}><strong>Time:</strong> {now.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</p>
            <p style={{ margin: 0 }}><strong>Operator:</strong> SEC-A942</p>
          </div>
        </div>

        {/* ── Overall Verdict Banner ──────────────── */}
        <div style={{
          background: overall.level === 'CRITICAL' ? '#fef2f2' : overall.level === 'WARNING' ? '#fffbeb' : '#f0fdf4',
          color: overall.level === 'CRITICAL' ? '#991b1b' : overall.level === 'WARNING' ? '#92400e' : '#166534',
          padding: '14px 20px', borderRadius: '6px',
          border: `1.5px solid ${overall.level === 'CRITICAL' ? '#fca5a5' : overall.level === 'WARNING' ? '#fcd34d' : '#86efac'}`,
          marginBottom: '24px', fontWeight: 700, fontSize: '14px',
          display: 'flex', alignItems: 'center', gap: '10px'
        }}>
          {overall.level === 'CRITICAL' ? '🚨' : overall.level === 'WARNING' ? '⚠️' : '✅'}
          OVERALL VERDICT: {overall.level} — {overall.classification}
        </div>

        {/* ── Summary ─────────────────────────────── */}
        <div style={{ marginBottom: '24px', fontSize: '12px', color: '#334155' }}>
          <strong>AI Assessment:</strong> {overall.explanation}
        </div>

        {/* ── Evidence Images ─────────────────────── */}
        <h3 style={{ fontSize: '14px', borderBottom: '1px solid #e2e8f0', paddingBottom: '6px', marginBottom: '14px', fontWeight: 700 }}>
          Visual Evidence
        </h3>
        <div style={{ display: 'flex', gap: '16px', marginBottom: '28px' }}>
          <div style={{ flex: 1 }}>
            <p style={{ fontSize: '10px', fontWeight: 700, marginBottom: '6px', color: '#64748b', textTransform: 'uppercase' }}>
              Raw X-Ray Capture
            </p>
            <img src={scanResult.original_image} alt="Raw X-Ray" style={{ width: '100%', border: '1px solid #e2e8f0', borderRadius: '4px' }} />
          </div>
          <div style={{ flex: 1 }}>
            <p style={{ fontSize: '10px', fontWeight: 700, marginBottom: '6px', color: '#64748b', textTransform: 'uppercase' }}>
              AI Bounding Box Overlay
            </p>
            <img src={scanResult.annotated_image} alt="AI Analysis" style={{ width: '100%', border: '1px solid #e2e8f0', borderRadius: '4px' }} />
          </div>
        </div>

        {/* ── Detected Objects Table ──────────────── */}
        <h3 style={{ fontSize: '14px', borderBottom: '1px solid #e2e8f0', paddingBottom: '6px', marginBottom: '14px', fontWeight: 700 }}>
          Detected Objects ({allBoxes.length})
        </h3>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px', marginBottom: '28px' }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ border: '1px solid #e2e8f0', padding: '8px', textAlign: 'left' }}>#</th>
              <th style={{ border: '1px solid #e2e8f0', padding: '8px', textAlign: 'left' }}>Material</th>
              <th style={{ border: '1px solid #e2e8f0', padding: '8px', textAlign: 'left' }}>Threat Level</th>
              <th style={{ border: '1px solid #e2e8f0', padding: '8px', textAlign: 'left' }}>Confidence</th>
              <th style={{ border: '1px solid #e2e8f0', padding: '8px', textAlign: 'left' }}>AI Explanation</th>
            </tr>
          </thead>
          <tbody>
            {allBoxes.map((box, i) => (
              <tr key={i} style={{ background: box.threat_level === 'CRITICAL' ? '#fef2f2' : box.threat_level === 'WARNING' ? '#fffbeb' : 'white' }}>
                <td style={{ border: '1px solid #e2e8f0', padding: '8px', fontWeight: 600 }}>{i + 1}</td>
                <td style={{ border: '1px solid #e2e8f0', padding: '8px', textTransform: 'uppercase' }}>{box.material}</td>
                <td style={{ border: '1px solid #e2e8f0', padding: '8px', fontWeight: 700,
                  color: box.threat_level === 'CRITICAL' ? '#dc2626' : box.threat_level === 'WARNING' ? '#d97706' : '#16a34a'
                }}>
                  {box.threat_level}
                </td>
                <td style={{ border: '1px solid #e2e8f0', padding: '8px' }}>
                  {(box.confidence * 100).toFixed(1)}%
                </td>
                <td style={{ border: '1px solid #e2e8f0', padding: '8px', fontSize: '10px', color: '#475569' }}>
                  {box.explanation}
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {/* ── Suspicious Items Detail ─────────────── */}
        {threats.length > 0 && (
          <>
            <h3 style={{ fontSize: '14px', borderBottom: '1px solid #e2e8f0', paddingBottom: '6px', marginBottom: '14px', fontWeight: 700, color: '#dc2626' }}>
              ⚠ Suspicious Items — Detailed Analysis
            </h3>
            {threats.map((box, i) => {
              const props = scanResult.properties?.[box.id] || {};
              return (
                <div key={i} style={{
                  border: `1.5px solid ${box.threat_level === 'CRITICAL' ? '#fca5a5' : '#fcd34d'}`,
                  borderRadius: '6px', padding: '16px', marginBottom: '16px',
                  background: box.threat_level === 'CRITICAL' ? '#fef2f2' : '#fffbeb'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
                    <div>
                      <span style={{ fontWeight: 700, fontSize: '13px' }}>Object #{box.id + 1}</span>
                      <span style={{ marginLeft: '8px', fontSize: '11px', color: '#64748b' }}>
                        ({box.material.toUpperCase()})
                      </span>
                    </div>
                    <span style={{
                      fontSize: '11px', fontWeight: 700, padding: '2px 10px', borderRadius: '10px',
                      background: box.threat_level === 'CRITICAL' ? '#fee2e2' : '#fef3c7',
                      color: box.threat_level === 'CRITICAL' ? '#dc2626' : '#d97706'
                    }}>
                      {box.threat_level}
                    </span>
                  </div>
                  <p style={{ fontSize: '11px', color: '#334155', marginBottom: '12px' }}>
                    <strong>AI Explanation:</strong> {box.explanation}
                  </p>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '10px' }}>
                    <thead>
                      <tr style={{ background: '#f1f5f9' }}>
                        <th style={{ border: '1px solid #e2e8f0', padding: '6px', textAlign: 'left' }}>Property</th>
                        <th style={{ border: '1px solid #e2e8f0', padding: '6px', textAlign: 'left' }}>Value</th>
                        <th style={{ border: '1px solid #e2e8f0', padding: '6px', textAlign: 'left' }}>Risk Indicator</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(props).map(([key, val]) => {
                        const meta = PROPERTY_META[key] || { label: key, max: 1 };
                        const norm = typeof val === 'number' ? Math.min(val / meta.max, 1) : 0;
                        let risk = 'Normal';
                        if (key === 'density_level' && val > 0.7) risk = '⚠ HIGH';
                        if (key === 'edge_sharpness' && val > 0.5) risk = '⚠ HIGH';
                        if (key === 'sharp_edge_count' && val > 6) risk = '⚠ HIGH';
                        if (key === 'material_category' && (val === 1 || val === 3)) risk = '⚠ FLAGGED';
                        return (
                          <tr key={key}>
                            <td style={{ border: '1px solid #e2e8f0', padding: '6px' }}>{meta.label}</td>
                            <td style={{ border: '1px solid #e2e8f0', padding: '6px', fontVariantNumeric: 'tabular-nums' }}>
                              {typeof val === 'number' ? (Number.isInteger(val) ? val : val.toFixed(4)) : val}
                            </td>
                            <td style={{
                              border: '1px solid #e2e8f0', padding: '6px',
                              color: risk.includes('⚠') ? '#dc2626' : '#16a34a', fontWeight: 600
                            }}>
                              {risk}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              );
            })}
          </>
        )}

        {/* ── Scan Metadata ───────────────────────── */}
        <h3 style={{ fontSize: '14px', borderBottom: '1px solid #e2e8f0', paddingBottom: '6px', marginBottom: '14px', fontWeight: 700, marginTop: '20px' }}>
          Scan Metadata
        </h3>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px', marginBottom: '40px' }}>
          <tbody>
            {[
              ['Inference Mode', scanResult.mode],
              ['Scan ID', scanResult.scan_id || 'N/A'],
              ['Total Objects Detected', allBoxes.length],
              ['Critical Threats', threats.filter(t => t.threat_level === 'CRITICAL').length],
              ['Warning Alerts', threats.filter(t => t.threat_level === 'WARNING').length],
              ['Checkpoint / Terminal', 'T4 - Alpha Gate'],
              ['Operator ID', 'SEC-A942'],
            ].map(([label, value]) => (
              <tr key={label}>
                <td style={{ border: '1px solid #e2e8f0', padding: '8px', fontWeight: 600, width: '40%', background: '#f8fafc' }}>{label}</td>
                <td style={{ border: '1px solid #e2e8f0', padding: '8px' }}>{value}</td>
              </tr>
            ))}
          </tbody>
        </table>

        {/* ── Signatures ──────────────────────────── */}
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '40px' }}>
          <div style={{ width: '42%' }}>
            <div style={{ borderBottom: '1px solid #0f172a', height: '36px' }} />
            <p style={{ fontSize: '10px', marginTop: '6px', color: '#64748b' }}>Inspecting Officer Signature & Date</p>
          </div>
          <div style={{ width: '42%' }}>
            <div style={{ borderBottom: '1px solid #0f172a', height: '36px' }} />
            <p style={{ fontSize: '10px', marginTop: '6px', color: '#64748b' }}>Shift Supervisor Signature & Date</p>
          </div>
        </div>

        {/* ── Footer ──────────────────────────────── */}
        <div style={{ marginTop: '32px', paddingTop: '12px', borderTop: '1px solid #e2e8f0', fontSize: '9px', color: '#94a3b8', textAlign: 'center' }}>
          This is a computer-generated report by VIT Security Systems — Dual-Model AI X-Ray Scanner.
          Report ID: {incidentId} | Generated: {now.toISOString()}
        </div>
      </div>
    );
  }
}

// ─── Main Scanner Tab ────────────────────────────────────
export default function ScannerTab({
  uploadedFile, setUploadedFile, uploadPreview, setUploadPreview,
  manualScanResult, setManualScanResult, scanLoading, setScanLoading,
  hoveredBoxId, setHoveredBoxId, selectedBoxId, setSelectedBoxId, analyzerImgRef
}) {
  const reportRef = useRef();
  const fileInputRef = useRef();
  const [isDragOver, setIsDragOver] = useState(false);

  // react-to-print hook
  const handlePrint = useReactToPrint({
    contentRef: reportRef,
    documentTitle: `VIT_Security_Report_${Date.now()}`,
  });

  // ── File Handling ──────────────────────────
  const processFile = useCallback((file) => {
    if (!file) return;
    setUploadedFile(file);
    setUploadPreview(URL.createObjectURL(file));
    setManualScanResult(null);
    setSelectedBoxId(null);

    setScanLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    fetch('/api/scan', { method: 'POST', body: formData })
      .then(async res => {
        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || `Server Error ${res.status}`);
        }
        return res.json();
      })
      .then(data => {
        setManualScanResult(data);
        setScanLoading(false);
        if (data.bboxes && data.bboxes.length > 0) setSelectedBoxId(0);
      })
      .catch(err => {
        console.error("Scan failed:", err);
        setScanLoading(false);
        import('sonner').then(({ toast }) => toast.error(err.message));
      });
  }, [setUploadedFile, setUploadPreview, setManualScanResult, setSelectedBoxId, setScanLoading]);

  const handleFileUpload = (e) => processFile(e.target.files[0]);

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) processFile(file);
  };

  const handleDragOver = (e) => { e.preventDefault(); setIsDragOver(true); };
  const handleDragLeave = () => setIsDragOver(false);

  // ── Derived Data ───────────────────────────
  const selectedBox = manualScanResult?.bboxes?.[selectedBoxId];
  const selectedProps = manualScanResult?.properties?.[selectedBoxId] || {};
  const threats = manualScanResult?.bboxes?.filter(b => b.threat_level === 'CRITICAL' || b.threat_level === 'WARNING') || [];

  // Radar chart data
  const radarData = selectedProps ? [
    { subject: 'Density', value: (selectedProps.density_level || 0) * 100 },
    { subject: 'Sharpness', value: (selectedProps.edge_sharpness || 0) * 100 },
    { subject: 'Symmetry', value: (selectedProps.symmetry_score || 0) * 100 },
    { subject: 'Absorption', value: (selectedProps.avg_absorption_intensity || 0) * 100 },
    { subject: 'Curvature', value: (selectedProps.curvature_index || 0) * 100 },
    { subject: 'Volume', value: (selectedProps.approximate_volume || 0) * 100 },
  ] : [];

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

      {/* ── Top Action Bar ─────────────────────── */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 className="section-title">
          <Scan size={22} color="var(--accent-primary)" />
          Deep Property Analyzer
        </h2>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          {manualScanResult && (
            <>
              <button onClick={handlePrint} className="btn-secondary" id="btn-generate-report">
                <FileText size={15} /> Generate Report
              </button>
              <button onClick={handlePrint} className="btn-primary" id="btn-print-report">
                <Printer size={15} /> Print / Download PDF
              </button>
            </>
          )}
          <label htmlFor="upload-file-input" className="btn-primary" style={{ cursor: 'pointer' }}>
            <Upload size={15} /> Upload X-Ray
          </label>
          <input
            ref={fileInputRef}
            id="upload-file-input"
            type="file"
            accept="image/jpeg,image/png,image/webp"
            style={{ display: 'none' }}
            onChange={handleFileUpload}
          />
        </div>
      </div>

      {/* ── Main Content: Image Viewer + Analysis Panel ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 380px', gap: '1.25rem', flex: 1, minHeight: 0 }}>

        {/* ─── LEFT: Image Viewer ──────────────── */}
        <div className="glass-panel" style={{ padding: '1.25rem', display: 'flex', flexDirection: 'column', gap: '1rem', overflow: 'hidden' }}>

          {/* Viewer Area */}
          <div
            style={{
              position: 'relative', flex: 1, background: 'rgba(0,0,0,0.2)',
              borderRadius: 'var(--radius-sm)', border: '1px solid var(--glass-border)',
              display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden',
              minHeight: '300px'
            }}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
          >
            {scanLoading ? (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
                <RefreshCw size={36} className="animate-spin" color="var(--accent-primary)" />
                <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  Running dual-head inference pipeline...
                </span>
                <div className="shimmer" style={{ width: '200px', height: '6px', borderRadius: '3px' }} />
              </div>
            ) : manualScanResult ? (
              <div style={{ position: 'relative', maxWidth: '100%', maxHeight: '100%', display: 'flex', justifyContent: 'center' }}>
                <img
                  ref={analyzerImgRef}
                  src={manualScanResult.annotated_image}
                  alt="AI-Analyzed X-Ray"
                  style={{ maxWidth: '100%', maxHeight: '480px', objectFit: 'contain', display: 'block', borderRadius: '4px' }}
                />
                {/* Interactive overlay boxes */}
                {manualScanResult.bboxes.map((box, index) => {
                  const imgEl = analyzerImgRef.current;
                  if (!imgEl) return null;
                  const origW = imgEl.naturalWidth || 640;
                  const origH = imgEl.naturalHeight || 480;
                  const leftPct = (box.x1 / origW) * 100;
                  const topPct = (box.y1 / origH) * 100;
                  const widthPct = ((box.x2 - box.x1) / origW) * 100;
                  const heightPct = ((box.y2 - box.y1) / origH) * 100;
                  const isSelected = selectedBoxId === index;
                  const isHovered = hoveredBoxId === index;
                  const color = box.threat_level === 'CRITICAL' ? 'var(--color-critical)' :
                                box.threat_level === 'WARNING' ? 'var(--color-warning)' :
                                getMaterialColor(box.material);
                  return (
                    <button
                      key={box.id}
                      id={`bbox-overlay-${box.id}`}
                      aria-label={`Object ${box.id + 1}: ${box.material} - ${box.threat_level}`}
                      style={{
                        position: 'absolute',
                        left: `${leftPct}%`, top: `${topPct}%`,
                        width: `${widthPct}%`, height: `${heightPct}%`,
                        border: isSelected ? `2.5px solid ${color}` : isHovered ? `1.5px dashed ${color}` : '1.5px solid transparent',
                        background: isSelected ? `${color}11` : isHovered ? 'rgba(255,255,255,0.03)' : 'transparent',
                        cursor: 'pointer', outline: 'none', padding: 0,
                        borderRadius: '3px', transition: 'all 0.15s ease'
                      }}
                      onMouseEnter={() => setHoveredBoxId(index)}
                      onMouseLeave={() => setHoveredBoxId(null)}
                      onClick={() => setSelectedBoxId(index)}
                    />
                  );
                })}
              </div>
            ) : (
              <div
                className={`dropzone ${isDragOver ? 'drag-over' : ''}`}
                style={{ width: '100%', height: '100%', border: 'none', background: 'transparent' }}
                onClick={() => fileInputRef.current?.click()}
              >
                <div className="dropzone-icon">
                  <Upload size={28} color="var(--accent-primary)" />
                </div>
                <div style={{ textAlign: 'center' }}>
                  <p style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', fontWeight: 600 }}>
                    Drop an X-Ray image here
                  </p>
                  <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.3rem' }}>
                    or click to browse • Supports JPEG, PNG, WebP
                  </p>
                </div>
                <div style={{ display: 'flex', gap: '2rem', marginTop: '0.5rem' }}>
                  {[
                    { icon: <Crosshair size={14} />, label: 'Object Detection' },
                    { icon: <Zap size={14} />, label: '11-D Properties' },
                    { icon: <Target size={14} />, label: 'Threat Classification' },
                  ].map(item => (
                    <span key={item.label} style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.7rem', color: 'var(--text-dim)' }}>
                      {item.icon} {item.label}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Object Selection Strip */}
          {manualScanResult && manualScanResult.bboxes.length > 0 && (
            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
              {manualScanResult.bboxes.map((box, i) => (
                <button
                  key={i}
                  id={`obj-select-${i}`}
                  onClick={() => setSelectedBoxId(i)}
                  className={selectedBoxId === i ? 'btn-primary' : 'btn-ghost'}
                  style={{
                    fontSize: '0.72rem', padding: '0.4rem 0.75rem',
                    borderRadius: 'var(--radius-full)',
                    border: selectedBoxId === i ? 'none' : `1px solid ${getThreatColor(box.threat_level)}33`,
                    color: selectedBoxId === i ? 'white' : getThreatColor(box.threat_level)
                  }}
                >
                  #{i + 1} {box.material.toUpperCase()}
                </button>
              ))}
            </div>
          )}

          {/* Metadata Strip */}
          <div className="metadata-strip">
            <div className="metadata-item">
              <span className="metadata-label">Operator ID</span>
              <span className="metadata-value">SEC-A942</span>
            </div>
            <div className="metadata-item">
              <span className="metadata-label">Checkpoint</span>
              <span className="metadata-value">T4 - Alpha Gate</span>
            </div>
            <div className="metadata-item">
              <span className="metadata-label">Scan Time</span>
              <span className="metadata-value">{new Date().toLocaleTimeString()}</span>
            </div>
            <div className="metadata-item">
              <span className="metadata-label">Engine Mode</span>
              <span className="metadata-value" style={{ color: 'var(--accent-primary)' }}>
                {manualScanResult ? (manualScanResult.mode.includes('Deep') ? 'DEEP LEARNING' : 'CV PIPELINE') : 'STANDBY'}
              </span>
            </div>
            <div className="metadata-item">
              <span className="metadata-label">Objects Found</span>
              <span className="metadata-value">{manualScanResult?.bboxes?.length || '—'}</span>
            </div>
          </div>
        </div>

        {/* ─── RIGHT: Analysis Panel ──────────── */}
        <div className="glass-panel" style={{ padding: '1.25rem', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

          <h3 className="section-subtitle" style={{ marginBottom: '0.75rem' }}>AI Threat Evaluation</h3>

          {selectedBox ? (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '1rem', overflowY: 'auto', paddingRight: '4px' }}>

              {/* Threat Badge */}
              <div className="glass-inset" style={{
                padding: '1rem', display: 'flex', alignItems: 'center', gap: '1rem',
                borderColor: `${getThreatColor(selectedBox.threat_level)}33`
              }}>
                <div style={{
                  width: '64px', height: '64px', borderRadius: '50%', flexShrink: 0,
                  background: `${getThreatColor(selectedBox.threat_level)}15`,
                  border: `2.5px solid ${getThreatColor(selectedBox.threat_level)}`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center'
                }}>
                  <span style={{ fontSize: '1.25rem', fontWeight: 800, color: getThreatColor(selectedBox.threat_level), fontVariantNumeric: 'tabular-nums' }}>
                    {Math.round((selectedBox.confidence || 0.95) * 100)}%
                  </span>
                </div>
                <div>
                  <span style={{ display: 'block', fontSize: '0.65rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>
                    Classification
                  </span>
                  <span style={{ display: 'block', fontSize: '1.05rem', fontWeight: 800, color: getThreatColor(selectedBox.threat_level) }}>
                    {selectedBox.threat_level}
                  </span>
                  <span className={`mat-chip ${selectedBox.material}`} style={{ marginTop: '0.25rem' }}>
                    {selectedBox.material}
                  </span>
                </div>
              </div>

              {/* AI Verdict */}
              <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                <strong style={{ color: 'var(--accent-primary)' }}>AI Verdict: </strong>
                {selectedBox.explanation}
              </div>

              {/* Radar Chart */}
              {radarData.length > 0 && (
                <div style={{ height: '200px', width: '100%' }}>
                  <ResponsiveContainer>
                    <RadarChart cx="50%" cy="50%" outerRadius="68%" data={radarData}>
                      <PolarGrid stroke="rgba(255,255,255,0.06)" />
                      <PolarAngleAxis dataKey="subject" tick={{ fill: 'var(--text-muted)', fontSize: 10 }} />
                      <PolarRadiusAxis tick={false} axisLine={false} />
                      <Radar
                        name="Properties"
                        dataKey="value"
                        stroke="var(--accent-primary)"
                        fill="var(--accent-primary)"
                        fillOpacity={0.25}
                        strokeWidth={2}
                      />
                      <Tooltip
                        contentStyle={{ background: '#0a0e1a', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '8px', fontSize: '12px' }}
                        formatter={(v) => [`${v.toFixed(1)}%`]}
                      />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* Property Bars */}
              <div>
                <h4 className="section-subtitle">Deep Property Vector</h4>
                <div className="props-grid">
                  {Object.entries(selectedProps).map(([key, value]) => {
                    const meta = PROPERTY_META[key] || { label: key, max: 1 };
                    const norm = typeof value === 'number' ? Math.min((value / meta.max) * 100, 100) : 0;
                    const isHigh = key === 'density_level' && value > 0.7 ||
                                   key === 'edge_sharpness' && value > 0.5 ||
                                   key === 'sharp_edge_count' && value > 6;
                    return (
                      <div key={key} className="prop-bar-container">
                        <div className="prop-label-row">
                          <span className="label">{meta.icon} {meta.label}</span>
                          <span className="value" style={isHigh ? { color: 'var(--color-critical)' } : {}}>
                            {typeof value === 'number' ? (Number.isInteger(value) ? value : value.toFixed(3)) : value}
                          </span>
                        </div>
                        <div className="prop-bar-outer">
                          <div
                            className="prop-bar-inner"
                            style={{
                              width: `${norm}%`,
                              background: isHigh
                                ? 'linear-gradient(90deg, var(--color-warning), var(--color-critical))'
                                : 'linear-gradient(90deg, var(--accent-primary), var(--accent-secondary))'
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Alert Summary */}
              {threats.length > 0 && (
                <div className="glass-inset" style={{
                  padding: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.6rem',
                  borderColor: 'rgba(239, 68, 68, 0.15)'
                }}>
                  <AlertTriangle size={18} color="var(--color-critical)" />
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                    <strong style={{ color: 'var(--color-critical)' }}>{threats.length} suspicious item(s)</strong> flagged.
                    Generate a PDF report for detailed documentation.
                  </div>
                </div>
              )}

            </div>
          ) : (
            <div style={{
              flex: 1, display: 'flex', flexDirection: 'column',
              justifyContent: 'center', alignItems: 'center', gap: '1rem',
              color: 'var(--text-muted)', textAlign: 'center'
            }}>
              <Eye size={40} style={{ opacity: 0.3 }} />
              <div>
                <p style={{ fontSize: '0.85rem', fontWeight: 600 }}>No Analysis Data</p>
                <p style={{ fontSize: '0.75rem', marginTop: '0.3rem' }}>
                  Upload an X-Ray image to begin deep property extraction and threat classification.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── Hidden Print Component ────────────── */}
      <div style={{ display: 'none' }}>
        <div ref={reportRef}>
          <IncidentReportTemplate
            scanResult={manualScanResult}
            selectedBoxId={selectedBoxId}
          />
        </div>
      </div>
    </div>
  );
}
