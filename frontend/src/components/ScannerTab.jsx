import React, { useRef } from 'react';
import { Upload, RefreshCw, FileText, Printer, ShieldAlert, BadgeInfo, CheckCircle } from 'lucide-react';
import { useReactToPrint } from 'react-to-print';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from 'recharts';

// --- Hidden Printable Report Component ---
// This is the official incident report that gets generated into a PDF
class IncidentReportTemplate extends React.Component {
  render() {
    const { scanResult, uploadPreview, selectedBoxId } = this.props;
    if (!scanResult) return null;

    const detectedThreats = scanResult.bboxes.filter(b => b.threat_level === 'CRITICAL' || b.threat_level === 'WARNING');
    const box = scanResult.bboxes[selectedBoxId] || scanResult.bboxes[0];

    return (
      <div style={{ padding: '40px', fontFamily: 'Arial, sans-serif', color: '#111', background: '#fff' }}>
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '3px solid #111', paddingBottom: '20px', marginBottom: '20px' }}>
          <div>
            <h1 style={{ fontSize: '24px', fontWeight: 'bold', margin: '0 0 5px 0' }}>VIT SECURITY SYSTEMS</h1>
            <h2 style={{ fontSize: '16px', color: '#555', margin: 0 }}>OFFICIAL INCIDENT REPORT</h2>
          </div>
          <div style={{ textAlign: 'right', fontSize: '12px' }}>
            <p style={{ margin: '0 0 5px 0' }}><strong>Date:</strong> {new Date().toLocaleDateString()}</p>
            <p style={{ margin: '0 0 5px 0' }}><strong>Time:</strong> {new Date().toLocaleTimeString()}</p>
            <p style={{ margin: 0 }}><strong>Incident ID:</strong> #{Math.floor(Math.random() * 90000) + 10000}</p>
          </div>
        </div>

        {/* Status Alert */}
        <div style={{ 
          background: scanResult.overall.level === 'CRITICAL' ? '#fee2e2' : (scanResult.overall.level === 'WARNING' ? '#fef3c7' : '#dcfce7'),
          color: scanResult.overall.level === 'CRITICAL' ? '#b91c1c' : (scanResult.overall.level === 'WARNING' ? '#b45309' : '#15803d'),
          padding: '15px', 
          borderRadius: '4px', 
          border: '1px solid currentColor',
          marginBottom: '20px',
          fontWeight: 'bold',
          display: 'flex',
          alignItems: 'center',
          gap: '10px'
        }}>
          {scanResult.overall.level === 'CRITICAL' ? <ShieldAlert /> : <CheckCircle />}
          <span>OVERALL VERDICT: {scanResult.overall.level}</span>
        </div>

        {/* Evidence Images */}
        <h3 style={{ fontSize: '16px', borderBottom: '1px solid #ccc', paddingBottom: '5px', marginBottom: '15px' }}>Visual Evidence</h3>
        <div style={{ display: 'flex', gap: '20px', marginBottom: '30px' }}>
          <div style={{ flex: 1 }}>
            <p style={{ fontSize: '12px', fontWeight: 'bold', marginBottom: '5px' }}>Raw X-Ray Capture</p>
            <img src={scanResult.original_image} alt="Raw" style={{ width: '100%', border: '1px solid #ccc' }} />
          </div>
          <div style={{ flex: 1 }}>
            <p style={{ fontSize: '12px', fontWeight: 'bold', marginBottom: '5px' }}>AI Bounding Box Analysis</p>
            <img src={scanResult.annotated_image} alt="Annotated" style={{ width: '100%', border: '1px solid #ccc' }} />
          </div>
        </div>

        {/* Deep Properties / Details */}
        <h3 style={{ fontSize: '16px', borderBottom: '1px solid #ccc', paddingBottom: '5px', marginBottom: '15px' }}>Extracted Threat Properties (Selected Item)</h3>
        {box ? (
          <div>
            <p style={{ margin: '0 0 10px 0' }}><strong>Material Classification:</strong> {box.material.toUpperCase()}</p>
            <p style={{ margin: '0 0 15px 0' }}><strong>AI Explanation:</strong> {box.explanation}</p>
            
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px' }}>
              <thead>
                <tr style={{ background: '#f4f4f4' }}>
                  <th style={{ border: '1px solid #ccc', padding: '8px', textAlign: 'left' }}>Property</th>
                  <th style={{ border: '1px solid #ccc', padding: '8px', textAlign: 'left' }}>Value</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(box.properties || {}).map(([key, val]) => (
                  <tr key={key}>
                    <td style={{ border: '1px solid #ccc', padding: '8px' }}>{key.replace(/_/g, ' ').toUpperCase()}</td>
                    <td style={{ border: '1px solid #ccc', padding: '8px' }}>{typeof val === 'number' ? val.toFixed(4) : val}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p>No objects flagged.</p>
        )}

        {/* Signatures */}
        <div style={{ marginTop: '50px', display: 'flex', justifyContent: 'space-between' }}>
          <div style={{ width: '45%' }}>
            <div style={{ borderBottom: '1px solid #111', height: '30px' }}></div>
            <p style={{ fontSize: '12px', marginTop: '5px' }}>Inspecting Officer Signature</p>
          </div>
          <div style={{ width: '45%' }}>
            <div style={{ borderBottom: '1px solid #111', height: '30px' }}></div>
            <p style={{ fontSize: '12px', marginTop: '5px' }}>Shift Supervisor Signature</p>
          </div>
        </div>
      </div>
    );
  }
}

// --- Main Scanner Tab UI ---
export default function ScannerTab({
  uploadedFile, setUploadedFile, uploadPreview, setUploadPreview,
  manualScanResult, setManualScanResult, scanLoading, setScanLoading,
  hoveredBoxId, setHoveredBoxId, selectedBoxId, setSelectedBoxId, analyzerImgRef
}) {
  
  const componentRef = useRef();
  const handlePrint = useReactToPrint({
    content: () => componentRef.current,
    documentTitle: 'VIT_Security_Incident_Report',
  });

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    setUploadedFile(file);
    setUploadPreview(URL.createObjectURL(file));
    setManualScanResult(null);
    setSelectedBoxId(null);
    
    // Automatically trigger scan
    setScanLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    
    fetch('/api/scan', {
      method: 'POST',
      body: formData
    })
    .then(res => res.json())
    .then(data => {
      setManualScanResult(data);
      setScanLoading(false);
      if (data.bboxes.length > 0) {
        setSelectedBoxId(0);
      }
    })
    .catch(err => {
      console.error(err);
      setScanLoading(false);
    });
  };

  // Format radar chart data from the selected box properties
  let radarData = [];
  if (manualScanResult && manualScanResult.bboxes[selectedBoxId]) {
    const props = manualScanResult.bboxes[selectedBoxId].properties;
    radarData = [
      { subject: 'Density', A: props.density_level * 100 },
      { subject: 'Sharpness', A: props.edge_sharpness * 100 },
      { subject: 'Symmetry', A: props.symmetry_score * 100 },
      { subject: 'Absorption', A: props.avg_absorption_intensity * 100 },
      { subject: 'Curvature', A: props.curvature_index * 100 },
      { subject: 'Volume', A: props.approximate_volume * 100 }
    ];
  }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 400px', gap: '1.5rem', height: '100%' }}>
      {/* --- Left Column: Viewer Area --- */}
      <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
        
        {/* Header Action Bar */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h2 style={{ fontSize: '1.2rem', fontWeight: '800', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <BadgeInfo size={20} color="var(--accent-cyan)" /> Deep Property Analyzer
          </h2>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            {manualScanResult && (
              <button onClick={handlePrint} className="btn-secondary" style={{ borderColor: 'var(--color-warning)', color: 'var(--color-warning)' }}>
                <FileText size={16} /> Generate PDF Report
              </button>
            )}
            <label htmlFor="upload-btn" className="btn-primary" style={{ cursor: 'pointer', display: 'inline-flex' }}>
              <Upload size={16} /> Upload X-Ray
            </label>
            <input id="upload-btn" type="file" accept="image/jpeg, image/png" style={{ display: 'none' }} onChange={handleFileUpload} />
          </div>
        </div>
        
        {/* Main Image Viewer */}
        <div style={{ position: 'relative', flex: 1, background: '#0e111a', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
          {scanLoading ? (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
              <RefreshCw size={32} className="animate-spin" color="var(--accent-cyan)" />
              <span style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Extracting Properties via Model 1...</span>
            </div>
          ) : manualScanResult ? (
            <div style={{ position: 'relative', maxWidth: '100%', maxHeight: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <img 
                ref={analyzerImgRef}
                src={manualScanResult.annotated_image} 
                alt="Analyzed Image" 
                style={{ maxWidth: '100%', maxHeight: '500px', objectFit: 'contain', display: 'block' }}
              />
              
              {/* Interactive Bounding Boxes Overlay */}
              {manualScanResult.bboxes.map((box, index) => {
                const imgEl = analyzerImgRef.current;
                if (!imgEl) return null;
                
                // Need to compute relative coordinates
                const origW = 600; // Hardcoded or extracted from image ideally
                const origH = 400; 
                
                const leftPct = (box.x1 / origW) * 100;
                const topPct = (box.y1 / origH) * 100;
                const widthPct = ((box.x2 - box.x1) / origW) * 100;
                const heightPct = ((box.y2 - box.y1) / origH) * 100;
                
                const isSelected = selectedBoxId === index;
                const isHovered = hoveredBoxId === index;
                
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
                      border: isSelected ? `3px solid ${strokeColor}` : (isHovered ? `2px dashed ${strokeColor}` : '2px solid transparent'),
                      background: isSelected ? 'rgba(56, 189, 248, 0.05)' : (isHovered ? 'rgba(255,255,255,0.05)' : 'transparent'),
                      cursor: 'pointer',
                      outline: 'none',
                      padding: 0
                    }}
                    onMouseEnter={() => setHoveredBoxId(index)}
                    onMouseLeave={() => setHoveredBoxId(null)}
                    onClick={() => setSelectedBoxId(index)}
                  />
                );
              })}
            </div>
          ) : uploadPreview ? (
            <img src={uploadPreview} alt="Preview" style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }} />
          ) : (
            <div style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
              <Upload size={48} style={{ opacity: 0.5, margin: '0 auto 1rem auto' }} />
              <p>Upload a suspicious luggage image</p>
              <p style={{ fontSize: '0.8rem' }}>Extracts 11-dimensional deep properties dynamically.</p>
            </div>
          )}
        </div>

        {/* Security Metadata Panel (Fills out the screen) */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', background: 'rgba(255,255,255,0.02)', padding: '1rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
          <div>
            <span style={{ fontSize: '0.65rem', textTransform: 'uppercase', color: 'var(--text-muted)', display: 'block' }}>Operator ID</span>
            <span style={{ fontSize: '0.85rem', fontWeight: 'bold', color: 'var(--text-main)' }}>SEC-A942</span>
          </div>
          <div>
            <span style={{ fontSize: '0.65rem', textTransform: 'uppercase', color: 'var(--text-muted)', display: 'block' }}>Terminal / Checkpoint</span>
            <span style={{ fontSize: '0.85rem', fontWeight: 'bold', color: 'var(--text-main)' }}>T4 - Alpha Gate</span>
          </div>
          <div>
            <span style={{ fontSize: '0.65rem', textTransform: 'uppercase', color: 'var(--text-muted)', display: 'block' }}>Scan Timestamp</span>
            <span style={{ fontSize: '0.85rem', fontWeight: 'bold', color: 'var(--text-main)' }}>{new Date().toLocaleTimeString()}</span>
          </div>
          <div>
            <span style={{ fontSize: '0.65rem', textTransform: 'uppercase', color: 'var(--text-muted)', display: 'block' }}>System Status</span>
            <span style={{ fontSize: '0.85rem', fontWeight: 'bold', color: 'var(--color-safe)' }}>CALIBRATED</span>
          </div>
        </div>

      </div>

      {/* --- Right Column: Analytics & Extracted Properties Panel --- */}
      <div className="glass-panel" style={{ padding: '1.25rem', display: 'flex', flexDirection: 'column' }}>
        
        <h3 style={{ fontSize: '0.85rem', fontWeight: '800', textTransform: 'uppercase', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>AI Threat Evaluation</h3>
        
        {manualScanResult && manualScanResult.bboxes[selectedBoxId] ? (
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflowY: 'auto' }}>
            
            {/* Probability Gauge (Visual enhancement) */}
            <div style={{ background: 'rgba(255,255,255,0.02)', borderRadius: '8px', padding: '1rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '1rem', border: `1px solid ${manualScanResult.bboxes[selectedBoxId].threat_level === 'CRITICAL' ? 'var(--color-critical)' : 'rgba(255,255,255,0.05)'}` }}>
              <div style={{ width: '60px', height: '60px', borderRadius: '50%', background: manualScanResult.bboxes[selectedBoxId].threat_level === 'CRITICAL' ? 'rgba(239, 68, 68, 0.2)' : 'rgba(16, 185, 129, 0.2)', display: 'flex', justifyContent: 'center', alignItems: 'center', border: `2px solid ${manualScanResult.bboxes[selectedBoxId].threat_level === 'CRITICAL' ? 'var(--color-critical)' : 'var(--color-safe)'}` }}>
                <span style={{ fontSize: '1.2rem', fontWeight: '900', color: manualScanResult.bboxes[selectedBoxId].threat_level === 'CRITICAL' ? 'var(--color-critical)' : 'var(--color-safe)' }}>
                  {Math.round((manualScanResult.bboxes[selectedBoxId].confidence || 0.95) * 100)}%
                </span>
              </div>
              <div>
                <span style={{ display: 'block', fontSize: '0.7rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>Classification</span>
                <span style={{ display: 'block', fontSize: '1.1rem', fontWeight: '800', color: manualScanResult.bboxes[selectedBoxId].threat_level === 'CRITICAL' ? 'var(--color-critical)' : 'var(--color-safe)' }}>
                  {manualScanResult.bboxes[selectedBoxId].threat_level} THREAT
                </span>
              </div>
            </div>

            {/* AI Explanation */}
            <div style={{ marginBottom: '1.25rem' }}>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-main)', lineHeight: '1.4' }}>
                <strong style={{ color: 'var(--accent-cyan)' }}>AI Verdict: </strong> 
                {manualScanResult.bboxes[selectedBoxId].explanation}
              </p>
            </div>

            {/* Radar Chart for Properties (Visual Enhancement) */}
            <div style={{ height: '220px', width: '100%', marginBottom: '1rem' }}>
              <ResponsiveContainer>
                <RadarChart cx="50%" cy="50%" outerRadius="70%" data={radarData}>
                  <PolarGrid stroke="rgba(255,255,255,0.1)" />
                  <PolarAngleAxis dataKey="subject" tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} />
                  <Radar name="Property Vector" dataKey="A" stroke="var(--accent-cyan)" fill="var(--accent-cyan)" fillOpacity={0.4} />
                  <Tooltip contentStyle={{ background: '#070a13', border: '1px solid rgba(255,255,255,0.1)' }} />
                </RadarChart>
              </ResponsiveContainer>
            </div>

            {/* Detailed Properties List */}
            <h4 style={{ fontSize: '0.7rem', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Raw Multi-Dimensional Vector</h4>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', paddingBottom: '1rem' }}>
              {Object.entries(manualScanResult.bboxes[selectedBoxId].properties || {}).map(([key, value]) => {
                const formattedKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                const formattedVal = typeof value === 'number' ? value.toFixed(4) : value;
                return (
                  <div key={key} style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px dashed rgba(255,255,255,0.05)', paddingBottom: '0.2rem' }}>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>{formattedKey}</span>
                    <span style={{ fontSize: '0.8rem', fontWeight: '600', color: 'var(--text-main)' }}>{formattedVal}</span>
                  </div>
                );
              })}
            </div>
            
          </div>
        ) : (
          <div style={{ flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'center', color: 'var(--text-muted)', fontSize: '0.8rem', textAlign: 'center' }}>
            Select a bounding box on the image to view the deep property vector analysis.
          </div>
        )}
      </div>

      {/* Hidden Print Component */}
      <div style={{ display: 'none' }}>
        <IncidentReportTemplate 
          ref={componentRef} 
          scanResult={manualScanResult} 
          uploadPreview={uploadPreview}
          selectedBoxId={selectedBoxId}
        />
      </div>
      
    </div>
  );
}
