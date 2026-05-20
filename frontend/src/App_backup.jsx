import React, { useState, useEffect, useRef } from 'react';
import { 
  ShieldAlert, ShieldCheck, Activity, Workflow as BeltIcon, 
  Upload, Layers, BarChart3, Settings, Play, Pause, AlertTriangle, 
  RefreshCw, PlusCircle, Compass, HelpCircle, User, Info, CheckCircle2 
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Cell } from 'recharts';

export default function App() {
  const [activeTab, setActiveTab] = useState('conveyor');
  const [modelStatus, setModelStatus] = useState(null);
  
  // Conveyor Simulator State
  const [isPlaying, setIsPlaying] = useState(true);
  const [luggageQueue, setLuggageQueue] = useState([]);
  const [selectedBag, setSelectedBag] = useState(null);
  const [scanResult, setScanResult] = useState(null);
  const [scanLoading, setScanLoading] = useState(false);
  const [stats, setStats] = useState({
    total_scanned: 1284,
    threats_detected: 47,
    false_alarms: 18,
    bypass_rate: 96.3,
    material_distribution: [
      { name: 'Organic (Orange)', value: 68.2 },
      { name: 'Metallic (Blue)', value: 18.5 },
      { name: 'Mixed (Green)', value: 11.1 },
      { name: 'Opaque (Black)', value: 2.2 }
    ]
  });

  // Manual Analyzer State
  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploadPreview, setUploadPreview] = useState(null);
  const [manualScanResult, setManualScanResult] = useState(null);
  const [hoveredBoxId, setHoveredBoxId] = useState(null);
  const [selectedBoxId, setSelectedBoxId] = useState(null);

  // TIP Sandbox State
  const [tipBgType, setTipBgType] = useState('safe_luggage');
  const [tipThreatType, setTipThreatType] = useState('knife');
  const [tipScale, setTipScale] = useState(1.0);
  const [tipAngle, setTipAngle] = useState(0);
  const [tipPosX, setTipPosX] = useState(50);
  const [tipPosY, setTipPosY] = useState(50);
  const [tipThickness, setTipThickness] = useState(1.0);
  
  const [tipResult, setTipResult] = useState(null);
  const [tipLoading, setTipLoading] = useState(false);
  const [customBgFile, setCustomBgFile] = useState(null);
  const [customFgFile, setCustomFgFile] = useState(null);
  const [tipHoveredBoxId, setTipHoveredBoxId] = useState(null);
  const [tipSelectedBoxId, setTipSelectedBoxId] = useState(null);

  // Analytics State
  const [analyticsData, setAnalyticsData] = useState(null);

  // Refs for tracking viewport sizing in canvas coordinate projection
  const analyzerImgRef = useRef(null);
  const tipImgRef = useRef(null);

  // Pre-configured simulation threats
  const threatPresets = [
    { id: 'knife', name: 'Steel Hunting Knife', material: 'metallic', complexity: 'High' },
    { id: 'scissors', name: 'Medical Scissors', material: 'metallic', complexity: 'Medium' },
    { id: 'handgun', name: 'Revolver Frame', material: 'metallic', complexity: 'Critical' },
    { id: 'aerosol', name: 'Deodorant Aerosol Can', material: 'mixed', complexity: 'Low' },
    { id: 'shield_block', name: 'Lead Block', material: 'opaque', complexity: 'Critical' }
  ];

  // Load Initial Status
  useEffect(() => {
    fetch('/api/model-status')
      .then(res => res.json())
      .then(data => setModelStatus(data))
      .catch(err => console.error("Error loading model status: ", err));

    fetch('/api/stats')
      .then(res => res.json())
      .then(data => setStats(data))
      .catch(err => console.error("Error loading stats: ", err));

    fetch('/api/mock-analytics')
      .then(res => res.json())
      .then(data => setAnalyticsData(data))
      .catch(err => console.error("Error loading analytics: ", err));
  }, []);

  // Fetch Luggage Feed for conveyor belt
  useEffect(() => {
    fetch('/api/feed')
      .then(res => res.json())
      .then(data => {
        setLuggageQueue(data);
        if (data.length > 0) {
          handleSelectBag(data[0]);
        }
      })
      .catch(err => console.error("Error loading luggage feed: ", err));
  }, []);

  // Simulator Timer: Auto moves bags on belt when playing
  useEffect(() => {
    let interval = null;
    if (isPlaying && luggageQueue.length > 0) {
      interval = setInterval(() => {
        // Rotate queue to simulate continuous feed
        setLuggageQueue(prev => {
          const rotated = [...prev.slice(1), prev[0]];
          // Keep selection synced to first item automatically
          handleSelectBag(rotated[0]);
          return rotated;
        });
      }, 8000);
    }
    return () => clearInterval(interval);
  }, [isPlaying, luggageQueue]);

  const handleSelectBag = (bag) => {
    setSelectedBag(bag);
    setScanLoading(true);
    
    // Fetch programmatic image from backend
    fetch(`/api/feed/image/${bag.mock_type}`)
      .then(res => res.json())
      .then(imgData => {
        // Now convert B64 to file blob and run /api/scan
        const base64Content = imgData.image.split(',')[1];
        const blob = b64toBlob(base64Content, 'image/jpeg');
        const file = new File([blob], `${bag.mock_type}.jpg`, { type: 'image/jpeg' });
        
        const formData = new FormData();
        formData.append('file', file);
        
        return fetch('/api/scan', {
          method: 'POST',
          body: formData
        });
      })
      .then(res => res.json())
      .then(scanData => {
        setScanResult(scanData);
        setScanLoading(false);
        setSelectedBoxId(scanData.bboxes.length > 0 ? 0 : null);
      })
      .catch(err => {
        console.error("Error scanning bag:", err);
        setScanLoading(false);
      });
  };

  // Convert Base64 string to Blob helper
  const b64toBlob = (b64Data, contentType = '', sliceSize = 512) => {
    const byteCharacters = atob(b64Data);
    const byteArrays = [];
    for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {
      const slice = byteCharacters.slice(offset, offset + sliceSize);
      const byteNumbers = new Array(slice.length);
      for (let i = 0; i < slice.length; i++) {
        byteNumbers[i] = slice.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      byteArrays.push(byteArray);
    }
    return new Blob(byteArrays, { type: contentType });
  };

  // Handle Manual Upload Analyze
  const handleManualUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setUploadedFile(file);
    setUploadPreview(URL.createObjectURL(file));
    
    const formData = new FormData();
    formData.append('file', file);
    
    setScanLoading(true);
    fetch('/api/scan', {
      method: 'POST',
      body: formData
    })
      .then(res => res.json())
      .then(data => {
        setManualScanResult(data);
        setScanLoading(false);
        setSelectedBoxId(data.bboxes.length > 0 ? 0 : null);
      })
      .catch(err => {
        console.error("Manual scan failed:", err);
        setScanLoading(false);
      });
  };

  // Execute TIP Projection Simulation
  const handleRunTIP = async () => {
    setTipLoading(true);
    try {
      let bgBlob, fgBlob;
      
      // Get background suitcase blob
      if (customBgFile) {
        bgBlob = customBgFile;
      } else {
        const bgImgRes = await fetch(`/api/feed/image/${tipBgType}`);
        const bgImgJson = await bgImgRes.json();
        const base64Bg = bgImgJson.image.split(',')[1];
        bgBlob = b64toBlob(base64Bg, 'image/jpeg');
      }

      // Get foreground threat blob (draw programmatic threat patterns or use preset file)
      if (customFgFile) {
        fgBlob = customFgFile;
      } else {
        // Draw a programmatic BGR threat matrix inside browser as a proxy or use standard file uploads.
        // For a seamless sandbox experience, let's request it from the program generator or draw on canvas.
        // We will generate the threat programmatically inside backend by sending the key.
        // We can just construct a small temporary 200x200 canvas in browser to draw the threat BGR shape!
        // This is 100% locally self-contained and extremely beautiful!
        fgBlob = await generateLocalCanvasThreatBlob(tipThreatType);
      }

      const formData = new FormData();
      formData.append('bg_file', bgBlob, 'bg.jpg');
      formData.append('fg_file', fgBlob, 'fg.jpg');
      formData.append('scale', tipScale);
      formData.append('angle', tipAngle);
      formData.append('pos_x_pct', tipPosX);
      formData.append('pos_y_pct', tipPosY);
      formData.append('thickness', tipThickness);

      const res = await fetch('/api/tip', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      setTipResult(data);
      setTipSelectedBoxId(data.bboxes.length > 0 ? 0 : null);
    } catch (err) {
      console.error("TIP projection failed:", err);
    } finally {
      setTipLoading(false);
    }
  };

  // Browser-side programmatically drew shape representation for TIP inputs
  const generateLocalCanvasThreatBlob = (type) => {
    return new Promise((resolve) => {
      const canvas = document.createElement('canvas');
      canvas.width = 300;
      canvas.height = 300;
      const ctx = canvas.getContext('2d');
      
      // Draw background white
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, 300, 300);
      
      ctx.lineWidth = 10;
      ctx.lineCap = 'round';
      
      if (type === 'knife') {
        // Draw knife blade (gray metallic) and handle (mixed)
        ctx.fillStyle = '#5555ff'; // Metallic color in dual-energy representation
        ctx.strokeStyle = '#2222aa';
        ctx.beginPath();
        ctx.moveTo(80, 150);
        ctx.lineTo(240, 130);
        ctx.lineTo(220, 170);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        
        // Handle
        ctx.fillStyle = '#33aa33'; // Mixed green
        ctx.fillRect(30, 140, 50, 20);
      } 
      else if (type === 'scissors') {
        // Dual loops + intersecting blades
        ctx.strokeStyle = '#2222ff';
        ctx.lineWidth = 14;
        
        // Loop 1
        ctx.beginPath();
        ctx.arc(80, 120, 25, 0, Math.PI * 2);
        ctx.stroke();
        // Loop 2
        ctx.beginPath();
        ctx.arc(80, 180, 25, 0, Math.PI * 2);
        ctx.stroke();
        
        // Blades
        ctx.beginPath();
        ctx.moveTo(105, 130);
        ctx.lineTo(240, 170);
        ctx.moveTo(105, 170);
        ctx.lineTo(240, 130);
        ctx.stroke();
      } 
      else if (type === 'handgun') {
        // Barrel + grip + trigger block in metallic blue/black
        ctx.fillStyle = '#1111aa';
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 4;
        
        // Barrel
        ctx.fillRect(100, 100, 140, 35);
        // Grip
        ctx.beginPath();
        ctx.moveTo(100, 125);
        ctx.lineTo(70, 210);
        ctx.lineTo(110, 210);
        ctx.lineTo(135, 135);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        
        // Trigger guard loop
        ctx.beginPath();
        ctx.arc(140, 150, 15, 0, Math.PI * 2);
        ctx.stroke();
      } 
      else if (type === 'aerosol') {
        // Cylindrical canister
        ctx.fillStyle = '#10c010'; // Mixed green
        ctx.strokeStyle = '#1030cc'; // Metal casing blue outline
        ctx.lineWidth = 10;
        
        ctx.fillRect(100, 80, 100, 160);
        ctx.strokeRect(100, 80, 100, 160);
        
        // Spray dome
        ctx.fillStyle = '#e87010'; // Organic orange spray valve
        ctx.beginPath();
        ctx.arc(150, 80, 20, Math.PI, 0);
        ctx.fill();
      } 
      else {
        // Shielded block (dense black)
        ctx.fillStyle = '#0f0f15';
        ctx.fillRect(70, 70, 160, 160);
        ctx.strokeStyle = '#2222ff';
        ctx.strokeRect(70, 70, 160, 160);
      }
      
      canvas.toBlob((blob) => {
        resolve(blob);
      }, 'image/jpeg', 0.95);
    });
  };

  const currentResult = activeTab === 'conveyor' ? scanResult : manualScanResult;
  const currentBox = currentResult?.bboxes?.[selectedBoxId];
  const currentProp = currentResult?.properties?.[selectedBoxId];

  return (
    <div className="dashboard-container">
      {/* 1. Header Bar */}
      <header className="glass-panel-glow" style={{ margin: '1rem', padding: '1rem 1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderRadius: 'var(--radius-md)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div style={{ background: 'linear-gradient(135deg, #f97316 0%, #3b82f6 100%)', width: '40px', height: '40px', borderRadius: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 0 15px rgba(59, 130, 246, 0.4)' }}>
            <Layers size={22} color="#fff" />
          </div>
          <div>
            <h1 style={{ fontSize: '1.25rem', fontWeight: '800', letterSpacing: '0.5px' }}>
              SECURE<span style={{ color: 'var(--accent-cyan)' }}>SCAN</span> <span style={{ fontSize: '0.85rem', padding: '2px 8px', borderRadius: '5px', background: 'rgba(56,189,248,0.1)', color: 'var(--accent-cyan)', fontWeight: '600', marginLeft: '5px' }}>DUAL-MODEL V2</span>
            </h1>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Airport Luggage Attenuation & Property Profiler</p>
          </div>
        </div>

        {/* Engine Pipeline Status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'rgba(255,255,255,0.03)', padding: '0.4rem 0.8rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
            <Activity size={16} color={modelStatus?.is_dl_mode ? "var(--accent-cyan)" : "var(--color-warning)"} />
            <span style={{ fontSize: '0.8rem', fontWeight: '600' }}>
              Engine: <span style={{ color: modelStatus?.is_dl_mode ? "var(--accent-cyan)" : "var(--color-warning)" }}>
                {modelStatus?.is_dl_mode ? "Deep Learning Active" : "Contour Fallback (Simulation Mode)"}
              </span>
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'rgba(255,255,255,0.03)', padding: '0.4rem 0.8rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
            <Info size={16} color="var(--text-secondary)" />
            <span style={{ fontSize: '0.8rem', fontWeight: '600' }}>
              Device: <span style={{ color: '#fff' }}>{modelStatus?.device || "CPU"}</span>
            </span>
          </div>
        </div>
      </header>

      <main className="content-area">
        {/* 2. Primary Tabs */}
        <nav className="tabs-nav">
          <button 
            id="tab-conveyor"
            className={`tab-btn ${activeTab === 'conveyor' ? 'active' : ''}`}
            onClick={() => setActiveTab('conveyor')}
          >
            <BeltIcon size={18} /> Real-time conveyor Belt
          </button>
          <button 
            id="tab-scanner"
            className={`tab-btn ${activeTab === 'scanner' ? 'active' : ''}`}
            onClick={() => setActiveTab('scanner')}
          >
            <Upload size={18} /> Manual Scanner Upload
          </button>
          <button 
            id="tab-sandbox"
            className={`tab-btn ${activeTab === 'sandbox' ? 'active' : ''}`}
            onClick={() => { setActiveTab('sandbox'); handleRunTIP(); }}
          >
            <PlusCircle size={18} /> Beer-Lambert TIP Sandbox
          </button>
          <button 
            id="tab-analytics"
            className={`tab-btn ${activeTab === 'analytics' ? 'active' : ''}`}
            onClick={() => setActiveTab('analytics')}
          >
            <BarChart3 size={18} /> Model Performance & Analytics
          </button>
        </nav>

        {/* 3. Tab Contents */}
        
        {/* VIEW A: REAL-TIME CONVEYOR SIMULATOR */}
        {activeTab === 'conveyor' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            {/* Belt Simulation Panel */}
            <section className="glass-panel" style={{ padding: '1.25rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: '700', textTransform: 'uppercase', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <BeltIcon size={18} color="var(--accent-cyan)" /> Live Operator Conveyor Belt
                </h3>
                <button 
                  id="play-pause-btn"
                  className="btn-secondary" 
                  onClick={() => setIsPlaying(!isPlaying)}
                  style={{ padding: '0.4rem 0.9rem', fontSize: '0.85rem' }}
                >
                  {isPlaying ? <Pause size={14} /> : <Play size={14} />} {isPlaying ? "Pause Belt" : "Start Belt"}
                </button>
              </div>

              {/* Scrolling Conveyor Belt container */}
              <div className={`bag-queue-belt ${isPlaying ? 'conveyor-animation' : ''}`}>
                {luggageQueue.map((bag, i) => (
                  <div 
                    id={`queue-item-${bag.id}`}
                    key={bag.id}
                    className={`queue-item ${selectedBag?.id === bag.id ? 'active' : ''} ${bag.expected_level}`}
                    onClick={() => { setSelectedBag(bag); handleSelectBag(bag); }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: '0.65rem', fontWeight: '600', background: 'rgba(255,255,255,0.06)', padding: '2px 6px', borderRadius: '4px' }}>
                        BAG #{1000 + bag.id}
                      </span>
                      {bag.expected_level === 'CRITICAL' && <AlertTriangle size={14} color="var(--color-critical)" />}
                    </div>
                    <span style={{ fontSize: '0.8rem', fontWeight: '700', color: '#fff', textOverflow: 'ellipsis', whiteSpace: 'nowrap', overflow: 'hidden' }}>
                      {bag.name}
                    </span>
                    <span style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', display: 'flex', justifyContent: 'space-between' }}>
                      <span>Risk Probability:</span>
                      <span style={{ color: bag.expected_level === 'CRITICAL' ? 'var(--color-critical)' : 'var(--text-secondary)', fontWeight: '700' }}>
                        {int(bag.threat_probability * 100)}%
                      </span>
                    </span>
                  </div>
                ))}
              </div>
            </section>

            {/* Scanning and Bounding Box View */}
            <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr', gap: '1.5rem' }}>
              {/* Suitcase Screen Viewer */}
              <div className="glass-panel pulse-critical" style={{ padding: '1.25rem', overflow: 'hidden', position: 'relative' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                  <span style={{ fontSize: '0.8rem', fontWeight: '700', textTransform: 'uppercase', color: 'var(--text-secondary)' }}>
                    Active Scan Inspection: <span style={{ color: '#fff' }}>BAG #{1000 + (selectedBag?.id || 0)}</span>
                  </span>
                  <div className="mat-chip" style={{ background: scanResult?.overall.level === 'CRITICAL' ? 'var(--color-critical-glow)' : 'rgba(255,255,255,0.03)', color: scanResult?.overall.level === 'CRITICAL' ? 'var(--color-critical)' : 'var(--text-secondary)', border: '1px solid currentColor' }}>
                    {scanResult?.overall.level || "UNKNOWN"}
                  </div>
                </div>

                <div style={{ position: 'relative', width: '100%', height: '360px', background: '#0e111a', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  {scanLoading ? (
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem' }}>
                      <RefreshCw size={24} className="animate-spin" color="var(--accent-cyan)" />
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Extracting 11D attenuations...</span>
                    </div>
                  ) : (
                    <div style={{ position: 'relative', maxWidth: '100%', maxHeight: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      {/* Original programmatic image loaded */}
                      <img 
                        ref={analyzerImgRef}
                        src={scanResult?.original_image} 
                        alt="Suitcase Scan Preview" 
                        style={{ maxWidth: '100%', maxHeight: '340px', objectFit: 'contain', display: 'block' }}
                      />
                      
                      {/* Bounding Box Absolute Overlays on Suitcase */}
                      {scanResult?.bboxes.map((box, index) => {
                        // Project absolute pixel bounding boxes back to responsive percentages
                        // Suitcases images from API feed are hardcoded at 600x400
                        const origW = 600;
                        const origH = 400;
                        
                        const leftPct = (box.x1 / origW) * 100;
                        const topPct = (box.y1 / origH) * 100;
                        const widthPct = ((box.x2 - box.x1) / origW) * 100;
                        const heightPct = ((box.y2 - box.y1) / origH) * 100;
                        
                        // Color styling matching materials
                        const isSelected = selectedBoxId === index;
                        const isHovered = hoveredBoxId === index;
                        
                        let strokeColor = 'var(--color-mixed)';
                        if (box.material === 'organic') strokeColor = 'var(--color-organic)';
                        if (box.material === 'metallic') strokeColor = 'var(--color-metallic)';
                        if (box.material === 'opaque') strokeColor = 'var(--color-opaque)';
                        if (box.threat_level === 'CRITICAL') strokeColor = 'var(--color-critical)';

                        return (
                          <button
                            id={`bbox-${index}`}
                            key={box.id}
                            style={{
                              position: 'absolute',
                              left: `${leftPct}%`,
                              top: `${topPct}%`,
                              width: `${widthPct}%`,
                              height: `${heightPct}%`,
                              border: isSelected ? `3.5px solid ${strokeColor}` : (isHovered ? `2.5px dashed ${strokeColor}` : `1.5px solid ${strokeColor}`),
                              background: isSelected ? 'rgba(56, 189, 248, 0.04)' : (isHovered ? 'rgba(255, 255, 255, 0.02)' : 'transparent'),
                              boxShadow: isSelected ? `0 0 12px ${strokeColor}` : 'none',
                              cursor: 'pointer',
                              padding: 0,
                              outline: 'none'
                            }}
                            onMouseEnter={() => setHoveredBoxId(index)}
                            onMouseLeave={() => setHoveredBoxId(null)}
                            onClick={() => setSelectedBoxId(index)}
                          />
                        );
                      })}
                    </div>
                  )}
                </div>
                
                {/* Decision Panel */}
                <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }}>
                  <button 
                    id="approve-bag-btn"
                    className="btn-primary" 
                    style={{ flex: 1, background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)', display: 'flex', justifyContent: 'center' }}
                    onClick={() => {
                      alert(`Luggage Approved! Belt resuming.`);
                      setIsPlaying(true);
                    }}
                  >
                    <ShieldCheck size={18} /> Automatically Approve Luggage
                  </button>
                  <button 
                    id="reject-bag-btn"
                    className="btn-primary" 
                    style={{ flex: 1, background: 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)', display: 'flex', justifyContent: 'center' }}
                    onClick={() => {
                      alert(`ALERT: Luggage Locked for secondary manual search!`);
                      setIsPlaying(true);
                    }}
                  >
                    <ShieldAlert size={18} /> Trigger Alarm & Reject
                  </button>
                </div>
              </div>

              {/* Bounding Box Information & Model 2 Explanations */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {/* Overall Assessment Card */}
                <div className="glass-panel" style={{ padding: '1rem', borderLeft: scanResult?.overall.level === 'CRITICAL' ? '4px solid var(--color-critical)' : '1px solid var(--border-color)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem' }}>
                    <ShieldAlert size={20} color={scanResult?.overall.level === 'CRITICAL' ? 'var(--color-critical)' : (scanResult?.overall.level === 'WARNING' ? 'var(--color-warning)' : 'var(--color-safe)')} />
                    <span style={{ fontSize: '0.9rem', fontWeight: '800' }}>{scanResult?.overall.classification || "Safe suitcase"}</span>
                  </div>
                  <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
                    {scanResult?.overall.explanation || "All items clear."}
                  </p>
                </div>

                {/* Specific BBox Properties Card */}
                <div className="glass-panel" style={{ padding: '1.25rem', flex: 1, display: 'flex', flexDirection: 'column' }}>
                  {currentBox ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', height: '100%' }}>
                      {/* Name & Material */}
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid rgba(255,255,255,0.06)', paddingBottom: '0.5rem' }}>
                        <div>
                          <h4 style={{ fontSize: '0.9rem', fontWeight: '700' }}>{currentBox.label}</h4>
                          <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>Confidence: {int(currentBox.confidence * 100)}%</span>
                        </div>
                        <div className={`mat-chip ${currentBox.material}`}>
                          {currentBox.material}
                        </div>
                      </div>

                      {/* Model 2 Explanation */}
                      <div style={{ background: 'rgba(255,255,255,0.02)', padding: '0.6rem 0.8rem', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(255,255,255,0.04)' }}>
                        <h5 style={{ fontSize: '0.7rem', color: 'var(--accent-cyan)', textTransform: 'uppercase', fontWeight: '800', marginBottom: '0.2rem' }}>
                          Model 2 (RF Classification & Explanation)
                        </h5>
                        <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
                          {currentBox.explanation}
                        </p>
                      </div>

                      {/* Slider outputs representing the 11 dimensions */}
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', flex: 1, overflowY: 'auto' }}>
                        <h5 style={{ fontSize: '0.75rem', fontWeight: '800', textTransform: 'uppercase', color: 'var(--text-muted)' }}>11-Dimensional Property Values</h5>
                        {currentProp && Object.entries(currentProp).map(([name, value]) => {
                          const formattedValue = typeof value === 'number' ? value.toFixed(3) : value;
                          
                          // Display material name or rounded count
                          let displayVal = formattedValue;
                          if (name === 'material_category') {
                            displayVal = `${value.toFixed(0)} (${MATERIAL_NAMES[Math.round(value)] || "mixed"})`;
                          } else if (name === 'sharp_edge_count') {
                            displayVal = value.toFixed(0);
                          }

                          // Normalize values for property slider length indicators
                          let widthPct = 0;
                          if (name === 'material_category') {
                            widthPct = (value / 3) * 100;
                          } else if (name === 'sharp_edge_count') {
                            widthPct = (value / 20) * 100;
                          } else if (name === 'length_to_width_ratio') {
                            widthPct = (value / 10) * 100;
                          } else {
                            widthPct = value * 100;
                          }

                          return (
                            <div key={name} className="prop-bar-container">
                              <div className="prop-label-row">
                                <span style={{ textTransform: 'capitalize' }}>{name.replace(/_/g, ' ')}</span>
                                <span style={{ fontWeight: '700', color: '#fff' }}>{displayVal}</span>
                              </div>
                              <div className="prop-bar-outer">
                                <div 
                                  className="prop-bar-inner" 
                                  style={{ 
                                    width: `${widthPct}%`,
                                    background: name === 'density_level' ? 'linear-gradient(90deg, #3b82f6 0%, #ef4444 100%)' : 'var(--accent-cyan)'
                                  }}
                                />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', flex: 1, color: 'var(--text-muted)', gap: '0.5rem' }}>
                      <Info size={30} />
                      <span style={{ fontSize: '0.8rem' }}>Hover or click bounding boxes inside suitcase to inspect property vectors.</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* VIEW B: MANUAL SCAN UPLOADER */}
        {activeTab === 'scanner' && (
          <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr', gap: '1.5rem' }}>
            <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <h3 style={{ fontSize: '1rem', fontWeight: '800', textTransform: 'uppercase', marginBottom: '1rem', color: 'var(--accent-cyan)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Upload size={18} /> Upload Customs X-Ray Image
              </h3>

              {/* Upload Dropzone */}
              <label 
                className="dropzone-box"
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  height: '180px',
                  border: '2px dashed rgba(255,255,255,0.1)',
                  borderRadius: 'var(--radius-md)',
                  cursor: 'pointer',
                  background: 'rgba(255,255,255,0.01)',
                  transition: 'all var(--transition-fast)',
                  marginBottom: '1.5rem'
                }}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  const file = e.dataTransfer.files[0];
                  if (file) handleManualUpload({ target: { files: [file] } });
                }}
              >
                <input id="custom-image-uploader" type="file" accept="image/*" onChange={handleManualUpload} style={{ display: 'none' }} />
                <Upload size={32} color="var(--text-muted)" style={{ marginBottom: '0.5rem' }} />
                <span style={{ fontSize: '0.9rem', fontWeight: '600' }}>Drag Suitcase Image here or Click to Browse</span>
                <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>Supports standard JPEG, PNG or Dual-Energy BMPs</span>
              </label>

              {/* Image Preview and Interactive Boxes */}
              {uploadPreview && (
                <div style={{ position: 'relative', width: '100%', height: '360px', background: '#0e111a', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  {scanLoading ? (
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem' }}>
                      <RefreshCw size={24} className="animate-spin" color="var(--accent-cyan)" />
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Extracting custom material attributes...</span>
                    </div>
                  ) : (
                    <div style={{ position: 'relative', maxWidth: '100%', maxHeight: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <img 
                        ref={analyzerImgRef}
                        src={uploadPreview} 
                        alt="Suitcase Preview" 
                        style={{ maxWidth: '100%', maxHeight: '340px', objectFit: 'contain', display: 'block' }}
                      />
                      
                      {manualScanResult?.bboxes.map((box, index) => {
                        const imgEl = analyzerImgRef.current;
                        if (!imgEl) return null;
                        
                        const renderW = imgEl.clientWidth;
                        const renderH = imgEl.clientHeight;
                        
                        // Scale coordinates relative to original upload dimensions
                        const origW = imgEl.naturalWidth;
                        const origH = imgEl.naturalHeight;
                        
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
                            id={`manual-bbox-${index}`}
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
                            onMouseEnter={() => setHoveredBoxId(index)}
                            onMouseLeave={() => setHoveredBoxId(null)}
                            onClick={() => setSelectedBoxId(index)}
                          />
                        );
                      })}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Properties side view */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div className="glass-panel" style={{ padding: '1rem' }}>
                <h4 style={{ fontSize: '0.85rem', fontWeight: '800', textTransform: 'uppercase', marginBottom: '0.4rem', color: 'var(--text-muted)' }}>Scan Insights</h4>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
                  {manualScanResult?.overall.classification ? (
                    <strong>{manualScanResult.overall.classification}: {manualScanResult.overall.explanation}</strong>
                  ) : "Upload a suitcase to inspect its internal composition and check for concealed weapon indicators."}
                </p>
              </div>

              {/* Slider Outputs */}
              <div className="glass-panel" style={{ padding: '1.25rem', flex: 1, display: 'flex', flexDirection: 'column' }}>
                {manualScanResult?.bboxes[selectedBoxId] ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', height: '100%' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid rgba(255,255,255,0.06)', paddingBottom: '0.5rem' }}>
                      <div>
                        <h4 style={{ fontSize: '0.9rem', fontWeight: '700' }}>{manualScanResult.bboxes[selectedBoxId].label}</h4>
                        <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>Confidence: {int(manualScanResult.bboxes[selectedBoxId].confidence * 100)}%</span>
                      </div>
                      <div className={`mat-chip ${manualScanResult.bboxes[selectedBoxId].material}`}>
                        {manualScanResult.bboxes[selectedBoxId].material}
                      </div>
                    </div>

                    <div style={{ background: 'rgba(255,255,255,0.02)', padding: '0.6rem 0.8rem', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(255,255,255,0.04)' }}>
                      <h5 style={{ fontSize: '0.7rem', color: 'var(--accent-cyan)', textTransform: 'uppercase', fontWeight: '800', marginBottom: '0.2rem' }}>Model 2 Explanation</h5>
                      <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
                        {manualScanResult.bboxes[selectedBoxId].explanation}
                      </p>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', flex: 1, overflowY: 'auto' }}>
                      {Object.entries(manualScanResult.properties[selectedBoxId]).map(([name, value]) => {
                        let widthPct = name === 'material_category' ? (value / 3) * 100 : (name === 'sharp_edge_count' ? (value / 20) * 100 : (name === 'length_to_width_ratio' ? (value / 10) * 100 : value * 100));
                        return (
                          <div key={name} className="prop-bar-container">
                            <div className="prop-label-row">
                              <span style={{ textTransform: 'capitalize' }}>{name.replace(/_/g, ' ')}</span>
                              <span style={{ fontWeight: '700', color: '#fff' }}>
                                {name === 'material_category' ? `${value.toFixed(0)} (${MATERIAL_NAMES[Math.round(value)]})` : (name === 'sharp_edge_count' ? value.toFixed(0) : value.toFixed(3))}
                              </span>
                            </div>
                            <div className="prop-bar-outer">
                              <div className="prop-bar-inner" style={{ width: `${widthPct}%` }} />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ) : (
                  <div style={{ display: 'flex', flex1: 1, alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
                    No boxes selected. Upload a scan above.
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* VIEW C: BEER-LAMBERT TIP SANDBOX */}
        {activeTab === 'sandbox' && (
          <div style={{ display: 'grid', gridTemplateColumns: '0.75fr 1.25fr', gap: '1.5rem' }}>
            {/* Control Sidebar */}
            <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
              <h3 style={{ fontSize: '1rem', fontWeight: '800', textTransform: 'uppercase', color: 'var(--accent-cyan)', borderBottom: '1px solid rgba(255,255,255,0.06)', paddingBottom: '0.5rem' }}>
                Beer-Lambert Config
              </h3>

              {/* Preset selection */}
              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.4rem', fontWeight: '600' }}>Suitcase Suit (Background)</label>
                <select 
                  id="tip-bg-select"
                  className="btn-secondary" 
                  value={tipBgType} 
                  onChange={(e) => setTipBgType(e.target.value)} 
                  style={{ width: '100%', padding: '0.5rem' }}
                >
                  <option value="safe_luggage">Clear Suitcase (Clothing Presets)</option>
                  <option value="toolbox">Heavy Toolbox (Metallic Structure)</option>
                  <option value="aerosol_bag">Carry-on Pack ( toiletries outline)</option>
                </select>
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.4rem', fontWeight: '600' }}>Concealed Threat (Foreground)</label>
                <select 
                  id="tip-fg-select"
                  className="btn-secondary" 
                  value={tipThreatType} 
                  onChange={(e) => setTipThreatType(e.target.value)} 
                  style={{ width: '100%', padding: '0.5rem' }}
                >
                  {threatPresets.map(t => (
                    <option key={t.id} value={t.id}>{t.name} ({t.material.toUpperCase()})</option>
                  ))}
                </select>
              </div>

              {/* Slider Variables */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.2rem' }}>
                  <span>Scale Factor</span>
                  <span style={{ color: '#fff', fontWeight: '700' }}>{tipScale.toFixed(2)}x</span>
                </div>
                <input id="tip-scale-slider" type="range" min="0.3" max="2.0" step="0.05" className="slider-control" value={tipScale} onChange={(e) => setTipScale(parseFloat(e.target.value))} />
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.2rem' }}>
                  <span>Rotation (Angle)</span>
                  <span style={{ color: '#fff', fontWeight: '700' }}>{tipAngle}°</span>
                </div>
                <input id="tip-angle-slider" type="range" min="0" max="360" step="5" className="slider-control" value={tipAngle} onChange={(e) => setTipAngle(parseInt(e.target.value))} />
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.2rem' }}>
                  <span>Attenuation Thickness</span>
                  <span style={{ color: '#fff', fontWeight: '700' }}>{tipThickness.toFixed(2)}mm</span>
                </div>
                <input id="tip-thickness-slider" type="range" min="0.2" max="3.0" step="0.1" className="slider-control" value={tipThickness} onChange={(e) => setTipThickness(parseFloat(e.target.value))} />
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.2rem' }}>
                  <span>X Position</span>
                  <span style={{ color: '#fff', fontWeight: '700' }}>{tipPosX}%</span>
                </div>
                <input id="tip-posx-slider" type="range" min="10" max="90" step="1" className="slider-control" value={tipPosX} onChange={(e) => setTipPosX(parseInt(e.target.value))} />
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.2rem' }}>
                  <span>Y Position</span>
                  <span style={{ color: '#fff', fontWeight: '700' }}>{tipPosY}%</span>
                </div>
                <input id="tip-posy-slider" type="range" min="10" max="90" step="1" className="slider-control" value={tipPosY} onChange={(e) => setTipPosY(parseInt(e.target.value))} />
              </div>

              <button 
                id="run-tip-btn"
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
                            id={`tip-bbox-${index}`}
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
                      <span style={{ fontSize: '0.85rem' }}>Adjust sliders and click "Run Attenuated Projection" to generate Beer-Lambert synthesis.</span>
                    </div>
                  )}
                </div>
                
                {/* Resulting Properties Inspector */}
                {tipResult?.bboxes[tipSelectedBoxId] && (
                  <div style={{ background: 'rgba(255,255,255,0.02)', padding: '1rem', border: '1px solid rgba(255,255,255,0.05)', borderRadius: 'var(--radius-md)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                      <h4 style={{ fontSize: '0.85rem', fontWeight: '700' }}>
                        Detected Element: {tipResult.bboxes[tipSelectedBoxId].label}
                      </h4>
                      <span style={{ fontSize: '0.75rem', fontWeight: '800', color: 'var(--color-critical)' }}>
                        Risk Score: {int(tipResult.bboxes[tipSelectedBoxId].threat_score * 100)}%
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
        )}

        {/* VIEW D: MODEL PERFORMANCE & ANALYTICS */}
        {activeTab === 'analytics' && analyticsData && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            {/* Key Metrics */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1.25rem' }}>
              <div className="glass-panel" style={{ padding: '1rem 1.25rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', textTransform: 'uppercase', fontWeight: '600' }}>Model 1 mAP@50</span>
                  <h4 style={{ fontSize: '1.5rem', fontWeight: '800', color: 'var(--accent-cyan)', marginTop: '0.2rem' }}>{analyticsData.metrics.backbone_mAP50.toFixed(3)}</h4>
                </div>
                <Activity size={24} color="var(--accent-cyan)" />
              </div>
              <div className="glass-panel" style={{ padding: '1rem 1.25rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', textTransform: 'uppercase', fontWeight: '600' }}>Property Regression MAE</span>
                  <h4 style={{ fontSize: '1.5rem', fontWeight: '800', color: 'var(--color-mixed)', marginTop: '0.2rem' }}>{analyticsData.metrics.property_reg_mae.toFixed(3)}</h4>
                </div>
                <Layers size={24} color="var(--color-mixed)" />
              </div>
              <div className="glass-panel" style={{ padding: '1rem 1.25rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', textTransform: 'uppercase', fontWeight: '600' }}>Material Classification Acc</span>
                  <h4 style={{ fontSize: '1.5rem', fontWeight: '800', color: 'var(--color-organic)', marginTop: '0.2rem' }}>{int(analyticsData.metrics.material_accuracy * 100)}%</h4>
                </div>
                <Settings size={24} color="var(--color-organic)" />
              </div>
              <div className="glass-panel" style={{ padding: '1rem 1.25rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', textTransform: 'uppercase', fontWeight: '600' }}>Model 2 Threat F1-Score</span>
                  <h4 style={{ fontSize: '1.5rem', fontWeight: '800', color: 'var(--color-critical)', marginTop: '0.2rem' }}>{analyticsData.metrics.model2_f1.toFixed(3)}</h4>
                </div>
                <ShieldCheck size={24} color="var(--color-critical)" />
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr', gap: '1.5rem' }}>
              {/* Training Progression Line Chart */}
              <div className="glass-panel" style={{ padding: '1.5rem' }}>
                <h3 style={{ fontSize: '0.9rem', fontWeight: '800', textTransform: 'uppercase', marginBottom: '1.25rem', color: 'var(--accent-cyan)' }}>
                  Two-Stage Multi-Task Training History
                </h3>
                <div style={{ width: '100%', height: '320px' }}>
                  <ResponsiveContainer>
                    <LineChart data={analyticsData.model1_training}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                      <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} stroke="var(--text-secondary)" />
                      <YAxis yAxisId="left" stroke="var(--accent-cyan)" />
                      <YAxis yAxisId="right" orientation="right" stroke="var(--color-critical)" />
                      <Tooltip contentStyle={{ background: 'var(--bg-main)', border: '1px solid rgba(255,255,255,0.1)' }} />
                      
                      {/* Detection stage metrics */}
                      <Line yAxisId="left" type="monotone" dataKey="mAP" stroke="var(--accent-cyan)" strokeWidth={2.5} name="Detection mAP@50" />
                      <Line yAxisId="right" type="monotone" dataKey="val_loss" stroke="#f43f5e" strokeWidth={2} name="Stage 1 Loss" />
                      
                      {/* Property regression Stage 2 MAE metrics */}
                      <Line yAxisId="right" type="monotone" dataKey="property_mae" stroke="var(--color-mixed)" strokeWidth={2.5} name="Property Regression MAE" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div style={{ display: 'flex', gap: '2rem', justifyContent: 'center', marginTop: '0.5rem', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}><div style={{ width: '8px', height: '8px', background: 'var(--accent-cyan)', borderRadius: '50%' }} /> Stage 1: Detection mAP (Backbone Training)</span>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}><div style={{ width: '8px', height: '8px', background: 'var(--color-mixed)', borderRadius: '50%' }} /> Stage 2: Property MLP Training (Backbone Frozen)</span>
                </div>
              </div>

              {/* Model 2 Feature Importances */}
              <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column' }}>
                <h3 style={{ fontSize: '0.9rem', fontWeight: '800', textTransform: 'uppercase', marginBottom: '1rem', color: 'var(--accent-cyan)' }}>
                  Model 2 Feature Importance
                </h3>
                <div style={{ width: '100%', height: '280px', flex: 1 }}>
                  <ResponsiveContainer>
                    <BarChart data={analyticsData.feature_importance} layout="vertical" margin={{ left: 15 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                      <XAxis type="number" stroke="var(--text-secondary)" />
                      <YAxis dataKey="property" type="category" stroke="var(--text-secondary)" fontSize={11} width={85} />
                      <Tooltip contentStyle={{ background: 'var(--bg-main)', border: '1px solid rgba(255,255,255,0.1)' }} />
                      <Bar dataKey="importance" fill="var(--accent-cyan)">
                        {analyticsData.feature_importance.map((entry, index) => {
                          // Top 3 glowing
                          const colors = ['#f43f5e', '#f59e0b', '#3b82f6'];
                          return <Cell key={`cell-${index}`} fill={colors[index] || '#0ea5e9'} opacity={0.85 - index*0.06} />;
                        })}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textAlign: 'center', marginTop: '0.5rem' }}>
                  Features evaluated dynamically by the Model 2 Random Forest and XGBoost classifiers.
                </span>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

// Inline JS round helper for UI percentages
function int(val) {
  return Math.round(val);
}

const MATERIAL_NAMES = ["organic", "metallic", "mixed", "opaque"];
