import React, { useState, useEffect, useRef, useCallback } from 'react';
import Sidebar from './components/Sidebar';
import AlertLog from './components/AlertLog';
import ScannerTab from './components/ScannerTab';
import ConveyorTab from './components/ConveyorTab';
import TIPTab from './components/TIPTab';
import AnalyticsTab from './components/AnalyticsTab';
import { Toaster } from 'sonner';

export default function App() {
  const [activeTab, setActiveTab] = useState('manual');
  const [modelStatus, setModelStatus] = useState(null);
  
  // Analytics State
  const [analyticsData, setAnalyticsData] = useState(null);
  
  // Manual Tab State
  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploadPreview, setUploadPreview] = useState(null);
  const [manualScanResult, setManualScanResult] = useState(null);
  const [scanLoading, setScanLoading] = useState(false);
  const [hoveredBoxId, setHoveredBoxId] = useState(null);
  const [selectedBoxId, setSelectedBoxId] = useState(null);
  const analyzerImgRef = useRef(null);

  // Conveyor State
  const [isPlaying, setIsPlaying] = useState(false);
  const [feedItems, setFeedItems] = useState([]);
  const [luggageQueue, setLuggageQueue] = useState([]);
  const [selectedBag, setSelectedBag] = useState(null);
  const [conveyorScanResult, setConveyorScanResult] = useState(null);
  const [conveyorLoading, setConveyorLoading] = useState(false);
  const [conveyorSelectedBoxId, setConveyorSelectedBoxId] = useState(null);

  // TIP State
  const [tipBgType, setTipBgType] = useState('safe_luggage');
  const [tipThreatType, setTipThreatType] = useState('glock_19');
  const [tipScale, setTipScale] = useState(1.0);
  const [tipAngle, setTipAngle] = useState(0);
  const [tipPosX, setTipPosX] = useState(50);
  const [tipPosY, setTipPosY] = useState(50);
  const [tipThickness, setTipThickness] = useState(1.0);
  const [tipResult, setTipResult] = useState(null);
  const [tipLoading, setTipLoading] = useState(false);
  const [tipHoveredBoxId, setTipHoveredBoxId] = useState(null);
  const [tipSelectedBoxId, setTipSelectedBoxId] = useState(null);
  const tipImgRef = useRef(null);

  const threatPresets = [
    { id: 'glock_19', name: 'Glock 19 (High Density Metal)' },
    { id: 'kitchen_knife', name: 'Chef Knife (Sharp Profile)' },
    { id: 'pipe_bomb', name: 'Pipe Bomb (Mixed Materials)' },
    { id: 'scissors', name: 'Scissors (Tool/Weapon)' },
    { id: 'water_bottle', name: 'Water Bottle (Dense Organic)' }
  ];

  // ── Helpers ──────────────────────────────────────
  const b64toBlob = (b64Data, contentType = '', sliceSize = 512) => {
    const byteCharacters = atob(b64Data);
    const byteArrays = [];
    for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {
      const slice = byteCharacters.slice(offset, offset + sliceSize);
      const byteNumbers = new Array(slice.length);
      for (let i = 0; i < slice.length; i++) {
        byteNumbers[i] = slice.charCodeAt(i);
      }
      byteArrays.push(new Uint8Array(byteNumbers));
    }
    return new Blob(byteArrays, { type: contentType });
  };

  // Fetch Status
  useEffect(() => {
    fetch('/api/model-status')
      .then(res => res.json())
      .then(data => setModelStatus(data))
      .catch(err => console.error(err));
  }, []);

  // Fetch Analytics
  useEffect(() => {
    if (activeTab === 'analytics' && !analyticsData) {
      Promise.all([
        fetch('/api/mock-analytics').then(r => r.json()),
        fetch('/api/stats').then(r => r.json())
      ])
        .then(([modelData, statsData]) => setAnalyticsData({ ...modelData, stats: statsData }))
        .catch(err => console.error(err));
    }
  }, [activeTab]);

  // Conveyor: Fetch Feed on Mount
  useEffect(() => {
    fetch('/api/feed')
      .then(res => res.json())
      .then(data => {
        setFeedItems(data);
      })
      .catch(err => console.error('Error loading luggage feed:', err));
  }, []);

  // Conveyor: Auto-Push when Playing
  useEffect(() => {
    let interval;
    if (isPlaying && feedItems.length > 0) {
      interval = setInterval(() => {
        const nextBagTemplate = feedItems[Math.floor(Math.random() * feedItems.length)];
        const newBag = { ...nextBagTemplate, id: Date.now().toString() + Math.random().toString().substring(2,6) };
        
        setLuggageQueue(prev => {
          const nextQueue = [...prev, newBag];
          return nextQueue.length > 10 ? nextQueue.slice(-10) : nextQueue;
        });
        
        handleSelectBag(newBag);
      }, 3000);
    }
    return () => clearInterval(interval);
  }, [isPlaying, feedItems]);

  // Conveyor: Select Bag & Run Real Scan
  const handleSelectBag = useCallback((bag) => {
    setSelectedBag(bag);
    setConveyorLoading(true);
    setConveyorScanResult(null);
    setConveyorSelectedBoxId(null);

    fetch(`/api/feed/image/${bag.mock_type}`)
      .then(res => res.json())
      .then(imgData => {
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
        setConveyorScanResult(scanData);
        setConveyorLoading(false);
        setConveyorSelectedBoxId(scanData.bboxes && scanData.bboxes.length > 0 ? 0 : null);
        
        // Update the bag in the queue with the real AI results
        setLuggageQueue(prev => 
          prev.map(item => item.id === bag.id ? { ...item, result: scanData } : item)
        );
      })
      .catch(err => {
        console.error('Error scanning bag:', err);
        setConveyorLoading(false);
      });
  }, []);

  const handleRunTIP = () => {
    setTipLoading(true);
    setTipResult(null);
    setTipSelectedBoxId(null);
    fetch('/api/tip/project', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        bg_type: tipBgType,
        threat_type: tipThreatType,
        scale: tipScale,
        angle: tipAngle,
        pos_x: tipPosX,
        pos_y: tipPosY,
        material_thickness: tipThickness
      })
    })
    .then(res => res.json())
    .then(data => {
      setTipResult(data);
      setTipLoading(false);
      if (data.scan_results && data.scan_results.bboxes && data.scan_results.bboxes.length > 0) {
        setTipSelectedBoxId(0);
      }
    })
    .catch(err => {
      console.error(err);
      setTipLoading(false);
    });
  };

  return (
    <div className="layout-container">
      <Toaster theme="dark" position="top-right" />
      
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} modelStatus={modelStatus} scanResult={manualScanResult} />
      
      <main className="main-content">
        {activeTab === 'manual' && (
          <ScannerTab 
            uploadedFile={uploadedFile} setUploadedFile={setUploadedFile}
            uploadPreview={uploadPreview} setUploadPreview={setUploadPreview}
            manualScanResult={manualScanResult} setManualScanResult={setManualScanResult}
            scanLoading={scanLoading} setScanLoading={setScanLoading}
            hoveredBoxId={hoveredBoxId} setHoveredBoxId={setHoveredBoxId}
            selectedBoxId={selectedBoxId} setSelectedBoxId={setSelectedBoxId}
            analyzerImgRef={analyzerImgRef}
          />
        )}

        {activeTab === 'conveyor' && (
          <ConveyorTab 
            isPlaying={isPlaying} setIsPlaying={setIsPlaying}
            luggageQueue={luggageQueue} selectedBag={selectedBag}
            scanResult={conveyorScanResult} scanLoading={conveyorLoading}
            selectedBoxId={conveyorSelectedBoxId} setSelectedBoxId={setConveyorSelectedBoxId}
            handleSelectBag={handleSelectBag}
          />
        )}

        {activeTab === 'tip' && (
          <TIPTab 
            tipBgType={tipBgType} setTipBgType={setTipBgType}
            tipThreatType={tipThreatType} setTipThreatType={setTipThreatType}
            tipScale={tipScale} setTipScale={setTipScale}
            tipAngle={tipAngle} setTipAngle={setTipAngle}
            tipPosX={tipPosX} setTipPosX={setTipPosX}
            tipPosY={tipPosY} setTipPosY={setTipPosY}
            tipThickness={tipThickness} setTipThickness={setTipThickness}
            tipResult={tipResult} tipLoading={tipLoading}
            tipHoveredBoxId={tipHoveredBoxId} setTipHoveredBoxId={setTipHoveredBoxId}
            tipSelectedBoxId={tipSelectedBoxId} setTipSelectedBoxId={setTipSelectedBoxId}
            threatPresets={threatPresets} handleRunTIP={handleRunTIP}
            tipImgRef={tipImgRef}
          />
        )}

        {activeTab === 'analytics' && (
          <AnalyticsTab analyticsData={analyticsData} />
        )}
      </main>
      
      <AlertLog />
    </div>
  );
}
