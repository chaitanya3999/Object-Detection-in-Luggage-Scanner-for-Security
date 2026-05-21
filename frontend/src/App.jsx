import React, { useState, useEffect, useRef } from 'react';
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
  const [luggageQueue, setLuggageQueue] = useState([]);
  const [selectedBag, setSelectedBag] = useState(null);

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
      fetch('/api/analytics')
        .then(res => res.json())
        .then(data => setAnalyticsData(data))
        .catch(err => console.error(err));
    }
  }, [activeTab]);

  // Conveyor Simulator
  useEffect(() => {
    let interval;
    if (isPlaying) {
      interval = setInterval(() => {
        const id = Math.random().toString(36).substring(7);
        const newBag = { id, timestamp: Date.now(), result: null };
        setLuggageQueue(prev => [newBag, ...prev].slice(0, 10));
        
        // Simulate scan delay
        setTimeout(() => {
          fetch('/api/analytics') // Using analytics as a dummy ping to simulate activity if no real stream exists
            .then(() => {
              setLuggageQueue(prev => prev.map(b => 
                b.id === id ? { ...b, result: { 
                  overall: { level: Math.random() > 0.8 ? 'CRITICAL' : 'SAFE', explanation: 'Auto-scanned.' },
                  bboxes: [] 
                }} : b
              ));
            });
        }, 1500);
      }, 3000);
    }
    return () => clearInterval(interval);
  }, [isPlaying]);

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
    })
    .catch(err => {
      console.error(err);
      setTipLoading(false);
    });
  };

  return (
    <div className="layout-container">
      <Toaster theme="dark" position="top-right" />
      
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} modelStatus={modelStatus} />
      
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
            scanResult={selectedBag?.result} scanLoading={false}
            selectedBoxId={selectedBoxId} setSelectedBoxId={setSelectedBoxId}
            handleSelectBag={setSelectedBag}
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
