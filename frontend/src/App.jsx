import React, { useState, useEffect, useRef } from 'react';
import { Toaster } from 'sonner';
import Sidebar from './components/Sidebar';
import AlertLog from './components/AlertLog';
import ConveyorTab from './components/ConveyorTab';
import ScannerTab from './components/ScannerTab';
import TIPTab from './components/TIPTab';
import AnalyticsTab from './components/AnalyticsTab';

export default function App() {
  const [activeTab, setActiveTab] = useState('conveyor');
  const [modelStatus, setModelStatus] = useState(null);
  
  // Conveyor Simulator State
  const [isPlaying, setIsPlaying] = useState(true);
  const [luggageQueue, setLuggageQueue] = useState([]);
  const [selectedBag, setSelectedBag] = useState(null);
  const [scanResult, setScanResult] = useState(null);
  const [scanLoading, setScanLoading] = useState(false);

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

  // Helper to convert B64 to Blob
  const b64toBlob = (b64Data, contentType='', sliceSize=512) => {
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
    return new Blob(byteArrays, {type: contentType});
  };

  // Load Initial Status
  useEffect(() => {
    fetch('/api/model-status')
      .then(res => res.json())
      .then(data => setModelStatus(data))
      .catch(err => console.error("Error loading model status: ", err));

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
        setLuggageQueue(prev => {
          const rotated = [...prev.slice(1), prev[0]];
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
        setScanResult(scanData);
        setScanLoading(false);
        setSelectedBoxId(scanData.bboxes.length > 0 ? 0 : null);
      })
      .catch(err => {
        console.error("Error scanning bag:", err);
        setScanLoading(false);
      });
  };

  const handleRunTIP = async () => {
    setTipLoading(true);
    setTipResult(null);
    setTipSelectedBoxId(null);
    try {
      const bgRes = await fetch(`/api/feed/image/${tipBgType}`);
      const bgData = await bgRes.json();
      const bgFile = new File([b64toBlob(bgData.image.split(',')[1], 'image/jpeg')], "bg.jpg", { type: 'image/jpeg' });

      let fgMockType = 'knife_bag';
      if (tipThreatType === 'scissors') fgMockType = 'toolbox';
      if (tipThreatType === 'handgun') fgMockType = 'knife_bag'; 
      if (tipThreatType === 'aerosol') fgMockType = 'aerosol_bag';
      if (tipThreatType === 'shield_block') fgMockType = 'shielded_bag';

      const fgRes = await fetch(`/api/feed/image/${fgMockType}`);
      const fgData = await fgRes.json();
      const fgFile = new File([b64toBlob(fgData.image.split(',')[1], 'image/jpeg')], "fg.jpg", { type: 'image/jpeg' });

      const formData = new FormData();
      formData.append('bg_file', bgFile);
      formData.append('fg_file', fgFile);
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
      if (data.bboxes && data.bboxes.length > 0) {
        setTipSelectedBoxId(0);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setTipLoading(false);
    }
  };

  return (
    <div className="layout-container">
      {/* Toast notifications anchored to top right */}
      <Toaster position="top-right" theme="dark" richColors />

      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} modelStatus={modelStatus} />
      
      <main className="main-content">
        {activeTab === 'conveyor' && (
          <ConveyorTab 
            isPlaying={isPlaying} setIsPlaying={setIsPlaying}
            luggageQueue={luggageQueue} selectedBag={selectedBag}
            scanResult={scanResult} scanLoading={scanLoading}
            selectedBoxId={selectedBoxId} setSelectedBoxId={setSelectedBoxId}
            handleSelectBag={handleSelectBag}
          />
        )}
        
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

      {/* Persistent real-time Alert Log on the right */}
      <AlertLog />
    </div>
  );
}
