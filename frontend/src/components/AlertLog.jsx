import React, { useEffect, useState } from 'react';
import { AlertTriangle, Clock } from 'lucide-react';
import { toast } from 'sonner';

export default function AlertLog() {
  const [logs, setLogs] = useState([]);

  const fetchLogs = () => {
    fetch('/api/scan-logs?limit=15')
      .then(res => res.json())
      .then(data => {
        // If there's a new critical threat, pop a toast if we didn't have it before
        if (logs.length > 0 && data.length > 0 && data[0].id !== logs[0].id && data[0].overall_threat === 'CRITICAL') {
          toast.error(`Critical Threat Detected in Scan #${data[0].id}`, {
            description: `${data[0].threats.length} prohibited item(s) found!`,
          });
        }
        setLogs(data);
      })
      .catch(err => console.error("Error fetching scan logs:", err));
  };

  // Poll for new logs every 5 seconds
  useEffect(() => {
    fetchLogs();
    const interval = setInterval(fetchLogs, 5000);
    return () => clearInterval(interval);
  }, [logs]);

  return (
    <aside style={{
      width: '280px',
      background: 'var(--bg-panel)',
      borderLeft: '1px solid rgba(255, 255, 255, 0.05)',
      display: 'flex',
      flexDirection: 'column',
      height: '100vh',
      overflow: 'hidden'
    }}>
      <div style={{ padding: '1.25rem 1.5rem', borderBottom: '1px solid rgba(255, 255, 255, 0.05)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <Clock size={18} color="var(--accent-cyan)" />
        <h2 style={{ fontSize: '0.9rem', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Recent Scans</h2>
      </div>
      
      <div style={{ flex: 1, overflowY: 'auto', padding: '1rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
        {logs.map((log) => {
          const isThreat = log.overall_threat === 'WARNING' || log.overall_threat === 'CRITICAL';
          const borderColor = log.overall_threat === 'CRITICAL' ? 'var(--color-critical)' : 
                             log.overall_threat === 'WARNING' ? 'var(--color-warning)' : 'var(--color-safe)';
                             
          return (
            <div key={log.id} style={{
              background: 'rgba(255, 255, 255, 0.02)',
              borderLeft: `3px solid ${borderColor}`,
              borderRadius: '4px',
              padding: '0.75rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.4rem'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Scan #{log.id}</span>
                <span style={{ 
                  fontSize: '0.65rem', 
                  fontWeight: '700',
                  padding: '2px 6px',
                  borderRadius: '10px',
                  background: isThreat ? 'rgba(244, 63, 94, 0.1)' : 'rgba(34, 197, 94, 0.1)',
                  color: borderColor
                }}>
                  {log.overall_threat}
                </span>
              </div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-main)' }}>
                {log.num_objects} objects analyzed
              </div>
              {isThreat && log.threats.length > 0 && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', color: 'var(--color-critical)', fontSize: '0.7rem' }}>
                  <AlertTriangle size={12} />
                  <span>Flagged: {log.threats.map(t => t.material).join(', ')}</span>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </aside>
  );
}
