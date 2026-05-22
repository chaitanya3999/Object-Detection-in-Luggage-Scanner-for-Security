import React, { useEffect, useState } from 'react';
import { AlertTriangle, Clock } from 'lucide-react';
import { toast } from 'sonner';

export default function AlertLog() {
  const [logs, setLogs] = useState([]);

  const fetchLogs = () => {
    fetch('/api/scan-logs?limit=15')
      .then(res => res.json())
      .then(data => {
        if (logs.length > 0 && data.length > 0 && data[0].id !== logs[0].id && data[0].overall_threat === 'CRITICAL') {
          toast.error(`Critical Threat Detected in Scan #${data[0].id}`, {
            description: `${data[0].threats?.length || 1} prohibited item(s) found!`,
          });
        }
        setLogs(data);
      })
      .catch(err => console.error("Error fetching scan logs:", err));
  };

  useEffect(() => {
    fetchLogs();
    const interval = setInterval(fetchLogs, 5000);
    return () => clearInterval(interval);
  }, [logs]);

  return (
    <aside className="alert-log">
      <div className="alert-log-header">
        <Clock size={18} color="var(--accent-primary)" />
        <h2>Recent Scans</h2>
      </div>
      
      <div className="alert-log-body">
        {logs.map((log) => {
          const isThreat = log.overall_threat === 'WARNING' || log.overall_threat === 'CRITICAL';
          const levelClass = log.overall_threat === 'CRITICAL' ? 'critical' : log.overall_threat === 'WARNING' ? 'warning' : 'safe';
                             
          return (
            <div key={log.id} className={`alert-entry ${levelClass}`}>
              <div className="alert-entry-header">
                <span className="alert-entry-id">Scan #{log.id}</span>
                <span className={`alert-badge ${levelClass}`}>
                  {log.overall_threat}
                </span>
              </div>
              <div className="alert-entry-body">
                {log.num_objects} objects analyzed
              </div>
              {isThreat && log.threats?.length > 0 && (
                <div className="alert-entry-flag">
                  <AlertTriangle size={12} />
                  <span>Flagged: {log.threats.map(t => t.material || t.classification).join(', ')}</span>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </aside>
  );
}
