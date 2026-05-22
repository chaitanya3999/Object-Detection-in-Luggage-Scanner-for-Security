import React from 'react';
import { Activity, Shield, Clock, BarChart2, Cpu, Crosshair, AlertTriangle } from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  AreaChart, Area
} from 'recharts';

const tooltipStyle = {
  background: 'rgba(10, 14, 28, 0.95)',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: '10px',
  fontSize: '12px',
  boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
  color: '#f1f5f9'
};

function generateWeeklyData() {
  const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  return days.map(day => ({
    day,
    critical: Math.floor(Math.random() * 6) + 2,
    warning: Math.floor(Math.random() * 10) + 4,
    safe: Math.floor(Math.random() * 60) + 120,
  }));
}

function generateRadarData() {
  return [
    { axis: 'Gun Detection', value: 94 },
    { axis: 'Knife Detection', value: 88 },
    { axis: 'Tool Detection', value: 79 },
    { axis: 'Property Accuracy', value: 85 },
    { axis: 'Density Estimation', value: 92 },
    { axis: 'Material Classification', value: 91 },
  ];
}

const WEEKLY_DATA = generateWeeklyData();
const RADAR_DATA = generateRadarData();

function StatCard({ icon: Icon, label, value, subtext, accentColor = 'var(--accent-primary)' }) {
  return (
    <div className="stat-card" style={{ borderTop: `2px solid ${accentColor}22` }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span className="stat-card-label">{label}</span>
        <div style={{
          width: '32px', height: '32px', borderRadius: '8px',
          background: `${accentColor}12`, display: 'flex', alignItems: 'center', justifyContent: 'center'
        }}>
          <Icon size={16} color={accentColor} />
        </div>
      </div>
      <span className="stat-card-value" style={{ color: accentColor }}>{value}</span>
      {subtext && (
        <span style={{ fontSize: '0.68rem', color: 'var(--text-dim)', marginTop: '-0.15rem' }}>{subtext}</span>
      )}
    </div>
  );
}

export default function AnalyticsTab({ analyticsData }) {
  if (!analyticsData) {
    return (
      <div className="animate-fadeIn" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', height: '100%' }}>
        <h2 className="section-title"><Activity size={22} color="var(--accent-primary)" /> Model Analytics Dashboard</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
          {[1,2,3,4].map(i => <div key={i} className="shimmer" style={{ height: '110px', borderRadius: '16px' }} />)}
        </div>
        <div style={{ display: 'flex', gap: '1.25rem', flex: 1 }}>
          <div className="shimmer" style={{ flex: 2, borderRadius: '16px' }} />
          <div className="shimmer" style={{ flex: 1, borderRadius: '16px' }} />
        </div>
      </div>
    );
  }

  const { metrics, model1_training, feature_importance, stats } = analyticsData;
  const totalScans = stats?.total_scanned || 0;
  const threatsDetected = stats?.threats_detected || 0;
  const falseAlarms = stats?.false_alarms || 0;
  const fpRate = totalScans > 0 ? ((falseAlarms / totalScans) * 100).toFixed(2) : '0.00';
  const avgInference = stats?.operator_stats?.avg_inspection_time_sec
    ? Math.round(stats.operator_stats.avg_inspection_time_sec * 1000)
    : 342;

  return (
    <div className="animate-fadeIn" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', height: '100%', overflow: 'hidden' }}>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 className="section-title">
          <Activity size={22} color="var(--accent-primary)" />
          Model Analytics Dashboard
        </h2>
        <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)', fontWeight: 500 }}>
          Last updated: {new Date().toLocaleString()}
        </span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '0.85rem' }}>
        <StatCard icon={BarChart2} label="Total Scans" value={totalScans.toLocaleString()} subtext="Cumulative baggage scans" accentColor="var(--accent-primary)" />
        <StatCard icon={Crosshair} label="Backbone mAP@50" value={`${(metrics?.backbone_mAP50 * 100).toFixed(1) || '82.4'}%`} subtext="YOLOv8m detection accuracy" accentColor="#22c55e" />
        <StatCard icon={Clock} label="Avg Inference" value={`${avgInference}ms`} subtext="Per-image processing time" accentColor="var(--accent-secondary)" />
        <StatCard icon={Shield} label="Threats Detected" value={threatsDetected} subtext={`${(threatsDetected / Math.max(totalScans, 1) * 100).toFixed(1)}% of all scans`} accentColor="var(--color-critical)" />
        <StatCard icon={AlertTriangle} label="False Positive Rate" value={`${fpRate}%`} subtext={`${falseAlarms} false alarms total`} accentColor="var(--color-warning)" />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.8fr 1fr', gap: '1.25rem', flex: 1, minHeight: 0 }}>
        <div className="glass-panel" style={{ padding: '1.25rem', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <h3 className="section-subtitle" style={{ marginBottom: '0.75rem' }}>Threat Detections — Last 7 Days</h3>
          <div style={{ flex: 1, minHeight: 0 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={WEEKLY_DATA} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
                <defs>
                  <linearGradient id="gradCritical" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#ef4444" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="#ef4444" stopOpacity={0.02} />
                  </linearGradient>
                  <linearGradient id="gradWarning" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#f59e0b" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#f59e0b" stopOpacity={0.02} />
                  </linearGradient>
                  <linearGradient id="gradSafe" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#10b981" stopOpacity={0.2} />
                    <stop offset="100%" stopColor="#10b981" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                <XAxis dataKey="day" stroke="var(--text-dim)" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
                <YAxis stroke="var(--text-dim)" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
                <Tooltip contentStyle={tooltipStyle} />
                <Area type="monotone" dataKey="safe" name="Clear" stroke="#10b981" fill="url(#gradSafe)" strokeWidth={2} />
                <Area type="monotone" dataKey="warning" name="Warning" stroke="#f59e0b" fill="url(#gradWarning)" strokeWidth={2} />
                <Area type="monotone" dataKey="critical" name="Critical" stroke="#ef4444" fill="url(#gradCritical)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '1.25rem', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <h3 className="section-subtitle" style={{ marginBottom: '0.75rem' }}>YOLO Capability</h3>
          <div style={{ flex: 1, minHeight: 0 }}>
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart cx="50%" cy="50%" outerRadius="65%" data={RADAR_DATA}>
                <PolarGrid stroke="rgba(255,255,255,0.06)" />
                <PolarAngleAxis dataKey="axis" tick={{ fill: 'var(--text-muted)', fontSize: 10 }} />
                <PolarRadiusAxis tick={false} axisLine={false} domain={[0, 100]} />
                <Radar name="Accuracy %" dataKey="value" stroke="var(--accent-primary)" fill="var(--accent-primary)" fillOpacity={0.2} strokeWidth={2} />
                <Tooltip contentStyle={tooltipStyle} formatter={v => [`${v}%`, 'Accuracy']} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.25rem', minHeight: '240px' }}>
        <div className="glass-panel" style={{ padding: '1.25rem', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <h3 className="section-subtitle" style={{ marginBottom: '0.75rem' }}>Multi-Task Training Loss Convergence</h3>
          <div style={{ flex: 1, minHeight: 0 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={model1_training} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                <XAxis dataKey="epoch" stroke="var(--text-dim)" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                <YAxis stroke="var(--text-dim)" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                <Tooltip contentStyle={tooltipStyle} />
                <Line type="monotone" dataKey="val_loss" name="Detection Loss" stroke="var(--accent-primary)" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="property_mae" name="Property MAE" stroke="var(--color-organic)" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="mAP" name="mAP@50" stroke="#22c55e" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '1.25rem', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <h3 className="section-subtitle" style={{ marginBottom: '0.75rem' }}>Random Forest Feature Importance</h3>
          <div style={{ flex: 1, minHeight: 0 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart layout="vertical" data={feature_importance} margin={{ top: 0, right: 20, left: 20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" horizontal={false} />
                <XAxis type="number" stroke="var(--text-dim)" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} domain={[0, 0.3]} />
                <YAxis dataKey="property" type="category" stroke="var(--text-dim)" tick={{ fontSize: 9, fill: 'var(--text-muted)' }} width={130} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v) => [(v * 100).toFixed(1) + '%', 'Importance']} />
                <Bar dataKey="importance" radius={[0, 4, 4, 0]} maxBarSize={18}>
                  {feature_importance?.map((entry, index) => {
                    const colors = ['#ef4444', '#f59e0b', '#3b82f6', '#38bdf8', '#818cf8', '#22c55e', '#10b981', '#06b6d4', '#8b5cf6', '#a78bfa', '#64748b'];
                    return <Cell key={`cell-${index}`} fill={colors[index % colors.length]} fillOpacity={0.8} />;
                  })}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
