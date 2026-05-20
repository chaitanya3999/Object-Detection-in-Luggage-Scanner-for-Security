import React from 'react';
import { Activity, Layers, Settings, ShieldCheck } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Cell } from 'recharts';

export default function AnalyticsTab({ analyticsData }) {
  if (!analyticsData) return null;

  return (
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
            <h4 style={{ fontSize: '1.5rem', fontWeight: '800', color: 'var(--color-organic)', marginTop: '0.2rem' }}>{Math.round(analyticsData.metrics.material_accuracy * 100)}%</h4>
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
  );
}
