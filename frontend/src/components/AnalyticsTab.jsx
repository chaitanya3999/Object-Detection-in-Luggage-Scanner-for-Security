import React from 'react';
import { Activity } from 'lucide-react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell 
} from 'recharts';

export default function AnalyticsTab({ analyticsData }) {
  if (!analyticsData) {
    return (
      <div className="animate-fadeIn" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', height: '100%' }}>
        <h2 className="section-title"><Activity size={22} color="var(--accent-primary)" /> Model Performance Analytics</h2>
        <div className="shimmer" style={{ width: '100%', height: '120px', borderRadius: '12px' }} />
        <div style={{ display: 'flex', gap: '1.25rem' }}>
          <div className="shimmer" style={{ flex: 1, height: '300px', borderRadius: '12px' }} />
          <div className="shimmer" style={{ flex: 1, height: '300px', borderRadius: '12px' }} />
        </div>
      </div>
    );
  }

  const { metrics, model1_training, feature_importance, model2_confusion_matrix } = analyticsData;

  return (
    <div className="animate-fadeIn" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', height: '100%' }}>
      <h2 className="section-title"><Activity size={22} color="var(--accent-primary)" /> Model Performance Analytics</h2>
      
      {/* Top Stats */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
        <div className="stat-card">
          <span className="stat-card-label">Backbone mAP50</span>
          <span className="stat-card-value">{(metrics.backbone_mAP50 * 100).toFixed(1)}%</span>
        </div>
        <div className="stat-card">
          <span className="stat-card-label">Property MAE</span>
          <span className="stat-card-value">{metrics.property_reg_mae.toFixed(3)}</span>
        </div>
        <div className="stat-card">
          <span className="stat-card-label">Classifier Precision</span>
          <span className="stat-card-value">{(metrics.model2_precision * 100).toFixed(1)}%</span>
        </div>
        <div className="stat-card">
          <span className="stat-card-label">Classifier F1-Score</span>
          <span className="stat-card-value">{(metrics.model2_f1 * 100).toFixed(1)}%</span>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.25rem', flex: 1 }}>
        {/* Loss Curve */}
        <div className="glass-panel" style={{ padding: '1.25rem', display: 'flex', flexDirection: 'column' }}>
          <h3 className="section-subtitle">Multi-Task Loss Convergence</h3>
          <div style={{ flex: 1, minHeight: 0 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={model1_training}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="epoch" stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
                <YAxis stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
                <Tooltip contentStyle={{ background: 'var(--bg-panel)', border: '1px solid var(--glass-border)', borderRadius: '8px' }} />
                <Line type="monotone" dataKey="val_loss" name="Val Loss (Detection)" stroke="var(--accent-primary)" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="property_mae" name="Property MAE" stroke="var(--color-organic)" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Feature Importance */}
        <div className="glass-panel" style={{ padding: '1.25rem', display: 'flex', flexDirection: 'column' }}>
          <h3 className="section-subtitle">Model 2: Feature Importance (Random Forest)</h3>
          <div style={{ flex: 1, minHeight: 0 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart layout="vertical" data={feature_importance} margin={{ top: 5, right: 20, left: 40, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
                <XAxis type="number" stroke="var(--text-muted)" tick={{ fontSize: 11 }} />
                <YAxis dataKey="feature" type="category" stroke="var(--text-muted)" tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ background: 'var(--bg-panel)', border: '1px solid var(--glass-border)', borderRadius: '8px' }} />
                <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                  {feature_importance.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill="var(--accent-secondary)" />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
