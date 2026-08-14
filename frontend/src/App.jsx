import React, { useState, useEffect } from 'react';

function App() {
  const [stream, setStream] = useState([]);
  const [decisions, setDecisions] = useState([]);
  const [interventions, setInterventions] = useState([]);

  // Fetch data stream and make decisions
  useEffect(() => {
    const fetchData = async () => {
      try {
        const streamRes = await fetch('http://localhost:8000/api/stream?count=2');
        const newData = await streamRes.json();
        
        setStream(prev => [...newData, ...prev].slice(0, 10)); // Keep last 10

        // Process each item through the decision engine
        for (const item of newData) {
          const decisionRes = await fetch('http://localhost:8000/api/decision', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(item)
          });
          const decisionData = await decisionRes.json();
          setDecisions(prev => [decisionData, ...prev].slice(0, 10));
        }
      } catch (err) {
        console.error("Error fetching stream:", err);
      }
    };

    const fetchInterventions = async () => {
      try {
        const res = await fetch('http://localhost:8000/api/interventions');
        const data = await res.json();
        setInterventions(data);
      } catch (err) {
        console.error("Error fetching interventions:", err);
      }
    };

    fetchData();
    fetchInterventions();
    const interval = setInterval(() => {
      fetchData();
      fetchInterventions();
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const handleIntervention = async (dataId, approved, newPrediction) => {
    try {
      await fetch(`http://localhost:8000/api/interventions/${dataId}/resolve?approved=${approved}&new_prediction=${newPrediction || ''}`, {
        method: 'POST'
      });
      setInterventions(prev => prev.filter(i => i.data_id !== dataId));
    } catch (err) {
      console.error("Error resolving intervention:", err);
    }
  };

  return (
    <div>
      <header className="app-header">
        <h1>Adaptive Decision Intelligence</h1>
        <p className="badge badge-normal">System Status: Online | Auto-processing streams</p>
      </header>

      <div className="dashboard-grid">
        {/* Stream Panel */}
        <div className="glass-panel">
          <h2>Live Data Stream</h2>
          <p style={{marginBottom: '1rem', color: 'var(--text-muted)'}}>Ingesting multimodal edge data</p>
          
          <div style={{display: 'flex', flexDirection: 'column', gap: '0.5rem'}}>
            {stream.length === 0 ? <p>Waiting for data...</p> : stream.map(item => {
              const isMissingData = item.temperature === null || item.pressure === null || item.vibration === null;
              return (
                <div key={item.id} className="stream-item glass-panel" style={{padding: '0.75rem', marginBottom: '0'}}>
                  <div>
                    <strong>{item.id}</strong>
                    <div style={{fontSize: '0.85rem', color: 'var(--text-muted)'}}>
                      Temp: {item.temperature ? item.temperature.toFixed(2) : 'NULL'} | Pressure: {item.pressure ? item.pressure.toFixed(2) : 'NULL'} | Vib: {item.vibration ? item.vibration.toFixed(2) : 'NULL'}
                    </div>
                  </div>
                  {isMissingData ? (
                    <span className="badge badge-warning">Missing Data</span>
                  ) : (
                    <span className="badge badge-normal">Complete</span>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Decisions Panel */}
        <div className="glass-panel">
          <h2>AI Decisions</h2>
          <p style={{marginBottom: '1rem', color: 'var(--text-muted)'}}>Real-time inference & confidence</p>
          
          <div style={{display: 'flex', flexDirection: 'column', gap: '0.5rem'}}>
            {decisions.length === 0 ? <p>Waiting for decisions...</p> : decisions.map((item, idx) => (
              <div key={`${item.data_id}-${idx}`} className="stream-item glass-panel" style={{padding: '0.75rem', marginBottom: '0'}}>
                <div style={{flex: 1}}>
                  <div style={{display: 'flex', justifyContent: 'space-between'}}>
                    <strong>{item.data_id}</strong>
                    <span className={`badge ${item.prediction === 'ANOMALY_DETECTED' ? 'badge-anomaly' : 'badge-normal'}`}>
                      {item.prediction}
                    </span>
                  </div>
                  <div style={{fontSize: '0.85rem', color: 'var(--text-muted)', margin: '0.5rem 0'}}>
                    {item.explanation}
                  </div>
                  <div>
                    <div style={{display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem'}}>
                      <span>Confidence</span>
                      <span>{(item.confidence_score * 100).toFixed(0)}%</span>
                    </div>
                    <div className="progress-bg">
                      <div className="progress-fill" style={{
                        width: `${item.confidence_score * 100}%`,
                        backgroundColor: item.confidence_score > 0.7 ? 'var(--success)' : 'var(--warning)'
                      }}></div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Human-in-the-loop Intervention Queue */}
        <div className="glass-panel intervention-section">
          <h2 style={{display: 'flex', alignItems: 'center', gap: '0.5rem'}}>
            <span className={interventions.length > 0 ? "pulsing" : ""} style={{
              width: '12px', height: '12px', borderRadius: '50%', 
              backgroundColor: interventions.length > 0 ? 'var(--danger)' : 'var(--success)'
            }}></span>
            Human Intervention Queue
          </h2>
          <p style={{marginBottom: '1rem', color: 'var(--text-muted)'}}>
            Decisions falling below confidence threshold require manual review
          </p>

          {interventions.length === 0 ? (
            <div style={{textAlign: 'center', padding: '2rem', color: 'var(--text-muted)'}}>
              No pending interventions. AI is highly confident.
            </div>
          ) : (
            <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '1rem'}}>
              {interventions.map(item => (
                <div key={item.data_id} className="intervention-card">
                  <div>
                    <h3 style={{marginBottom: '0.25rem'}}>Data ID: {item.data_id}</h3>
                    <div className="badge badge-warning" style={{display: 'inline-block', marginBottom: '0.5rem'}}>
                      Low Confidence: {(item.confidence_score * 100).toFixed(0)}%
                    </div>
                    <p style={{fontSize: '0.9rem'}}>{item.explanation}</p>
                    <p style={{fontSize: '0.9rem', marginTop: '0.5rem'}}>
                      <strong>AI Suggestion:</strong> {item.prediction}
                    </p>
                  </div>
                  
                  <div className="intervention-actions">
                    <button className="btn btn-success" style={{flex: 1}} 
                      onClick={() => handleIntervention(item.data_id, true, item.prediction)}>
                      Approve
                    </button>
                    <button className="btn btn-danger" style={{flex: 1}}
                      onClick={() => handleIntervention(item.data_id, false, item.prediction === 'NORMAL_OPERATION' ? 'ANOMALY_DETECTED' : 'NORMAL_OPERATION')}>
                      Override
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
