import { useState } from 'react';
import './App.css';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

function App() {
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [threadId, setThreadId] = useState(null);
  const [question, setQuestion] = useState(null);
  const [requirementId, setRequirementId] = useState(null);
  const [status, setStatus] = useState(null);

  const handleStart = async () => {
    if (!message.trim()) {
      setError('Please enter a description of your business to get started.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: message.trim() }),
      });

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }

      const data = await response.json();
      setThreadId(data.thread_id);
      setQuestion(data.question);
      setRequirementId(data.requirement_id);
      setStatus(data.status);
    } catch (err) {
      console.error('Failed to start conversation:', err);
      setError(err.message || 'Failed to start conversation. Please check connection and try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '700px', margin: '40px auto', padding: '20px', fontFamily: 'system-ui, sans-serif' }}>
      <h1>Marketing Strategy Agent</h1>

      {!threadId ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <label htmlFor="initial-message" style={{ fontWeight: 'bold' }}>
            Tell us about your business:
          </label>
          <textarea
            id="initial-message"
            rows={6}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Describe your business, what you offer, your marketing goal, and who you're trying to reach..."
            style={{ width: '100%', padding: '12px', fontSize: '15px', borderRadius: '6px', border: '1px solid #ccc' }}
            disabled={loading}
          />
          <button
            onClick={handleStart}
            disabled={loading}
            style={{
              padding: '12px 24px',
              fontSize: '16px',
              fontWeight: 'bold',
              color: '#fff',
              backgroundColor: loading ? '#888' : '#0066cc',
              border: 'none',
              borderRadius: '6px',
              cursor: loading ? 'not-allowed' : 'pointer',
              alignSelf: 'flex-start'
            }}
          >
            {loading ? 'Starting Conversation...' : 'Start'}
          </button>
          {error && (
            <div style={{ padding: '12px', color: '#d9534f', backgroundColor: '#fdf7f7', border: '1px solid #d9534f', borderRadius: '6px' }}>
              {error}
            </div>
          )}
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', backgroundColor: '#f9f9f9', padding: '20px', borderRadius: '8px' }}>
          <div style={{ fontSize: '14px', color: '#555' }}>
            <strong>Session Thread ID:</strong> <code>{threadId}</code>
            {requirementId && (
              <span style={{ marginLeft: '16px' }}>
                <strong>Requirement ID:</strong> <code>{requirementId}</code>
              </span>
            )}
            <span style={{ marginLeft: '16px' }}>
              <strong>Status:</strong> <code>{status}</code>
            </span>
          </div>
          <h3>First Clarifying Question:</h3>
          <p style={{ fontSize: '18px', lineHeight: '1.5', margin: 0, fontStyle: 'italic', color: '#222' }}>
            "{question || 'No question returned'}"
          </p>
        </div>
      )}
    </div>
  );
}

export default App;
