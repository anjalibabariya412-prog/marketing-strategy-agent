import { useState, useRef, useEffect } from 'react';
import './App.css';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

function App() {
  const [initialMessage, setInitialMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [threadId, setThreadId] = useState(null);
  const [status, setStatus] = useState(null);
  const [requirementId, setRequirementId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [replyInput, setReplyInput] = useState('');
  const [isCompleted, setIsCompleted] = useState(false);

  const messagesEndRef = useRef(null);

  // Auto-scroll chat to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleStart = async () => {
    if (!initialMessage.trim()) {
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
        body: JSON.stringify({ message: initialMessage.trim() }),
      });

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }

      const data = await response.json();
      setThreadId(data.thread_id);
      setStatus(data.status);
      setRequirementId(data.requirement_id);

      if (data.status === 'completed') {
        setIsCompleted(true);
        setMessages([
          { sender: 'agent', text: 'Great, I have enough information! Generating your strategy...' }
        ]);
      } else if (data.question) {
        setMessages([
          { sender: 'agent', text: data.question }
        ]);
      }
    } catch (err) {
      console.error('Failed to start conversation:', err);
      setError(err.message || 'Failed to start conversation. Please check connection and try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleReply = async (e) => {
    if (e) e.preventDefault();
    if (!replyInput.trim() || !threadId || loading) return;

    const userText = replyInput.trim();
    setReplyInput('');
    setError(null);

    // Append user message immediately to chat history
    setMessages((prev) => [...prev, { sender: 'user', text: userText }]);
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/reply`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          thread_id: threadId,
          message: userText,
        }),
      });

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }

      const data = await response.json();
      setStatus(data.status);
      setRequirementId(data.requirement_id);

      if (data.status === 'completed') {
        setIsCompleted(true);
        setMessages((prev) => [
          ...prev,
          { sender: 'agent', text: 'Great, I have enough information! Generating your strategy...' }
        ]);
      } else if (data.question) {
        setMessages((prev) => [
          ...prev,
          { sender: 'agent', text: data.question }
        ]);
      }
    } catch (err) {
      console.error('Failed to send reply:', err);
      setError(err.message || 'Failed to send reply. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '750px', margin: '30px auto', padding: '20px', fontFamily: 'system-ui, sans-serif' }}>
      <h1>Marketing Strategy Agent</h1>

      {!threadId ? (
        /* Start Screen */
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <label htmlFor="initial-message" style={{ fontWeight: 'bold' }}>
            Tell us about your business:
          </label>
          <textarea
            id="initial-message"
            rows={6}
            value={initialMessage}
            onChange={(e) => setInitialMessage(e.target.value)}
            placeholder="Describe your business, what you offer, your marketing goal, and who you're trying to reach..."
            style={{ width: '100%', padding: '12px', fontSize: '15px', borderRadius: '6px', border: '1px solid #ccc', boxSizing: 'border-box' }}
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
        /* Full Chat Interface */
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {/* Header Bar */}
          <div style={{ padding: '12px 16px', backgroundColor: '#eef2f7', borderRadius: '6px', fontSize: '14px', color: '#333' }}>
            <strong>Session Thread ID:</strong> <code>{threadId}</code>
            {requirementId && (
              <span style={{ marginLeft: '16px' }}>
                <strong>Active Requirement:</strong> <code>{requirementId}</code>
              </span>
            )}
            <span style={{ marginLeft: '16px' }}>
              <strong>Status:</strong> <code>{status}</code>
            </span>
          </div>

          {/* Messages Scrolling Area */}
          <div style={{
            height: '400px',
            overflowY: 'auto',
            padding: '16px',
            border: '1px solid #ddd',
            borderRadius: '8px',
            backgroundColor: '#fafafa',
            display: 'flex',
            flexDirection: 'column',
            gap: '12px'
          }}>
            {messages.map((msg, index) => (
              <div
                key={index}
                style={{
                  alignSelf: msg.sender === 'user' ? 'flex-end' : 'flex-start',
                  maxWidth: '80%',
                  padding: '12px 16px',
                  borderRadius: '12px',
                  backgroundColor: msg.sender === 'user' ? '#0066cc' : '#e6e6e6',
                  color: msg.sender === 'user' ? '#ffffff' : '#111111',
                  lineHeight: '1.4'
                }}
              >
                <div style={{ fontSize: '11px', fontWeight: 'bold', marginBottom: '4px', opacity: 0.8 }}>
                  {msg.sender === 'user' ? 'You' : 'Agent'}
                </div>
                <div>{msg.text}</div>
              </div>
            ))}

            {loading && (
              <div style={{ alignSelf: 'flex-start', padding: '10px 16px', borderRadius: '12px', backgroundColor: '#eee', color: '#666', fontStyle: 'italic' }}>
                Agent is thinking...
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Error Banner */}
          {error && (
            <div style={{ padding: '10px 14px', color: '#d9534f', backgroundColor: '#fdf7f7', border: '1px solid #d9534f', borderRadius: '6px', fontSize: '14px' }}>
              {error}
            </div>
          )}

          {/* Chat Input Form / Completion Screen Placeholder */}
          {!isCompleted ? (
            <form onSubmit={handleReply} style={{ display: 'flex', gap: '10px' }}>
              <input
                type="text"
                value={replyInput}
                onChange={(e) => setReplyInput(e.target.value)}
                placeholder="Type your answer..."
                disabled={loading}
                style={{ flex: 1, padding: '12px', fontSize: '15px', borderRadius: '6px', border: '1px solid #ccc' }}
              />
              <button
                type="submit"
                disabled={loading || !replyInput.trim()}
                style={{
                  padding: '12px 24px',
                  fontSize: '15px',
                  fontWeight: 'bold',
                  color: '#fff',
                  backgroundColor: loading || !replyInput.trim() ? '#888' : '#0066cc',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: loading || !replyInput.trim() ? 'not-allowed' : 'pointer'
                }}
              >
                Send
              </button>
            </form>
          ) : (
            <div style={{ padding: '16px', backgroundColor: '#d4edda', border: '1px solid #c3e6cb', color: '#155724', borderRadius: '6px', textAlign: 'center', fontWeight: 'bold' }}>
              🎉 Strategy Generation Complete (Strategy Display Component will be added in Task 9.4)
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
