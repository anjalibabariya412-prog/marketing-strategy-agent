import { useState, useRef, useEffect } from 'react';
import './App.css';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

const STANDARD_SECTIONS = [
  { key: 'business_overview', label: 'Business Overview' },
  { key: 'target_audience_insights', label: 'Target Audience & Customer Insights' },
  { key: 'competitive_positioning', label: 'Competitive Positioning' },
  { key: 'value_proposition', label: 'Value Proposition' },
  { key: 'marketing_channels_and_tactics', label: 'Marketing Channels & Tactics' },
  { key: 'customer_acquisition_approach', label: 'Customer Acquisition Approach' },
  { key: 'budget_considerations', label: 'Budget Considerations' },
  { key: 'kpis', label: 'KPIs / Success Metrics' },
  { key: 'action_plan', label: 'Action Plan' },
];

function App() {
  // Form State
  const [formData, setFormData] = useState({
    company_name: '',
    product_or_service: '',
    marketing_goal: '',
    target_audience: '',
    budget_resources: '',
  });
  const [formErrors, setFormErrors] = useState({});
  const [formSubmitting, setFormSubmitting] = useState(false);
  const [startError, setStartError] = useState(null);

  // Conversation State
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [threadId, setThreadId] = useState(null);
  const [status, setStatus] = useState(null);
  const [requirementId, setRequirementId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [replyInput, setReplyInput] = useState('');
  const [isCompleted, setIsCompleted] = useState(false);

  // Strategy fetch state
  const [strategy, setStrategy] = useState(null);
  const [loadingStrategy, setLoadingStrategy] = useState(false);
  const [strategyError, setStrategyError] = useState(null);

  const messagesEndRef = useRef(null);
  const replyTextareaRef = useRef(null);

  // Auto-scroll chat to bottom on new messages
  useEffect(() => {
    if (threadId && !isCompleted) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, loading, threadId, isCompleted]);

  // Dynamic vertical auto-expand for multiline reply textarea
  useEffect(() => {
    if (replyTextareaRef.current) {
      replyTextareaRef.current.style.height = 'auto';
      const scrollHeight = replyTextareaRef.current.scrollHeight;
      replyTextareaRef.current.style.height = `${Math.min(Math.max(scrollHeight, 80), 200)}px`;
    }
  }, [replyInput]);

  // Fetch strategy automatically when conversation completes
  useEffect(() => {
    if (isCompleted && threadId) {
      fetchStrategy();
    }
  }, [isCompleted, threadId]);

  const fetchStrategy = async () => {
    setLoadingStrategy(true);
    setStrategyError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/strategy?thread_id=${encodeURIComponent(threadId)}`);
      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }

      const data = await response.json();
      setStrategy(data.strategy);
    } catch (err) {
      console.error('Failed to fetch strategy:', err);
      setStrategyError(err.message || 'Failed to load strategy. Please try again.');
    } finally {
      setLoadingStrategy(false);
    }
  };

  const handleInputChange = (field, value) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
    if (formErrors[field]) {
      setFormErrors((prev) => ({ ...prev, [field]: '' }));
    }
  };

  const handleStartSubmit = async (e) => {
    e.preventDefault();
    if (formSubmitting) return;

    // Validate required fields (whitespace trimmed)
    const errors = {};
    if (!formData.company_name.trim()) {
      errors.company_name = 'Company / Business name is required.';
    }
    if (!formData.product_or_service.trim()) {
      errors.product_or_service = 'Product or service is required.';
    }
    if (!formData.marketing_goal.trim()) {
      errors.marketing_goal = 'Marketing goal is required.';
    }
    if (!formData.target_audience.trim()) {
      errors.target_audience = 'Target audience is required.';
    }

    setFormErrors(errors);
    if (Object.keys(errors).length > 0) {
      return;
    }

    setFormSubmitting(true);
    setStartError(null);

    const payload = {
      company_name: formData.company_name.trim(),
      product_or_service: formData.product_or_service.trim(),
      marketing_goal: formData.marketing_goal.trim(),
      target_audience: formData.target_audience.trim(),
      budget_resources: formData.budget_resources.trim() || null,
    };

    try {
      const response = await fetch(`${API_BASE_URL}/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        const detailMsg = errorData && errorData.detail
          ? (Array.isArray(errorData.detail) ? errorData.detail.map((d) => d.msg).join(', ') : errorData.detail)
          : null;
        throw new Error(detailMsg || `Server returned status ${response.status}`);
      }

      const data = await response.json();
      if (data.thread_id) {
        setThreadId(data.thread_id);
      }
      setStatus(data.status);
      setRequirementId(data.requirement_id || null);

      if (data.status === 'completed') {
        setMessages([
          {
            sender: 'agent',
            text: "Great, I have everything I need! ✨ Generating your personalized marketing strategy...",
          },
        ]);
        setTimeout(() => {
          setIsCompleted(true);
        }, 6000);
      } else if (data.question) {
        setMessages([
          { sender: 'agent', text: data.question },
        ]);
      }
    } catch (err) {
      console.error('Failed to start conversation:', err);
      setStartError(err.message || 'Failed to submit business details. Please try again.');
    } finally {
      setFormSubmitting(false);
    }
  };

  const handleSendMessage = async (e) => {
    if (e) e.preventDefault();
    if (!replyInput.trim() || loading) return;

    const userText = replyInput.trim();
    setReplyInput('');
    setError(null);

    setMessages((prev) => [...prev, { sender: 'user', text: userText }]);
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/reply`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ thread_id: threadId, message: userText }),
      });

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }

      const data = await response.json();
      if (data.thread_id) {
        setThreadId(data.thread_id);
      }
      setStatus(data.status);
      setRequirementId(data.requirement_id);

      if (data.status === 'completed') {
        setMessages((prev) => [
          ...prev,
          {
            sender: 'agent',
            text: "Great, I have everything I need! ✨ Generating your personalized marketing strategy...",
          },
        ]);
        setTimeout(() => {
          setIsCompleted(true);
        }, 6000);
      } else if (data.question) {
        setMessages((prev) => [
          ...prev,
          { sender: 'agent', text: data.question }
        ]);
      }
    } catch (err) {
      console.error('Failed to send message:', err);
      setError(err.message || 'Failed to send message. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    // Enter sends the message, while Shift+Enter creates a new line
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!loading && replyInput.trim()) {
        handleSendMessage();
      }
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1 className="app-title">Marketing Strategy Agent</h1>
        <p className="app-subtitle">AI-driven marketing strategy consultant for your business</p>
      </header>

      {!isCompleted ? (
        !threadId ? (
          /* Business Context Form Screen */
          <div className="card">
            <h2 className="form-heading">Tell us about your business</h2>
            <form onSubmit={handleStartSubmit} className="start-form" noValidate>
              <div className="form-group">
                <label className="form-label" htmlFor="company_name">
                  What is your business, and what does it offer? <span className="required-star">*</span>
                </label>
                <input
                  id="company_name"
                  type="text"
                  className={`form-input ${formErrors.company_name ? 'invalid' : ''}`}
                  value={formData.company_name}
                  onChange={(e) => handleInputChange('company_name', e.target.value)}
                  disabled={formSubmitting}
                />
                {formErrors.company_name && (
                  <div className="field-error">{formErrors.company_name}</div>
                )}
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="product_or_service">
                  What product or service would you like to promote? <span className="required-star">*</span>
                </label>
                <input
                  id="product_or_service"
                  type="text"
                  className={`form-input ${formErrors.product_or_service ? 'invalid' : ''}`}
                  value={formData.product_or_service}
                  onChange={(e) => handleInputChange('product_or_service', e.target.value)}
                  disabled={formSubmitting}
                />
                {formErrors.product_or_service && (
                  <div className="field-error">{formErrors.product_or_service}</div>
                )}
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="marketing_goal">
                  What would you like to achieve through your marketing? <span className="required-star">*</span>
                </label>
                <textarea
                  id="marketing_goal"
                  className={`form-textarea ${formErrors.marketing_goal ? 'invalid' : ''}`}
                  rows={3}
                  value={formData.marketing_goal}
                  onChange={(e) => handleInputChange('marketing_goal', e.target.value)}
                  disabled={formSubmitting}
                />
                {formErrors.marketing_goal && (
                  <div className="field-error">{formErrors.marketing_goal}</div>
                )}
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="target_audience">
                  Who are you trying to reach with your product or service? <span className="required-star">*</span>
                </label>
                <textarea
                  id="target_audience"
                  className={`form-textarea ${formErrors.target_audience ? 'invalid' : ''}`}
                  rows={3}
                  value={formData.target_audience}
                  onChange={(e) => handleInputChange('target_audience', e.target.value)}
                  disabled={formSubmitting}
                />
                {formErrors.target_audience && (
                  <div className="field-error">{formErrors.target_audience}</div>
                )}
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="budget_resources">
                  How much are you planning to invest in marketing?
                </label>
                <input
                  id="budget_resources"
                  type="text"
                  className="form-input"
                  value={formData.budget_resources}
                  onChange={(e) => handleInputChange('budget_resources', e.target.value)}
                  disabled={formSubmitting}
                />
              </div>

              {startError && <div className="error-banner">{startError}</div>}

              <button
                type="submit"
                className="btn-primary"
                disabled={formSubmitting}
              >
                {formSubmitting ? 'Starting...' : 'Start'}
              </button>
            </form>
          </div>
        ) : (
          /* Full Chat Interface Card */
          <div className="card">
            {/* Messages Scroll Area */}
            <div className="chat-scroll-area">
              {messages.map((msg, index) => (
                <div key={index} className={`message-wrapper ${msg.sender}`}>
                  <div className="message-label">
                    {msg.sender === 'user' ? 'You' : 'Agent'}
                  </div>
                  <div className="message-bubble">
                    {msg.text}
                  </div>
                </div>
              ))}

              {loading && (
                <div className="message-wrapper agent">
                  <div className="message-label">Agent</div>
                  <div className="message-bubble typing-indicator-bubble">
                    <div className="typing-dots">
                      <span className="dot" />
                      <span className="dot" />
                      <span className="dot" />
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Error Banner */}
            {error && <div className="error-banner" style={{ marginTop: '14px' }}>{error}</div>}

            {/* Multiline Chat Input Form */}
            <form onSubmit={handleSendMessage} className="chat-input-form">
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                <textarea
                  ref={replyTextareaRef}
                  className="chat-input-textarea"
                  value={replyInput}
                  onChange={(e) => setReplyInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Type your message... (Press Enter to send, Shift+Enter for new line)"
                  disabled={loading}
                  rows={3}
                />
                <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '4px', textAlign: 'right' }}>
                  Press <kbd style={{ fontFamily: 'sans-serif', backgroundColor: '#f1f5f9', padding: '1px 5px', borderRadius: '3px', border: '1px solid #cbd5e1' }}>Enter</kbd> to send, <kbd style={{ fontFamily: 'sans-serif', backgroundColor: '#f1f5f9', padding: '1px 5px', borderRadius: '3px', border: '1px solid #cbd5e1' }}>Shift</kbd> + <kbd style={{ fontFamily: 'sans-serif', backgroundColor: '#f1f5f9', padding: '1px 5px', borderRadius: '3px', border: '1px solid #cbd5e1' }}>Enter</kbd> for new line
                </div>
              </div>
              <button
                type="submit"
                className="btn-send"
                disabled={loading || !replyInput.trim()}
              >
                Send
              </button>
            </form>
          </div>
        )
      ) : (
        /* Task 9.4 Strategy Display Screen Card */
        <div>
          {loadingStrategy ? (
            <div className="card" style={{ textAlign: 'center', padding: '40px 20px' }}>
              <h2>Generating Your Customized Marketing Strategy...</h2>
              <p style={{ color: '#64748b' }}>Synthesizing your business context and facts into actionable recommendations.</p>
              <div className="thinking-dots" style={{ fontSize: '24px', color: '#2563eb', marginTop: '16px' }}>
                <span>.</span><span>.</span><span>.</span>
              </div>
            </div>
          ) : strategyError ? (
            <div className="card">
              <div className="error-banner">
                <h3 style={{ margin: '0 0 6px 0' }}>Failed to Load Marketing Strategy</h3>
                <p style={{ margin: 0 }}>{strategyError}</p>
              </div>
              <button
                onClick={fetchStrategy}
                className="btn-primary"
                style={{ marginTop: '16px' }}
              >
                Retry Loading Strategy
              </button>
            </div>
          ) : strategy ? (
            <div className="strategy-card">
              <div className="strategy-header">
                <h2 className="strategy-title">Your Marketing Strategy</h2>
              </div>

              {/* Render 9 Standard Strategy Sections (Skipping null/empty fields) */}
              {STANDARD_SECTIONS.map(({ key, label }) => {
                const val = strategy[key];
                if (!val || typeof val !== 'string' || !val.trim()) {
                  return null;
                }

                return (
                  <section key={key} className="strategy-section">
                    <h3 className="section-title">{label}</h3>
                    <p className="section-content">{val.trim()}</p>
                  </section>
                );
              })}

              {/* Render Additional Custom Sections (Skipping null/empty values) */}
              {strategy.additional_sections &&
                typeof strategy.additional_sections === 'object' &&
                Object.entries(strategy.additional_sections).map(([title, content]) => {
                  if (!content || typeof content !== 'string' || !content.trim()) {
                    return null;
                  }

                  return (
                    <section key={title} className="strategy-section">
                      <h3 className="section-title">{title}</h3>
                      <p className="section-content">{content.trim()}</p>
                    </section>
                  );
                })}
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
}

export default App;
