import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
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
    current_marketing_channels: '',
    website_social_links: '',
  });
  const [formErrors, setFormErrors] = useState({});
  const [formSubmitting, setFormSubmitting] = useState(false);
  const [startError, setStartError] = useState(null);

  // PDF Document Upload State
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileUploading, setFileUploading] = useState(false);
  const [pastMarketingDocText, setPastMarketingDocText] = useState(null);
  const [fileUploadError, setFileUploadError] = useState(null);
  const [fileUploadSuccess, setFileUploadSuccess] = useState(false);

  const fileInputRef = useRef(null);

  // Question Wizard State
  const [threadId, setThreadId] = useState(null);
  const [showIntroScreen, setShowIntroScreen] = useState(false);
  const [status, setStatus] = useState(null);
  const [requirementId, setRequirementId] = useState(null);
  const [questionNumber, setQuestionNumber] = useState(1);
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [answerInput, setAnswerInput] = useState('');
  const [questionLoading, setQuestionLoading] = useState(false);
  const [questionError, setQuestionError] = useState(null);
  const [isTransitioningToStrategy, setIsTransitioningToStrategy] = useState(false);
  const [isCompleted, setIsCompleted] = useState(false);

  // Strategy fetch state
  const [strategy, setStrategy] = useState(null);
  const [loadingStrategy, setLoadingStrategy] = useState(false);
  const [strategyError, setStrategyError] = useState(null);

  const answerTextareaRef = useRef(null);

  // File Upload Handler
  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setSelectedFile(file);
    setFileUploading(true);
    setFileUploadError(null);
    setFileUploadSuccess(false);
    setPastMarketingDocText(null);

    const uploadFormData = new FormData();
    uploadFormData.append('file', file);

    try {
      const response = await fetch(`${API_BASE_URL}/extract-pdf-text`, {
        method: 'POST',
        body: uploadFormData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        const detailMsg = errorData && errorData.detail
          ? (Array.isArray(errorData.detail) ? errorData.detail.map((d) => d.msg).join(', ') : errorData.detail)
          : null;
        throw new Error(detailMsg || `Server returned status ${response.status}`);
      }

      const data = await response.json();
      setPastMarketingDocText(data.summary || null);
      setFileUploadSuccess(true);

    } catch (err) {
      console.error('Failed to extract PDF text:', err);
      setFileUploadError(err.message || 'Failed to process document. Please try a different file or proceed without one.');
    } finally {
      setFileUploading(false);
    }
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setPastMarketingDocText(null);
    setFileUploadError(null);
    setFileUploadSuccess(false);
    setFileUploading(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleViewPdf = () => {
    if (!selectedFile) return;
    const fileUrl = URL.createObjectURL(selectedFile);
    window.open(fileUrl, '_blank');
  };


  // Auto-focus textarea whenever a new question appears or question card is active
  useEffect(() => {
    if (threadId && !showIntroScreen && !isCompleted && !isTransitioningToStrategy) {
      answerTextareaRef.current?.focus();
    }
  }, [currentQuestion, threadId, showIntroScreen, isCompleted, isTransitioningToStrategy]);

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
    if (formSubmitting || fileUploading) return;

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
      current_marketing_channels: formData.current_marketing_channels.trim() || null,
      website_social_links: formData.website_social_links.trim() || null,
      past_marketing_document: pastMarketingDocText || null,
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
        setIsTransitioningToStrategy(true);
        setTimeout(() => {
          setIsCompleted(true);
        }, 6000);
      } else if (data.question) {
        setCurrentQuestion(data.question);
        setQuestionNumber(1);
        setAnswerInput('');
        setShowIntroScreen(true);
      }
    } catch (err) {
      console.error('Failed to start conversation:', err);
      setStartError(err.message || 'Failed to submit business details. Please try again.');
    } finally {
      setFormSubmitting(false);
    }
  };

  const handleNextQuestion = async () => {
    if (!answerInput.trim() || questionLoading) return;

    const messageText = answerInput.trim();
    setQuestionLoading(true);
    setQuestionError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/reply`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ thread_id: threadId, message: messageText }),
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
        setIsTransitioningToStrategy(true);
        setTimeout(() => {
          setIsCompleted(true);
        }, 6000);
      } else if (data.question) {
        setQuestionNumber((prev) => prev + 1);
        setCurrentQuestion(data.question);
        setAnswerInput('');
      }
    } catch (err) {
      console.error('Failed to send answer:', err);
      setQuestionError(err.message || 'Failed to send answer. Please try again.');
    } finally {
      setQuestionLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    // Enter key submits the answer; Shift+Enter inserts a new line
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!questionLoading && answerInput.trim()) {
        handleNextQuestion();
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
            <form onSubmit={handleStartSubmit} className="start-form" noValidate autoComplete="off">
              <div className="form-group">
                <label className="form-label" htmlFor="company_name">
                  Tell us about your business and what it does. <span className="required-star">*</span>
                </label>
                <input
                  id="company_name"
                  type="text"
                  className={`form-input ${formErrors.company_name ? 'invalid' : ''}`}
                  value={formData.company_name}
                  onChange={(e) => handleInputChange('company_name', e.target.value)}
                  disabled={formSubmitting}
                  autoComplete="off"
                />
                {formErrors.company_name && (
                  <div className="field-footer">
                    <div className="field-error">{formErrors.company_name}</div>
                  </div>
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
                  autoComplete="off"
                />
                {formErrors.product_or_service && (
                  <div className="field-footer">
                    <div className="field-error">{formErrors.product_or_service}</div>
                  </div>
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
                  autoComplete="off"
                />
                {formErrors.marketing_goal && (
                  <div className="field-footer">
                    <div className="field-error">{formErrors.marketing_goal}</div>
                  </div>
                )}
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="target_audience">
                  Who is your target audience?<span className="required-star">*</span>
                </label>
                <textarea
                  id="target_audience"
                  className={`form-textarea ${formErrors.target_audience ? 'invalid' : ''}`}
                  rows={3}
                  value={formData.target_audience}
                  onChange={(e) => handleInputChange('target_audience', e.target.value)}
                  disabled={formSubmitting}
                  autoComplete="off"
                />
                {formErrors.target_audience && (
                  <div className="field-footer">
                    <div className="field-error">{formErrors.target_audience}</div>
                  </div>
                )}
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="budget_resources">
                  How much are you planning to invest in marketing? <span className="required-star">*</span>
                </label>
                <input
                  id="budget_resources"
                  type="text"
                  className="form-input"
                  value={formData.budget_resources}
                  onChange={(e) => handleInputChange('budget_resources', e.target.value)}
                  disabled={formSubmitting}
                  autoComplete="off"
                />
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="current_marketing_channels">
                  Which marketing channels do you want to use? (e.g., Instagram, Google Ads, Offline flyers) <span className="required-star">*</span>
                </label>
                <textarea
                  id="current_marketing_channels"
                  className={`form-textarea ${formErrors.current_marketing_channels ? 'invalid' : ''}`}
                  rows={3}
                  value={formData.current_marketing_channels}
                  onChange={(e) => handleInputChange('current_marketing_channels', e.target.value)}
                  disabled={formSubmitting}
                  autoComplete="off"
                />
                {formErrors.current_marketing_channels && (
                  <div className="field-footer">
                    <div className="field-error">{formErrors.current_marketing_channels}</div>
                  </div>
                )}
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="website_social_links">
                  Website / Social Media Links (Optional)
                </label>
                <div className="field-helper">
                  Add your website, Instagram, Facebook, LinkedIn, or other public links.
                </div>
                <textarea
                  id="website_social_links"
                  className="form-textarea"
                  rows={3}
                  value={formData.website_social_links}
                  onChange={(e) => handleInputChange('website_social_links', e.target.value)}
                  disabled={formSubmitting}
                  autoComplete="off"
                />
              </div>

              {/* Optional PDF File Upload Input */}
              <div className="form-group">
                <label className="form-label" htmlFor="past_marketing_doc">
                  Have you done any marketing activities before? Upload your previous marketing plan, action plan, or campaign report (Optional)
                </label>

                {!selectedFile && (
                  <input
                    id="past_marketing_doc"
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf,application/pdf"
                    className={`file-input ${fileUploadError ? 'invalid' : ''}`}
                    onChange={handleFileChange}
                    disabled={formSubmitting || fileUploading}
                  />
                )}

                {selectedFile && (
                  <div className={`file-upload-status ${fileUploading ? 'loading' : fileUploadSuccess ? 'success' : 'error'}`}>
                    <div className="file-info-group">
                      {fileUploading && <span className="spinner-sm" />}
                      {fileUploadSuccess && <span className="success-icon">✓</span>}
                      <span className="file-name">{selectedFile.name}</span>
                      {fileUploading && <span className="uploading-text">Uploading...</span>}
                    </div>

                    <div className="file-actions-group">
                      {fileUploadSuccess && (
                        <button
                          type="button"
                          className="btn-view-pdf"
                          onClick={handleViewPdf}
                        >
                          View PDF
                        </button>
                      )}
                      <button
                        type="button"
                        className="btn-remove-file"
                        onClick={handleRemoveFile}
                        title="Remove file"
                        aria-label="Remove uploaded file"
                      >
                        ✕
                      </button>
                    </div>
                  </div>
                )}

                {fileUploadError && (
                  <div className="field-footer">
                    <div className="field-error">{fileUploadError}</div>
                  </div>
                )}
              </div>


              {startError && <div className="error-banner">{startError}</div>}

              <button
                type="submit"
                className="btn-primary"
                disabled={formSubmitting || fileUploading}
              >
                {formSubmitting ? 'Starting...' : fileUploading ? 'Uploading File...' : 'Start'}
              </button>


            </form>
          </div>
        ) : (
          /* Question Wizard Card View */
          <div className="card">
            {showIntroScreen ? (
              <div className="intro-card-content">
                <h2 className="intro-heading">Let's build your strategy</h2>
                <p className="intro-text">
                  Thanks for sharing the basics about your business. To create a strategy that's relevant to your specific business, I need a few more details. I'll ask a few focused questions based on the information you've provided.
                </p>
                <div className="intro-actions">
                  <button
                    type="button"
                    className="btn-primary"
                    onClick={() => setShowIntroScreen(false)}
                  >
                    Continue
                  </button>
                </div>
              </div>
            ) : isTransitioningToStrategy ? (
              <div className="transition-container">
                <h2 className="transition-title">I have everything I need.</h2>
                <p className="transition-text">
                  Generating your personalized marketing strategy...
                </p>
                <div className="loading-dots">
                  <span className="dot" />
                  <span className="dot" />
                  <span className="dot" />
                </div>
              </div>
            ) : (
              <div key={currentQuestion || questionNumber} className="question-card-content">
                <h2 className="question-heading">{currentQuestion}</h2>

                <div className="question-input-wrapper">
                  <textarea
                    ref={answerTextareaRef}
                    className="question-textarea"
                    value={answerInput}
                    onChange={(e) => setAnswerInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Type your answer here..."
                    disabled={questionLoading}
                    rows={4}
                    autoComplete="off"
                  />
                  <div className="keyboard-hint">
                    Press <kbd>Enter</kbd> to continue (or <kbd>Shift</kbd> + <kbd>Enter</kbd> for new line)
                  </div>
                </div>

                {questionError && <div className="error-banner">{questionError}</div>}

                <div className="question-actions">
                  <button
                    type="button"
                    className="btn-primary btn-next"
                    onClick={handleNextQuestion}
                    disabled={questionLoading || !answerInput.trim()}
                  >
                    {questionLoading ? 'Thinking...' : 'Next'}
                  </button>
                </div>
              </div>
            )}
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

                const formattedContent = val.replace(/\\n/g, '\n').trim();

                return (
                  <section key={key} className="strategy-section">
                    <h3 className="section-title">{label}</h3>
                    <div className="section-content">
                      <ReactMarkdown>{formattedContent}</ReactMarkdown>
                    </div>
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

                  const formattedContent = content.replace(/\\n/g, '\n').trim();

                  return (
                    <section key={title} className="strategy-section">
                      <h3 className="section-title">{title}</h3>
                      <div className="section-content">
                        <ReactMarkdown>{formattedContent}</ReactMarkdown>
                      </div>
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
