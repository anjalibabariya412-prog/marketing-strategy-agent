import { useState, useRef, useEffect } from 'react';
import './App.css';


const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

const STANDARD_SECTIONS = [
  { key: 'business_overview', label: 'Business Overview', category: 'Overview' },
  { key: 'target_audience_insights', label: 'Target Audience & Insights', category: 'Market' },
  { key: 'competitive_positioning', label: 'Competitive Positioning', category: 'Market' },
  { key: 'value_proposition', label: 'Value Proposition & Positioning', category: 'Strategy' },
  { key: 'marketing_channels_and_tactics', label: 'Marketing Channels & Tactics', category: 'Strategy' },
  { key: 'customer_acquisition_approach', label: 'Customer Acquisition Approach', category: 'Execution' },
  { key: 'budget_considerations', label: 'Budget Considerations & Allocation', category: 'Execution' },
  { key: 'kpis', label: 'KPIs & Success Metrics', category: 'Metrics' },
  { key: 'action_plan', label: 'Action Plan & Timeline', category: 'Execution' },
];

const getReportDescription = (goal) => {
  if (!goal || !goal.trim()) {
    return 'A comprehensive, tailored marketing strategy to drive growth and achieve business objectives.';
  }
  const cleanGoal = goal.trim().replace(/\.$/, '');
  const lower = cleanGoal.toLowerCase();

  if (lower.startsWith('a ') || lower.startsWith('to ')) {
    return `A comprehensive marketing strategy ${cleanGoal}.`;
  }

  return `A comprehensive marketing strategy to ${cleanGoal.charAt(0).toLowerCase() + cleanGoal.slice(1)}.`;
};

// SVG Icon Helpers to ensure no raw SVG text appears
const Icons = {
  Copy: () => (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  ),
  Check: () => (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  ),
  Download: () => (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" y1="15" x2="12" y2="3" />
    </svg>
  ),
  Overview: () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
      <polyline points="9 22 9 12 15 12 15 22" />
    </svg>
  ),
  Audience: () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
      <circle cx="9" cy="7" r="4" />
      <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
      <path d="M16 3.13a4 4 0 0 1 0 7.75" />
    </svg>
  ),
  Competitor: () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <line x1="2" y1="12" x2="22" y2="12" />
      <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
    </svg>
  ),
  ValueProp: () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
    </svg>
  ),
  Channels: () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="2" y="3" width="20" height="14" rx="2" ry="2" />
      <line x1="8" y1="21" x2="16" y2="21" />
      <line x1="12" y1="17" x2="12" y2="21" />
    </svg>
  ),
  Acquisition: () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
    </svg>
  ),
  Budget: () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="12" y1="1" x2="12" y2="23" />
      <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
    </svg>
  ),
  KPI: () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="20" x2="18" y2="10" />
      <line x1="12" y1="20" x2="12" y2="4" />
      <line x1="6" y1="20" x2="6" y2="14" />
    </svg>
  ),
  ActionPlan: () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
      <line x1="16" y1="2" x2="16" y2="6" />
      <line x1="8" y1="2" x2="8" y2="6" />
      <line x1="3" y1="10" x2="21" y2="10" />
    </svg>
  ),
  Additional: () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="8" x2="12" y2="16" />
      <line x1="8" y1="12" x2="16" y2="12" />
    </svg>
  ),
  Sparkles: () => (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
    </svg>
  ),
  Bot: () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="11" width="18" height="10" rx="2" />
      <circle cx="12" cy="5" r="2" />
      <path d="M12 7v4" />
      <line x1="8" y1="16" x2="8.01" y2="16" />
      <line x1="16" y1="16" x2="16.01" y2="16" />
    </svg>
  ),
  User: () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
      <circle cx="12" cy="7" r="4" />
    </svg>
  )
};

const SectionIconMap = {
  business_overview: Icons.Overview,
  target_audience_insights: Icons.Audience,
  competitive_positioning: Icons.Competitor,
  value_proposition: Icons.ValueProp,
  marketing_channels_and_tactics: Icons.Channels,
  customer_acquisition_approach: Icons.Acquisition,
  budget_considerations: Icons.Budget,
  kpis: Icons.KPI,
  action_plan: Icons.ActionPlan,
};

// Helper: Clean text content (used for copy-to-clipboard)
const cleanTextContent = (items) => {
  if (!items) return '';
  if (Array.isArray(items)) return items.filter(Boolean).join('\n');
  if (typeof items === 'string') return items.trim();
  return '';
};

// Helper: Normalize a field value to an array of strings or objects
const toItemArray = (val) => {
  if (!val) return [];
  if (Array.isArray(val)) return val.filter((s) => (typeof s === 'string' && s.trim()) || (typeof s === 'object' && s !== null));
  if (typeof val === 'string' && val.trim()) return [val.trim()];
  return [];
};


// Component: Generic Smart Section Renderer — renders an array of plain-text items as a bullet list
const SmartSectionRenderer = ({ content }) => {
  const items = toItemArray(content);

  if (items.length === 0) {
    return <p className="section-text-empty">No details provided for this section.</p>;
  }

  if (items.length === 1) {
    return (
      <div className="report-markdown-body">
        <p className="md-p">{items[0]}</p>
      </div>
    );
  }

  return (
    <div className="report-markdown-body">
      <ul className="md-ul">
        {items.map((item, idx) => (
          <li key={idx} className="md-li">{item}</li>
        ))}
      </ul>
    </div>
  );
};


// Component: Specialized Renderer for KPIs Section
const KPIsSectionRenderer = ({ content }) => {
  if (!content) return <p className="section-text-empty">No KPIs specified for this strategy.</p>;
  const rawItems = Array.isArray(content) ? content : [content];
  if (rawItems.length === 0) return <p className="section-text-empty">No KPIs specified for this strategy.</p>;

  const kpis = rawItems.map((item) => {
    if (typeof item === 'object' && item !== null) {
      return {
        metric: item.metric || item.title || 'Metric',
        target: item.target || item.value || '',
        description: item.description || ''
      };
    }
    return { metric: 'KPI Metric', target: '', description: String(item) };
  });

  return (
    <div className="kpis-section-container">
      <div className="kpi-grid">
        {kpis.map((kpi, idx) => (
          <div key={idx} className="kpi-card">
            <div className="kpi-card-header">
              <span className="kpi-card-title">{kpi.metric}</span>
              {kpi.target && <span className="kpi-card-value">{kpi.target}</span>}
            </div>
            {kpi.description && (
              <p className="kpi-card-description">{kpi.description}</p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};


// Component: Specialized Renderer for Budget Section
const BudgetSectionRenderer = ({ content }) => {
  if (!content) return <p className="section-text-empty">No budget details provided.</p>;
  const rawItems = Array.isArray(content) ? content : [content];
  if (rawItems.length === 0) return <p className="section-text-empty">No budget details provided.</p>;

  const items = rawItems.map((item) => {
    if (typeof item === 'object' && item !== null) {
      return {
        label: item.label || 'Category',
        amount: item.amount || '',
        percentage: item.percentage || null,
        description: item.description || ''
      };
    }
    return { label: 'Category', amount: '', percentage: null, description: String(item) };
  });

  return (
    <div className="budget-section-container">
      <div className="budget-table-wrapper">
        <table className="report-budget-table">
          <thead>
            <tr>
              <th>Allocation Category</th>
              <th>Budget Amount</th>
              <th>Share (%)</th>
              <th>Deployment &amp; Strategy Details</th>
            </tr>
          </thead>
          <tbody>
            {items.map((row, idx) => (
              <tr key={idx}>
                <td className="budget-cat-cell">{row.label}</td>
                <td className="budget-val-cell">{row.amount || '—'}</td>
                <td className="budget-pct-cell">{row.percentage || '—'}</td>
                <td className="budget-desc-cell">{row.description || '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};


// Component: Specialized Renderer for Action Plan Timeline
const ActionPlanRenderer = ({ content }) => {
  const items = toItemArray(content);
  if (items.length === 0) return <SmartSectionRenderer content={content} />;

  return (
    <div className="action-plan-timeline">
      {items.map((item, idx) => (
        <div key={idx} className="timeline-card">
          <div className="timeline-badge-column">
            <span className="timeline-num">{idx + 1}</span>
            {idx < items.length - 1 && <span className="timeline-line" />}
          </div>
          <div className="timeline-content-column">
            <p className="timeline-item">{item}</p>
          </div>
        </div>
      ))}
    </div>
  );
};


// Component: Specialized Renderer for Value Proposition
const ValuePropositionRenderer = ({ content }) => {
  const items = toItemArray(content);
  if (items.length === 0) return <SmartSectionRenderer content={content} />;

  const heroText = items[0] || '';
  const detailItems = items.slice(1);

  return (
    <div className="value-prop-container">
      {heroText && (
        <div className="value-prop-hero-box">
          <div className="value-prop-hero-icon">
            <Icons.ValueProp />
          </div>
          <div className="value-prop-hero-text">{heroText}</div>
        </div>
      )}

      {detailItems.length > 0 && (
        <div className="value-prop-details">
          <SmartSectionRenderer content={detailItems} />
        </div>
      )}
    </div>
  );
};


function App() {
  // Form State
  const [formData, setFormData] = useState({
    company_name: '',
    product_or_service: '',
    marketing_goal: '',
    target_audience: '',
    budget_currency: 'INR',
    budget_amount: '',
    current_marketing_channels: '',
    website_social_links: '',
  });
  const [formErrors, setFormErrors] = useState({});
  const [formSubmitting, setFormSubmitting] = useState(false);
  const [startError, setStartError] = useState(null);

  // PDF Upload State
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileUploading, setFileUploading] = useState(false);
  const [pastMarketingDocText, setPastMarketingDocText] = useState(null);
  const [fileUploadError, setFileUploadError] = useState(null);
  const [fileUploadSuccess, setFileUploadSuccess] = useState(false);

  const fileInputRef = useRef(null);

  // Question Wizard State & Chat Message History
  const [messages, setMessages] = useState([]);
  const [sessionIntro, setSessionIntro] = useState(null);
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
  const [copied, setCopied] = useState(false);

  const handleCopyStrategy = () => {
    if (!strategy) return;

    let fullText = `MARKETING STRATEGY REPORT\n\n`;
    STANDARD_SECTIONS.forEach(({ key, label }) => {
      const val = strategy[key];
      const items = toItemArray(val);
      if (items.length > 0) {
        fullText += `=== ${label.toUpperCase()} ===\n${items.join('\n')}\n\n`;
      }
    });

    if (strategy.additional_sections && typeof strategy.additional_sections === 'object') {
      Object.entries(strategy.additional_sections).forEach(([title, val]) => {
        const items = toItemArray(val);
        if (items.length > 0) {
          fullText += `=== ${title.toUpperCase()} ===\n${items.join('\n')}\n\n`;
        }
      });
    }

    navigator.clipboard.writeText(fullText.trim());
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };


  const handleDownloadPdf = () => {
    window.print();
  };

  // Collect active report sections cleanly from strategy object
  const activeReportSections = [];
  if (strategy) {
    STANDARD_SECTIONS.forEach(({ key, label, category }) => {
      const val = strategy[key];
      const items = toItemArray(val);
      if (items.length > 0) {
        activeReportSections.push({
          key,
          label,
          category,
          content: items,
          IconComponent: SectionIconMap[key] || Icons.Additional,
        });
      }
    });

    // Dynamically add ALL returned additional sections
    if (strategy.additional_sections && typeof strategy.additional_sections === 'object') {
      Object.entries(strategy.additional_sections).forEach(([title, val]) => {
        const items = toItemArray(val);
        if (items.length > 0) {
          activeReportSections.push({
            key: `add_${title}`,
            label: title,
            category: 'Additional',
            content: items,
            IconComponent: Icons.Additional,
          });
        }
      });
    }
  }

  const answerTextareaRef = useRef(null);
  const chatMessagesRef = useRef(null);
  const messagesEndRef = useRef(null);

  // File Upload Handler
  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setSelectedFile(file);
    setFileUploadError(null);
    setFileUploadSuccess(false);
    setPastMarketingDocText(null);

    if (file.size > 10 * 1024 * 1024) {
      setFileUploadError('File size exceeds the maximum limit of 10 MB.');
      return;
    }

    setFileUploading(true);

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

  useEffect(() => {
    if (chatMessagesRef.current) {
      chatMessagesRef.current.scrollTo({
        top: chatMessagesRef.current.scrollHeight,
        behavior: 'smooth'
      });
    }
  }, [messages, questionLoading]);

  useEffect(() => {
    if (threadId && !showIntroScreen && !isCompleted && !isTransitioningToStrategy && !questionLoading) {
      answerTextareaRef.current?.focus();
    }
  }, [currentQuestion, threadId, showIntroScreen, isCompleted, isTransitioningToStrategy, questionLoading]);

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
    if (field === 'budget_amount' || field === 'budget_currency') {
      setFormErrors((prev) => ({ ...prev, budget_amount: '', budget_currency: '' }));
    }
  };

  const handleStartSubmit = async (e) => {
    e.preventDefault();
    if (formSubmitting || fileUploading) return;

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
    if (!formData.current_marketing_channels.trim()) {
      errors.current_marketing_channels = 'Marketing channels are required.';
    }

    if (!formData.budget_currency) {
      errors.budget_currency = 'Currency selection is required.';
    } else if (!['INR', 'USD'].includes(formData.budget_currency)) {
      errors.budget_currency = 'Currency must be INR or USD.';
    }

    if (!formData.budget_amount || !String(formData.budget_amount).trim()) {
      errors.budget_amount = 'Marketing budget amount is required.';
    } else {
      const numAmount = Number(formData.budget_amount);
      if (isNaN(numAmount) || numAmount <= 0) {
        errors.budget_amount = 'Marketing budget amount must be greater than 0.';
      }
    }

    setFormErrors(errors);
    if (Object.keys(errors).length > 0) return;

    setMessages([]);
    setFormSubmitting(true);
    setStartError(null);

    const payload = {
      company_name: formData.company_name.trim(),
      product_or_service: formData.product_or_service.trim(),
      marketing_goal: formData.marketing_goal.trim(),
      target_audience: formData.target_audience.trim(),
      budget_resources: {
        amount: Number(formData.budget_amount),
        currency: formData.budget_currency,
      },
      current_marketing_channels: formData.current_marketing_channels.trim(),
      website_social_links: formData.website_social_links.trim() || null,
      past_marketing_document: pastMarketingDocText || null,
    };

    try {
      const response = await fetch(`${API_BASE_URL}/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
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
      if (data.thread_id) setThreadId(data.thread_id);
      if (data.session_intro) setSessionIntro(data.session_intro);
      setStatus(data.status);
      setRequirementId(data.requirement_id || null);

      if (data.status === 'completed') {
        setIsTransitioningToStrategy(true);
        setTimeout(() => setIsCompleted(true), 6000);
      } else if (data.question) {
        const initialAgentMsg = {
          id: `msg_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`,
          sender: 'agent',
          content: data.question,
          requirementId: data.requirement_id || null,
          timestamp: new Date().toISOString(),
        };
        setMessages([initialAgentMsg]);

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
    const activeReqId = requirementId || null;
    const userMsg = {
      id: `msg_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`,
      sender: 'user',
      content: messageText,
      requirementId: activeReqId,
      timestamp: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMsg]);

    setQuestionLoading(true);
    setQuestionError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/reply`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
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
      if (data.thread_id) setThreadId(data.thread_id);
      setStatus(data.status);
      setRequirementId(data.requirement_id || null);

      if (data.status === 'completed') {
        setIsTransitioningToStrategy(true);
        setTimeout(() => setIsCompleted(true), 6000);
      } else if (data.question) {
        const nextAgentMsg = {
          id: `msg_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`,
          sender: 'agent',
          content: data.question,
          requirementId: data.requirement_id || null,
          timestamp: new Date().toISOString(),
        };
        setMessages((prev) => [...prev, nextAgentMsg]);

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
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!questionLoading && answerInput.trim()) {
        handleNextQuestion();
      }
    }
  };

  const renderSectionContent = (secKey, content) => {
    if (secKey === 'kpis') {
      return <KPIsSectionRenderer content={content} />;
    }
    if (secKey === 'budget_considerations') {
      return <BudgetSectionRenderer content={content} />;
    }
    if (secKey === 'action_plan') {
      return <ActionPlanRenderer content={content} />;
    }
    if (secKey === 'value_proposition') {
      return <ValuePropositionRenderer content={content} />;
    }
    return <SmartSectionRenderer content={content} />;
  };

  const currentStep = isCompleted || isTransitioningToStrategy ? 3 : threadId ? 2 : 1;

  return (
    <div className="app-container">
      {/* Top Header - Hidden on final strategy report page for clean report view */}
      {!isCompleted && (
        <header className="app-header">
          <div className="app-brand-mark" aria-hidden="true">
            <Icons.Sparkles />
          </div>
          <h1 className="app-title">Marketing Strategy Agent</h1>
          <p className="app-subtitle">AI-driven marketing strategy consultant for your business</p>
        </header>
      )}

      {/* Global 3-Step Progress Bar (Hidden on report page) */}
      {!isCompleted && (
        <div className="step-indicator-bar" role="navigation" aria-label="Progress">
          <div className={`step-item ${currentStep === 1 ? 'active' : currentStep > 1 ? 'completed' : ''}`}>
            <div className="step-circle">
              {currentStep > 1 ? <Icons.Check /> : '1'}
            </div>
            <span className="step-label">Business Details</span>
          </div>
          <div className={`step-connector ${currentStep > 1 ? 'completed' : ''}`} />
          <div className={`step-item ${currentStep === 2 ? 'active' : currentStep > 2 ? 'completed' : ''}`}>
            <div className="step-circle">
              {currentStep > 2 ? <Icons.Check /> : '2'}
            </div>
            <span className="step-label">Additional Questions</span>
          </div>
          <div className={`step-connector ${currentStep > 2 ? 'completed' : ''}`} />
          <div className={`step-item ${currentStep === 3 ? 'active' : ''}`}>
            <div className="step-circle">3</div>
            <span className="step-label">Generate Strategy</span>
          </div>
        </div>
      )}

      {!isCompleted ? (
        !threadId ? (
          /* Business Details Form */
          <div className="card form-card">
            <div className="form-card-header">
              <h2 className="form-heading">Tell us about your business</h2>
              <p className="form-subheading">Fill in the details below to help us understand your business.</p>
            </div>

            <form onSubmit={handleStartSubmit} className="start-form" noValidate autoComplete="off">
              <div className="form-grid">
                <div className="form-group">
                  <label className="form-label" htmlFor="company_name">
                    About your business <span className="required-star">*</span>
                  </label>
                  <textarea
                    id="company_name"
                    className={`form-textarea ${formErrors.company_name ? 'invalid' : ''}`}
                    rows={2}
                    value={formData.company_name}
                    onChange={(e) => handleInputChange('company_name', e.target.value)}
                    placeholder="Tell us about your business and what it does..."
                    disabled={formSubmitting}
                  />
                  {formErrors.company_name && <div className="field-error">{formErrors.company_name}</div>}
                </div>

                <div className="form-group">
                  <label className="form-label" htmlFor="product_or_service">
                    Product or service <span className="required-star">*</span>
                  </label>
                  <textarea
                    id="product_or_service"
                    className={`form-textarea ${formErrors.product_or_service ? 'invalid' : ''}`}
                    rows={2}
                    value={formData.product_or_service}
                    onChange={(e) => handleInputChange('product_or_service', e.target.value)}
                    placeholder="What would you like to promote?"
                    disabled={formSubmitting}
                  />
                  {formErrors.product_or_service && <div className="field-error">{formErrors.product_or_service}</div>}
                </div>

                <div className="form-group">
                  <label className="form-label" htmlFor="marketing_goal">
                    Marketing goal <span className="required-star">*</span>
                  </label>
                  <textarea
                    id="marketing_goal"
                    className={`form-textarea ${formErrors.marketing_goal ? 'invalid' : ''}`}
                    rows={2}
                    value={formData.marketing_goal}
                    onChange={(e) => handleInputChange('marketing_goal', e.target.value)}
                    placeholder="What would you like to achieve through marketing?"
                    disabled={formSubmitting}
                  />
                  {formErrors.marketing_goal && <div className="field-error">{formErrors.marketing_goal}</div>}
                </div>

                <div className="form-group">
                  <label className="form-label" htmlFor="target_audience">
                    Target audience <span className="required-star">*</span>
                  </label>
                  <textarea
                    id="target_audience"
                    className={`form-textarea ${formErrors.target_audience ? 'invalid' : ''}`}
                    rows={2}
                    value={formData.target_audience}
                    onChange={(e) => handleInputChange('target_audience', e.target.value)}
                    placeholder="Who is your target audience?"
                    disabled={formSubmitting}
                  />
                  {formErrors.target_audience && <div className="field-error">{formErrors.target_audience}</div>}
                </div>

                <div className="form-group">
                  <label className="form-label" htmlFor="budget_amount">
                    Marketing budget (per month) <span className="required-star">*</span>
                  </label>
                  <div className="budget-input-group">
                    <select
                      id="budget_currency"
                      className="form-select budget-currency-select"
                      value={formData.budget_currency}
                      onChange={(e) => handleInputChange('budget_currency', e.target.value)}
                      disabled={formSubmitting}
                    >
                      <option value="INR">INR (₹) - Indian Rupee</option>
                      <option value="USD">USD ($) - US Dollar</option>
                    </select>
                    <input
                      id="budget_amount"
                      type="number"
                      min="1"
                      className={`form-input budget-amount-input ${formErrors.budget_amount ? 'invalid' : ''}`}
                      value={formData.budget_amount}
                      onChange={(e) => handleInputChange('budget_amount', e.target.value)}
                      placeholder="e.g. 30000"
                      disabled={formSubmitting}
                    />
                  </div>
                  {formErrors.budget_amount && <div className="field-error">{formErrors.budget_amount}</div>}
                </div>

                <div className="form-group">
                  <label className="form-label" htmlFor="current_marketing_channels">
                    Marketing channels <span className="required-star">*</span>
                  </label>
                  <input
                    id="current_marketing_channels"
                    type="text"
                    className={`form-input ${formErrors.current_marketing_channels ? 'invalid' : ''}`}
                    value={formData.current_marketing_channels}
                    onChange={(e) => handleInputChange('current_marketing_channels', e.target.value)}
                    placeholder="Which marketing channels do you want to use?"
                    disabled={formSubmitting}
                  />
                  {formErrors.current_marketing_channels && <div className="field-error">{formErrors.current_marketing_channels}</div>}
                </div>

                <div className="form-group full-width">
                  <label className="form-label label-with-icon" htmlFor="website_social_links">
                    <svg className="field-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
                      <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
                    </svg>
                    Website &amp; social media <span className="optional-badge">Optional</span>
                  </label>
                  <div className="form-helper-text">
                    Share your website or social media profile links to help us understand your online presence.
                  </div>
                  <textarea
                    id="website_social_links"
                    className="form-textarea"
                    rows={2}
                    value={formData.website_social_links}
                    onChange={(e) => handleInputChange('website_social_links', e.target.value)}
                    placeholder="https://yourwebsite.com"
                    disabled={formSubmitting}
                  />
                </div>

                <div className="form-group full-width">
                  <label className="form-label" htmlFor="past_marketing_doc">
                    Previous marketing activity <span className="optional-badge">Optional</span>
                  </label>
                  <div className="form-helper-text">
                    Upload a previous action plan or marketing plan if you have one.
                  </div>
                  {!selectedFile ? (
                    <div className="file-upload-box">
                      <input
                        id="past_marketing_doc"
                        ref={fileInputRef}
                        type="file"
                        accept=".pdf,application/pdf"
                        className="file-input-control"
                        onChange={handleFileChange}
                        disabled={formSubmitting || fileUploading}
                      />
                      <div className="file-upload-content">
                        <svg className="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                          <polyline points="17 8 12 3 7 8" />
                          <line x1="12" y1="3" x2="12" y2="15" />
                        </svg>
                        <span>Drag and drop your PDF here or click to upload</span>
                      </div>
                    </div>
                  ) : (
                    <div className="file-upload-status success">
                      <span className="file-name">{selectedFile.name}</span>
                      <button type="button" className="btn-remove-file" onClick={handleRemoveFile}>✕</button>
                    </div>
                  )}
                </div>
              </div>

              {startError && <div className="error-banner">{startError}</div>}

              <div className="form-actions">
                <button
                  type="submit"
                  className={`btn-primary btn-start ${formSubmitting ? 'is-submitting' : ''}`}
                  disabled={formSubmitting || fileUploading}
                >
                  {formSubmitting ? (
                    <span className="btn-loading-content">
                      <span className="btn-shimmer-sweep" aria-hidden="true" />
                      <svg className="btn-circular-spinner" width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                        <circle cx="12" cy="12" r="9" stroke="rgba(255, 255, 255, 0.35)" strokeWidth="2.5" />
                        <path d="M12 3a9 9 0 0 1 9 9" stroke="#ffffff" strokeWidth="2.5" strokeLinecap="round" />
                      </svg>
                      <span>Analyzing your business</span>
                    </span>
                  ) : fileUploading ? (
                    'Uploading File...'
                  ) : (
                    'Start →'
                  )}
                </button>
              </div>
            </form>
          </div>
        ) : (
          /* Chatbot Conversation View */
          <div className="card chat-card">
            {isTransitioningToStrategy ? (
              <div className="transition-container">
                <h2 className="transition-title">Creating your personalized marketing strategy</h2>
                <p className="transition-text">Analyzing your business context, competitors, and goals...</p>
                <div className="transition-progress-bar-container">
                  <div className="transition-progress-track">
                    <div className="transition-progress-fill-animated" />
                  </div>
                </div>
              </div>
            ) : (
              <div className="chat-container">
                {/* Chat Header */}
                <div className="chat-header">
                  <h2 className="chat-header-title">Marketing Strategy Agent</h2>
                </div>

                {/* Dynamic Session Intro (Session-level metadata, NOT stored in messages array) */}
                {sessionIntro && (
                  <div className="chat-session-intro-banner">
                    <p className="chat-session-intro-text">{sessionIntro}</p>
                  </div>
                )}

                {/* Chat Messages Feed */}
                <div className="chat-messages-container" ref={chatMessagesRef}>
                  {messages.map((message) => (
                    <div
                      key={message.id}
                      className={`chat-message-row ${message.sender === 'agent' ? 'agent-row' : 'user-row'}`}
                    >
                      {message.sender === 'agent' ? (
                        <div className="chat-avatar agent-avatar" aria-hidden="true">
                          <Icons.Bot />
                        </div>
                      ) : null}
                      <div className={`chat-bubble ${message.sender === 'agent' ? 'agent-bubble' : 'user-bubble'}`}>
                        <p className="chat-bubble-text">{message.content}</p>
                      </div>
                      {message.sender === 'user' ? (
                        <div className="chat-avatar user-avatar" aria-hidden="true">
                          <Icons.User />
                        </div>
                      ) : null}
                    </div>
                  ))}

                  {/* Temporary Thinking Indicator (does not mutate messages array) */}
                  {questionLoading && (
                    <div className="chat-message-row agent-row thinking-row">
                      <div className="chat-avatar agent-avatar" aria-hidden="true">
                        <Icons.Bot />
                      </div>
                      <div className="chat-bubble agent-bubble thinking-bubble">
                        <div className="typing-dots" aria-label="Thinking">
                          <span className="dot" />
                          <span className="dot" />
                          <span className="dot" />
                        </div>
                      </div>
                    </div>
                  )}

                  {questionError && <div className="error-banner chat-error-banner">{questionError}</div>}
                  <div ref={messagesEndRef} />
                </div>

                {/* Chat Composer */}
                <div className="chat-composer-container">
                  <textarea
                    ref={answerTextareaRef}
                    className="chat-textarea"
                    value={answerInput}
                    onChange={(e) => setAnswerInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Type your answer here..."
                    disabled={questionLoading}
                    rows={2}
                  />
                  <button
                    type="button"
                    className="btn-primary chat-send-btn"
                    onClick={handleNextQuestion}
                    disabled={questionLoading || !answerInput.trim()}
                  >
                    <span>Send</span>
                    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                      <line x1="22" y1="2" x2="11" y2="13" />
                      <polygon points="22 2 15 22 11 13 2 9 22 2" />
                    </svg>
                  </button>
                </div>
              </div>
            )}
          </div>
        )
      ) : (
        /* Redesigned React Frontend Report Presentation */
        <div className="strategy-report-wrapper">
          {loadingStrategy ? (
            <div className="card strategy-loading-card">
              <h2 className="strategy-loading-title">Generating Your Customized Marketing Strategy...</h2>
              <p className="strategy-loading-subtitle">Analyzing requirements &amp; drafting strategy...</p>
            </div>
          ) : strategyError ? (
            <div className="card strategy-error-card">
              <h3>Failed to Load Marketing Strategy</h3>
              <p>{strategyError}</p>
              <button type="button" onClick={fetchStrategy} className="btn-primary">Retry</button>
            </div>
          ) : strategy ? (
            <div className="strategy-report-container">
              {/* Header */}
              <header className="report-header-centered">
                <div className="report-header-eyebrow">STRATEGIC MARKETING REPORT</div>
                <h1 className="report-header-company">
                  Marketing Strategy Report
                </h1>
                <p className="report-header-description">
                  {getReportDescription(formData.marketing_goal)}
                </p>
                <div className="report-header-date">
                  Generated on {new Date().toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' })}
                </div>
              </header>

              {/* Strategy Snapshot Grid */}
              {(formData.marketing_goal || formData.target_audience || formData.budget_amount) && (
                <div className="report-snapshot-block">
                  <h3 className="snapshot-block-title">Executive Strategy Snapshot</h3>
                  <div className="snapshot-block-grid">
                    {formData.marketing_goal && (
                      <div className="snapshot-cell">
                        <span className="snapshot-cell-label">Primary Goal</span>
                        <span className="snapshot-cell-value">{formData.marketing_goal}</span>
                      </div>
                    )}
                    {formData.target_audience && (
                      <div className="snapshot-cell">
                        <span className="snapshot-cell-label">Target Audience</span>
                        <span className="snapshot-cell-value">{formData.target_audience}</span>
                      </div>
                    )}
                    {formData.budget_amount && (
                      <div className="snapshot-cell">
                        <span className="snapshot-cell-label">Marketing Budget</span>
                        <span className="snapshot-cell-value">
                          {formData.budget_currency === 'INR' ? '₹' : '$'}
                          {Number(formData.budget_amount).toLocaleString()}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Strategy Report Sections */}
              <div className="report-paper-document">
                <div className="report-sections-list">
                  {activeReportSections.map((sec, index) => {
                    const secNum = String(index + 1).padStart(2, '0');
                    const SectionIcon = sec.IconComponent;
                    return (
                      <section key={sec.key} className="report-doc-section">
                        <div className="report-section-header">
                          <div className="report-section-title-group">
                            <span className="report-num-badge">{secNum}</span>
                            <span className="report-section-icon" aria-hidden="true">
                              <SectionIcon />
                            </span>
                            <h2 className="report-section-heading">{sec.label}</h2>
                          </div>
                        </div>

                        <div className="report-section-content-body">
                          {renderSectionContent(sec.key, sec.content)}
                        </div>
                      </section>
                    );
                  })}
                </div>
              </div>

              {/* Report Bottom Actions */}
              <div className="report-bottom-actions">
                <button type="button" className="btn-action-pill" onClick={handleCopyStrategy}>
                  {copied ? <><Icons.Check /> Copied!</> : <><Icons.Copy /> Copy Strategy</>}
                </button>
                <button type="button" className="btn-action-pill btn-pdf" onClick={handleDownloadPdf}>
                  <Icons.Download /> Download PDF
                </button>
              </div>
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
}

export default App;
