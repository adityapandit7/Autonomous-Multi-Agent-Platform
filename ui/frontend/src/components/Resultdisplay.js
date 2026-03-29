import React, { useState } from 'react';

/**
 * ResultDisplay
 * Props:
 *   result.originalContent   — the source code as uploaded
 *   result.processedContent  — primary output (refactored or documented)
 *   result.refactoredContent — set when option is "both"
 *   result.documentedContent — set when option is "both"
 *   option                   — "refactor" | "document" | "both"
 */
function ResultDisplay({ result, option }) {
  const [activeTab, setActiveTab] = useState('processed');

  const original  = result.originalContent  || '';
  const refactored = result.refactoredContent || result.processedContent || '';
  const documented = result.documentedContent || result.processedContent || '';
  const processed  = result.processedContent || '';

  // For "both" mode show tabs; otherwise just two panels
  const showTabs = option === 'both';

  const rightContent = showTabs
    ? (activeTab === 'refactored' ? refactored : documented)
    : processed;

  const rightLabel = showTabs
    ? (activeTab === 'refactored' ? 'Refactored' : 'Documented')
    : option === 'refactor' ? 'Refactored' : 'Documented';

  return (
    <div style={styles.wrapper}>

      {/* Tab bar — only shown in "both" mode */}
      {showTabs && (
        <div style={styles.tabBar}>
          <button
            style={{ ...styles.tab, ...(activeTab === 'refactored' ? styles.tabActive : {}) }}
            onClick={() => setActiveTab('refactored')}
          >
            Refactored
          </button>
          <button
            style={{ ...styles.tab, ...(activeTab === 'documented' ? styles.tabActive : {}) }}
            onClick={() => setActiveTab('documented')}
          >
            Documented
          </button>
        </div>
      )}

      {/* Two code panels */}
      <div style={styles.panels}>
        <Panel label="Original" code={original} />
        <Panel label={rightLabel} code={rightContent} highlight />
      </div>
    </div>
  );
}

function Panel({ label, code, highlight }) {
  const handleCopy = () => navigator.clipboard.writeText(code);

  return (
    <div style={styles.panel}>
      <div style={styles.panelHeader}>
        <span style={highlight ? styles.labelHighlight : styles.label}>{label}</span>
        <button style={styles.copyBtn} onClick={handleCopy} title="Copy to clipboard">
          Copy
        </button>
      </div>
      <pre style={styles.code}>{code || '(no output)'}</pre>
    </div>
  );
}

const styles = {
  wrapper: {
    width: '100%',
    fontFamily: 'monospace',
  },
  tabBar: {
    display: 'flex',
    gap: '8px',
    marginBottom: '12px',
  },
  tab: {
    padding: '6px 18px',
    border: '1px solid #444',
    borderRadius: '6px',
    background: '#1e1e1e',
    color: '#aaa',
    cursor: 'pointer',
    fontSize: '13px',
  },
  tabActive: {
    background: '#2d2d2d',
    color: '#fff',
    borderColor: '#666',
  },
  panels: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '16px',
  },
  panel: {
    background: '#1e1e1e',
    borderRadius: '8px',
    overflow: 'hidden',
    border: '1px solid #333',
    display: 'flex',
    flexDirection: 'column',
  },
  panelHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '8px 14px',
    background: '#2d2d2d',
    borderBottom: '1px solid #333',
  },
  label: {
    color: '#aaa',
    fontSize: '12px',
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  },
  labelHighlight: {
    color: '#7dd3fc',
    fontSize: '12px',
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  },
  copyBtn: {
    padding: '3px 10px',
    fontSize: '11px',
    background: '#3a3a3a',
    border: '1px solid #555',
    borderRadius: '4px',
    color: '#ccc',
    cursor: 'pointer',
  },
  code: {
    margin: 0,
    padding: '16px',
    color: '#d4d4d4',
    fontSize: '13px',
    lineHeight: '1.6',
    overflowX: 'auto',
    overflowY: 'auto',
    maxHeight: '520px',
    whiteSpace: 'pre',
    flex: 1,
  },
};

export default ResultDisplay;