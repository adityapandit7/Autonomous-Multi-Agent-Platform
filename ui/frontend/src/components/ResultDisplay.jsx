import React, { useState } from 'react';
import './ResultDisplay.css';

function ResultDisplay({ result, option }) {
  const [copiedStates, setCopiedStates] = useState({
    original: false,
    refactored: false,
    documented: false
  });

  const handleDownload = (code, filename) => {
    if (!code) return;
    const blob = new Blob([code], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  };

  const handleCopy = (code, type) => {
    if (!code) return;
    navigator.clipboard.writeText(code);
    setCopiedStates({ ...copiedStates, [type]: true });
    setTimeout(() => setCopiedStates({ ...copiedStates, [type]: false }), 2000);
  };

  // If no result, show nothing
  if (!result) {
    return null;
  }

  // Get the data from result
  const originalContent = result.originalContent || '// No original code available';
  const processedContent = result.processedContent || '// No processed code available';
  
  // For both option, use specific content if available
  const refactoredContent = result.refactoredContent || processedContent;
  const documentedContent = result.documentedContent || processedContent;

  return (
    <div className="result-display">
      <div className="result-header">
        <h2>Processing Complete</h2>
      </div>

      {/* Dynamic grid based on option */}
      <div className={`code-grid ${option === 'both' ? 'three-columns' : 'two-columns'}`}>
        {/* Original Code Panel - Always shown */}
        <div className="code-panel original">
          <div className="panel-header">
            <div className="panel-title">
              <span className="panel-icon">📄</span>
              <h3>Original Code</h3>
            </div>
            <span className="line-count">{originalContent.split('\n').length} lines</span>
          </div>
          <div className="panel-content">
            <pre className="code-display">{originalContent}</pre>
          </div>
          <div className="panel-actions">
            <button 
              className="panel-btn download"
              onClick={() => handleDownload(originalContent, 'Original.java')}
            >
              <span>⬇️</span> Save Original
            </button>
            <button 
              className="panel-btn copy"
              onClick={() => handleCopy(originalContent, 'original')}
            >
              {copiedStates.original ? '✅' : '📋'} Copy
            </button>
          </div>
        </div>

        {/* For Refactor Option - Show Refactored Code Panel */}
        {option === 'refactor' && (
          <div className="code-panel refactored">
            <div className="panel-header">
              <div className="panel-title">
                <span className="panel-icon">🔄</span>
                <h3>Refactored Code</h3>
              </div>
              <span className="line-count">{processedContent.split('\n').length} lines</span>
            </div>
            <div className="panel-content">
              <pre className="code-display">{processedContent}</pre>
            </div>
            <div className="panel-actions">
              <button 
                className="panel-btn download"
                onClick={() => handleDownload(processedContent, 'Refactored.java')}
              >
                <span>⬇️</span> Save Refactored
              </button>
              <button 
                className="panel-btn copy"
                onClick={() => handleCopy(processedContent, 'refactored')}
              >
                {copiedStates.refactored ? '✅' : '📋'} Copy
              </button>
            </div>
          </div>
        )}

        {/* For Document Option - Show Documented Code Panel */}
        {option === 'document' && (
          <div className="code-panel documented">
            <div className="panel-header">
              <div className="panel-title">
                <span className="panel-icon">📝</span>
                <h3>Documented Code</h3>
              </div>
              <span className="line-count">{processedContent.split('\n').length} lines</span>
            </div>
            <div className="panel-content">
              <pre className="code-display">{processedContent}</pre>
            </div>
            <div className="panel-actions">
              <button 
                className="panel-btn download"
                onClick={() => handleDownload(processedContent, 'Documented.java')}
              >
                <span>⬇️</span> Save Documented
              </button>
              <button 
                className="panel-btn copy"
                onClick={() => handleCopy(processedContent, 'documented')}
              >
                {copiedStates.documented ? '✅' : '📋'} Copy
              </button>
            </div>
          </div>
        )}

        {/* For Both Option - Show Both Refactored AND Documented Panels */}
        {option === 'both' && (
          <>
            <div className="code-panel refactored">
              <div className="panel-header">
                <div className="panel-title">
                  <span className="panel-icon">🔄</span>
                  <h3>Refactored Code</h3>
                </div>
                <span className="line-count">{refactoredContent.split('\n').length} lines</span>
              </div>
              <div className="panel-content">
                <pre className="code-display">{refactoredContent}</pre>
              </div>
              <div className="panel-actions">
                <button 
                  className="panel-btn download"
                  onClick={() => handleDownload(refactoredContent, 'Refactored.java')}
                >
                  <span>⬇️</span> Save Refactored
                </button>
                <button 
                  className="panel-btn copy"
                  onClick={() => handleCopy(refactoredContent, 'refactored')}
                >
                  {copiedStates.refactored ? '✅' : '📋'} Copy
                </button>
              </div>
            </div>

            <div className="code-panel documented">
              <div className="panel-header">
                <div className="panel-title">
                  <span className="panel-icon">📝</span>
                  <h3>Documented Code</h3>
                </div>
                <span className="line-count">{documentedContent.split('\n').length} lines</span>
              </div>
              <div className="panel-content">
                <pre className="code-display">{documentedContent}</pre>
              </div>
              <div className="panel-actions">
                <button 
                  className="panel-btn download"
                  onClick={() => handleDownload(documentedContent, 'Documented.java')}
                >
                  <span>⬇️</span> Save Documented
                </button>
                <button 
                  className="panel-btn copy"
                  onClick={() => handleCopy(documentedContent, 'documented')}
                >
                  {copiedStates.documented ? '✅' : '📋'} Copy
                </button>
              </div>
            </div>
          </>
        )}
      </div>

      <p className="result-note">
        {result.summary || 'Each version can be saved individually or copied to your clipboard.'}
      </p>
    </div>
  );
}

export default ResultDisplay;