import React, { useState } from 'react';
import FileUploader from './components/FileUploader';
import ProcessingOptions from './components/ProcessingOptions';
import ResultDisplay from './components/ResultDisplay';
import FeaturesPage from './components/FeaturesPage';
import './App.css';

function App() {
  const [currentPage, setCurrentPage] = useState('home');
  const [uploadedFile, setUploadedFile] = useState(null);
  const [processingResult, setProcessingResult] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);

  const handleFileUpload = (file) => {
    setUploadedFile(file);
    setProcessingResult(null);
    setError(null);
  };

  const handleClearFile = () => {
    setUploadedFile(null);
    setProcessingResult(null);
  };

  const handleProcessFile = async (option) => {
    if (!uploadedFile) {
      setError('Please upload a file first');
      return;
    }

    setIsProcessing(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', uploadedFile);
    formData.append('option', option);

    try {
      const response = await fetch('http://localhost:3001/api/process-file', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Processing failed');
      }

      const result = await response.json();
      setProcessingResult(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
  };

  return (
    <div className="App">
      {/* Simple Navigation */}
      <nav className="navbar">
        <div className="nav-container">
          <div className="logo" onClick={() => setCurrentPage('home')} role="button" tabIndex={0}>
            <span className="logo-icon">✦</span>
            <span className="logo-text">CodeRefine</span>
          </div>
          
          <div className="nav-center">
            <button 
              className={`nav-link ${currentPage === 'home' ? 'active' : ''}`}
              onClick={() => setCurrentPage('home')}
            >
              Home
            </button>
            <button 
              className={`nav-link ${currentPage === 'features' ? 'active' : ''}`}
              onClick={() => setCurrentPage('features')}
            >
              Features
            </button>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="main-container">
        {currentPage === 'home' ? (
          <>
            {/* Hero Section */}
            <div className="hero-section fade-in">
              <h1 className="hero-title">
                Java Code Refactoring & Documentation
              </h1>
              <p className="hero-subtitle">
                Upload your Java file to clean up formatting or add documentation. 
                Currently optimized for small code snippets.
              </p>
            </div>

            {/* Upload Card */}
            <div className="upload-card fade-in">
              <FileUploader onFileUpload={handleFileUpload} />
              
              {uploadedFile && (
                <div className="file-info-card">
                  <div className="file-info-content">
                    <span className="file-icon">📄</span>
                    <div className="file-details">
                      <span className="file-label">Uploaded File</span>
                      <span className="file-name">{uploadedFile.name}</span>
                    </div>
                  </div>
                  <div className="file-actions">
                    <span className="file-size">{formatFileSize(uploadedFile.size)}</span>
                    <button className="file-clear-btn" onClick={handleClearFile} title="Remove file">
                      ✕
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Options Section */}
            {uploadedFile && (
              <div className="options-section fade-in">
                <div className="section-header">
                  <h2>Choose What You Need</h2>
                  <p>Select how you want to improve your code</p>
                </div>
                
                <ProcessingOptions 
                  onProcess={handleProcessFile} 
                  isProcessing={isProcessing}
                />
              </div>
            )}

            {/* Error Message */}
            {error && (
              <div className="error-message fade-in">
                <span>⚠️ {error}</span>
              </div>
            )}

            {/* Result Section */}
            {processingResult && (
              <div className="result-card fade-in">
                <ResultDisplay 
                  result={processingResult} 
                  option={processingResult.option}
                />
              </div>
            )}
          </>
        ) : (
          <FeaturesPage />
        )}
      </div>

      {/* Simple Footer */}
      <footer className="footer">
        <p>© 2026 CodeRefine. Built for small Java projects.</p>
      </footer>
    </div>
  );
}

export default App;