import React, { useState } from 'react';

function ProcessingOptions({ onProcess, isProcessing }) {
  const [selectedOption, setSelectedOption] = useState('');

  const options = [
    { 
      id: 'refactor', 
      label: 'Refactor Only', 
      description: 'Clean up formatting, fix syntax, improve structure',
      icon: '🔄'
    },
    { 
      id: 'document', 
      label: 'Document Only', 
      description: 'Add Javadoc comments and method descriptions',
      icon: '📝'
    },
    { 
      id: 'both', 
      label: 'Both', 
      description: 'Complete code improvement with documentation',
      icon: '✨'
    }
  ];

  const handleSubmit = (e) => {
    e.preventDefault();
    if (selectedOption) {
      onProcess(selectedOption);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <div className="options-grid">
        {options.map((option) => (
          <label 
            key={option.id} 
            className={`option-card ${selectedOption === option.id ? 'selected' : ''}`}
          >
            <input
              type="radio"
              name="processingOption"
              value={option.id}
              checked={selectedOption === option.id}
              onChange={(e) => setSelectedOption(e.target.value)}
              disabled={isProcessing}
              style={{ display: 'none' }}
            />
            <span className="option-icon">{option.icon}</span>
            <h3>{option.label}</h3>
            <p>{option.description}</p>
          </label>
        ))}
      </div>

      <div className="process-button-container">
        <button 
          type="submit" 
          disabled={!selectedOption || isProcessing}
          className="process-button"
        >
          {isProcessing ? (
            <>
              <span className="button-loader"></span>
              Processing...
            </>
          ) : (
            'Process File'
          )}
        </button>
      </div>
    </form>
  );
}

export default ProcessingOptions;