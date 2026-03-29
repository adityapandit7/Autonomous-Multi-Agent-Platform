import React from 'react';

function FeaturesPage() {
  return (
    <div className="features-page fade-in">
      <div className="features-header">
        <h1>About CodeRefine</h1>
        <p>A simple tool for Java code cleanup and documentation</p>
      </div>

      {/* About Section */}
      <div className="about-section">
        <h2>Our Focus</h2>
        <p className="about-text">
          CodeRefine is designed to help developers quickly clean up small Java code snippets 
          and add basic documentation. We're currently focused on smaller files, perfect for student projects, coding exercises, and small utilities.
        </p>
        <p className="about-text">
          As we grow and improve our infrastructure, we plan to handle larger codebases. 
          For now, we prioritize speed and reliability for small-scale refactoring needs.
        </p>
      </div>

      {/* Refactor Agent */}
      <div className="feature-block">
        <h2>Refactor Agent</h2>
        <p className="feature-description">
          The Refactor Agent fixes common formatting issues in your Java code. It adds 
          consistent indentation, proper spacing, and correct brace placement.
        </p>
        <div className="feature-list">
          <h3>What it fixes:</h3>
          <ul>
            <li>Inconsistent indentation </li>
            <li>Missing semicolons at line ends</li>
            <li>Improper spacing around operators</li>
            <li>Brace placement standardization</li>
            <li>Basic code structure cleanup</li>
          </ul>
        </div>
      </div>

      {/* Document Agent */}
      <div className="feature-block">
        <h2>Document Agent</h2>
        <p className="feature-description">
          The Document Agent adds Javadoc comments to your classes and methods, making 
          your code easier to understand and share.
        </p>
        <div className="feature-list">
          <h3>What it adds:</h3>
          <ul>
            <li>Class-level documentation with descriptions</li>
            <li>Method documentation with @param tags</li>
            <li>@return tags for methods that return values</li>
            <li>Basic structure for Javadoc compliance</li>
          </ul>
        </div>
      </div>

      {/* Future Plans */}
      <div className="future-section">
        <h2>What's Next</h2>
        <p>
          We're working on improving our processing capabilities to handle larger files 
          and more complex code structures. Future updates will include:
        </p>
        <ul>
          <li>Support for larger codebases </li>
          <li>More intelligent refactoring patterns</li>
          <li>Better documentation context understanding</li>
          <li>Support for additional Java versions</li>
        </ul>
      </div>
    </div>
  );
}

export default FeaturesPage;