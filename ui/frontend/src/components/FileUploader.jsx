import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

function FileUploader({ onFileUpload }) {
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      onFileUpload(acceptedFiles[0]);
    }
  }, [onFileUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    multiple: false,
    accept: {
      'text/x-java': ['.java']
    }
  });

  return (
    <div 
      {...getRootProps()} 
      className={`upload-area ${isDragActive ? 'active' : ''}`}
    >
      <input {...getInputProps()} />
      <span className="upload-icon">{isDragActive ? '📂' : '📁'}</span>
      {isDragActive ? (
        <>
          <p className="upload-text">Drop your Java file here</p>
          <p className="upload-hint">Release to upload</p>
        </>
      ) : (
        <>
          <p className="upload-text">Drag & drop your Java file here</p>
          <p className="upload-hint">or click to browse (.java files only)</p>
        </>
      )}
    </div>
  );
}

export default FileUploader;