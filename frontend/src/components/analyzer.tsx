'use client';

import React, { useState } from 'react';
import { useUpload } from '../hooks/useUpload';
import { ChatProvider } from '../contexts/chatProvider';
import FileUpload from './fileUpload';
import AnalysisResults from './analysisResult';
import SkeletonLoader from './skeletonLoader';

// Inner component that uses the upload hook
function AnalyzerContent() {
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [dataFile, setDataFile] = useState<File | null>(null);
  const [error, setError] = useState<{
    type: 'validation' | 'upload' | 'server' | 'network' | null;
    message: string;
    details?: string;
  }>({ type: null, message: '' });
  
  const { uploadResult, setUploadResult, setIsUploading, isUploading } = useUpload();

  const handleModelFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      if (!file.name.toLowerCase().endsWith('.onnx')) {
        setError({
          type: 'validation',
          message: 'Invalid file type',
          details: 'Please select a valid .onnx model file'
        });
        return;
      }
      setModelFile(file);
      setError({ type: null, message: '' });
    }
  };

  const handleDataFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      if (!file.name.toLowerCase().endsWith('.csv')) {
        setError({
          type: 'validation',
          message: 'Invalid file type',
          details: 'Please select a valid .csv data file'
        });
        return;
      }
      setDataFile(file);
      setError({ type: null, message: '' });
    }
  };

  const handleUploadAndAnalyze = async () => {
    if (!modelFile || !dataFile) {
      setError({
        type: 'validation',
        message: 'Missing files',
        details: 'Please select both model and data files'
      });
      return;
    }
    
    const formData = new FormData();
    formData.append('model', modelFile);
    formData.append('data', dataFile);

    try {
      setIsUploading(true);
      setError({ type: null, message: '' });
      console.log('Sending request to backend...');
      
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      console.log('Response status:', response.status);
      
      const result = await response.json();
      console.log('Response data:', result);

      if (response.ok) {
        console.log('Upload successful:', result);
        setUploadResult({
          waterfall: result.waterfall,
          bar: result.bar,
          summary: result.summary, // Added the missing summary field
          modelFilename: modelFile.name,
          dataFilename: dataFile.name
        });
      } else {
        console.error('Upload failed:', result.error);
        setError({
          type: 'server',
          message: 'Analysis failed',
          details: result.error || 'Server returned an error'
        });
        setUploadResult(null);
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      if (error instanceof TypeError && error.message.includes('fetch')) {
        setError({
          type: 'network',
          message: 'Connection error',
          details: 'Unable to connect to the server. Please check if the backend service is running.'
        });
      } else {
        setError({
          type: 'upload',
          message: 'Upload failed',
          details: 'An unexpected error occurred during file upload'
        });
      }
      setUploadResult(null);
    } finally {
      setIsUploading(false);
    }
  };

  const clearError = () => {
    setError({ type: null, message: '' });
  };

  const clearResults = () => {
    setUploadResult(null);
    setModelFile(null);
    setDataFile(null);
    setError({ type: null, message: '' });
  };

  const isUploadDisabled = !modelFile || !dataFile;

  if (isUploading) {
    return <SkeletonLoader />;
  }

  if (!uploadResult) {
    return (
      <FileUpload
        modelFile={modelFile}
        dataFile={dataFile}
        error={error}
        isUploadDisabled={isUploadDisabled}
        onModelFileChange={handleModelFileChange}
        onDataFileChange={handleDataFileChange}
        onUploadAndAnalyze={handleUploadAndAnalyze}
        onClearError={clearError}
      />
    );
  }

  return (
    <AnalysisResults
      uploadResult={uploadResult}
      error={error}
      onClearResults={clearResults}
      onClearError={clearError}
    />
  );
}

// Main export that wraps with ChatProvider
export default function Analyzer() {
  return (
    <ChatProvider>
      <AnalyzerContent />
    </ChatProvider>
  );
}