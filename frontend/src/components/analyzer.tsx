'use client';

import React, { useState } from 'react';
import { useUpload } from '../hooks/useUpload';
import FileUpload from './fileUpload';
import AnalysisResult from './analysisResult';
import SkeletonLoader from './skeletonLoader';

export default function Analyzer() {
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
          message: 'Invalid model file',
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
          message: 'Invalid data file',
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
          modelFilename: modelFile.name,
          dataFilename: dataFile.name
        });
      } else {
        console.error('Upload failed with status:', response.status, result.error);
        
        // Map backend errors to the four main types
        if (response.status === 400) {
          // All 400 errors are validation errors
          setError({
            type: 'validation',
            message: 'Invalid request',
            details: result.error || 'Please check your file selections and try again.'
          });
        } else if (response.status === 500) {
          // All 500 errors are server errors
          if (result.error?.includes('Error processing files') || result.error?.includes('SHAP')) {
            setError({
              type: 'server',
              message: 'Analysis failed',
              details: result.error || 'The SHAP analysis failed. Please check your model and data files are compatible.'
            });
          } else {
            setError({
              type: 'server',
              message: 'Server error',
              details: result.error || 'An internal server error occurred. Please try again later.'
            });
          }
        } else {
          setError({
            type: 'server',
            message: 'Server error',
            details: result.error || `Server returned error: ${response.status}`
          });
        }
        setUploadResult(null);
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      if (error instanceof TypeError && error.message.includes('fetch')) {
        setError({
          type: 'network',
          message: 'Connection failed',
          details: 'Unable to connect to the analysis server. Please ensure the backend service is running.'
        });
      } else if (error instanceof SyntaxError) {
        setError({
          type: 'server',
          message: 'Invalid server response',
          details: 'The server returned an invalid response. Please check if the backend is functioning correctly.'
        });
      } else {
        setError({
          type: 'upload',
          message: 'Upload failed',
          details: 'An unexpected error occurred while uploading files. Please try again.'
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
    <AnalysisResult
      uploadResult={uploadResult}
      error={error}
      onClearResults={clearResults}
      onClearError={clearError}
    />
  );
}