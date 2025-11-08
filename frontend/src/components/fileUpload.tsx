'use client';

import React from 'react';

interface FileUploadProps {
  readonly modelFile: File | null;
  readonly dataFile: File | null;
  readonly error: {
    type: 'validation' | 'upload' | 'server' | 'network' | null;
    message: string;
    details?: string;
  };
  readonly isUploadDisabled: boolean;
  readonly onModelFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  readonly onDataFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  readonly onUploadAndAnalyze: () => void;
  readonly onClearError: () => void;
}

export default function FileUpload({
  modelFile,
  dataFile,
  error,
  isUploadDisabled,
  onModelFileChange,
  onDataFileChange,
  onUploadAndAnalyze,
  onClearError
}: FileUploadProps) {

  const ErrorDisplay = () => {
    if (!error.type) return null;

    const errorConfig = {
      validation: {
        icon: '⚠️',
        title: 'Validation Error',
        bgColor: 'bg-yellow-50',
        borderColor: 'border-yellow-200',
        textColor: 'text-yellow-800'
      },
      upload: {
        icon: '📤',
        title: 'Upload Error',
        bgColor: 'bg-orange-50',
        borderColor: 'border-orange-200',
        textColor: 'text-orange-800'
      },
      server: {
        icon: '🔧',
        title: 'Server Error',
        bgColor: 'bg-red-50',
        borderColor: 'border-red-200',
        textColor: 'text-red-800'
      },
      network: {
        icon: '🌐',
        title: 'Network Error',
        bgColor: 'bg-blue-50',
        borderColor: 'border-blue-200',
        textColor: 'text-blue-800'
      }
    };

    const config = errorConfig[error.type] || errorConfig.upload;

    return (
      <div className={`mb-6 p-4 rounded-lg border ${config.borderColor} ${config.bgColor}`}>
        <div className="flex items-start">
          <span className="text-xl mr-3">{config.icon}</span>
          <div className="flex-1">
            <h3 className={`font-semibold ${config.textColor}`}>{config.title}</h3>
            <p className={`mt-1 text-sm ${config.textColor}`}>{error.message}</p>
            {error.details && (
              <p className={`mt-2 text-sm opacity-90 ${config.textColor}`}>{error.details}</p>
            )}
          </div>
          <button
            onClick={onClearError}
            className={`ml-4 text-lg ${config.textColor} hover:opacity-70`}
            aria-label="Dismiss error"
          >
            ×
          </button>
        </div>
      </div>
    );
  };

  return (
    <div className="max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-md">
      <h2 className="text-xl font-bold text-gray-800 mb-4">SHAP Analysis</h2>
      
      <ErrorDisplay />
      
      {/* File Upload Section */}
      <div className="space-y-4 mb-6">
        {/* Model File Input */}
        <div>
          <label htmlFor="modelFile" className="block text-sm font-medium text-gray-700 mb-2">
            Model File:
          </label>
          
          {/* Hidden file input */}
          <input
            type="file"
            onChange={onModelFileChange}
            className="hidden"
            accept=".onnx"
            id="model-file-input"
          />
          
          {/* Custom styled button/label */}
          <label
            htmlFor="model-file-input"
            className={`flex items-center justify-center cursor-pointer bg-white border-2 border-dashed rounded-lg p-4 w-full text-center transition-colors group ${
              error.type === 'validation' ? 'border-red-300 bg-red-50' : 'border-gray-300 hover:border-blue-400 hover:bg-blue-50'
            }`}
          >
            <svg className={`w-6 h-6 mr-2 group-hover:text-blue-500 ${
              error.type === 'validation' ? 'text-red-400' : 'text-gray-400'
            }`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
            <span className={`group-hover:text-blue-600 ${
              error.type === 'validation' ? 'text-red-600' : 'text-gray-600'
            }`}>
              {modelFile ? `Selected: ${modelFile.name}` : 'Click to upload model file (.onnx)'}
            </span>
          </label>
        </div>

        {/* Data File Input */}
        <div>
          <label htmlFor="dataFile" className="block text-sm font-medium text-gray-700 mb-2">
            Data File:
          </label>
          
          {/* Hidden file input */}
          <input
            type="file"
            onChange={onDataFileChange}
            className="hidden"
            accept=".csv"
            id="data-file-input"
          />
          
          {/* Custom styled button/label */}
          <label
            htmlFor="data-file-input"
            className={`flex items-center justify-center cursor-pointer bg-white border-2 border-dashed rounded-lg p-4 w-full text-center transition-colors group ${
              error.type === 'validation' ? 'border-red-300 bg-red-50' : 'border-gray-300 hover:border-blue-400 hover:bg-blue-50'
            }`}
          >
            <svg className={`w-6 h-6 mr-2 group-hover:text-blue-500 ${
              error.type === 'validation' ? 'text-red-400' : 'text-gray-400'
            }`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <span className={`group-hover:text-blue-600 ${
              error.type === 'validation' ? 'text-red-600' : 'text-gray-600'
            }`}>
              {dataFile ? `Selected: ${dataFile.name}` : 'Click to upload data file (.csv)'}
            </span>
          </label>
        </div>

        <button
          onClick={onUploadAndAnalyze}
          className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-3 px-4 rounded-lg transition-colors disabled:opacity-50 w-full"
          disabled={isUploadDisabled}
        >
          Upload and Analyze
        </button>
      </div>

      <p className="text-gray-600 text-center">Please upload model (.onnx) and data (.csv) files to see the analysis results.</p>
    </div>
  );
}