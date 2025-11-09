'use client';

import Graph from './graph';
import ChatBot from './chatBot';
import ChatTrigger from './chatTrigger';
import { useChat } from '../hooks/useChat';

// Match the context UploadResult interface exactly
interface UploadResult {
  waterfall: string;
  bar: string;
  summary?: string;
  modelFilename?: string;
  dataFilename?: string;
}

interface ErrorState {
  type: 'validation' | 'upload' | 'server' | 'network' | null;
  message: string;
  details?: string;
}

interface AnalysisResultsProps {
  readonly uploadResult: UploadResult | null;
  readonly error: ErrorState;
  readonly onClearResults: () => void;
  readonly onClearError: () => void;
}

export default function AnalysisResults({ 
  uploadResult, 
  error, 
  onClearResults, 
  onClearError 
}: AnalysisResultsProps) {
  const { openChat } = useChat();

  // Check if we have valid data for chatbot
  const hasValidAnalysisData = uploadResult && 
                              uploadResult.waterfall && 
                              uploadResult.bar && 
                              uploadResult.summary;

  // Error display component
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
        
        {/* Suggested actions based on error type */}
        <div className="mt-3 pt-3 border-t border-current border-opacity-20">
          <p className={`text-sm font-medium ${config.textColor} mb-2`}>Suggested actions:</p>
          <ul className={`text-sm list-disc list-inside ${config.textColor} space-y-1`}>
            {error.type === 'validation' && (
              <>
                <li>Check that both files are selected</li>
                <li>Ensure model file is .onnx format</li>
                <li>Ensure data file is .csv format</li>
              </>
            )}
            {error.type === 'network' && (
              <>
                <li>Verify the backend server is running</li>
                <li>Check your internet connection</li>
                <li>Try again in a few moments</li>
              </>
            )}
            {error.type === 'server' && (
              <>
                <li>The server encountered an error processing your files</li>
                <li>Check that your files are valid and not corrupted</li>
                <li>Try with different files if possible</li>
              </>
            )}
            {error.type === 'upload' && (
              <>
                <li>Check your file sizes (very large files may timeout)</li>
                <li>Try uploading one file at a time</li>
                <li>Ensure you have a stable internet connection</li>
              </>
            )}
          </ul>
        </div>
      </div>
    );
  };

  // Loading state
  if (!uploadResult) {
    return (
      <div className="max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-md">
        <h2 className="text-xl font-bold text-gray-800 mb-4">SHAP Analysis</h2>
        <ErrorDisplay />
        <p className="text-gray-600 text-center">Please upload model (.onnx) and data (.csv) files to see the analysis results.</p>
      </div>
    );
  }

  // Results state
  return (
    <>
      <div className="max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-md">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold text-gray-800">SHAP Analysis Results</h2>
          
          {/* AI Assistant Button - Only show when we have valid data */}
          {hasValidAnalysisData && (
            <button
              onClick={openChat}
              className="flex items-center space-x-2 bg-gradient-to-r from-blue-500 to-purple-500 text-white px-4 py-2 rounded-lg hover:from-blue-600 hover:to-purple-600 transition-all duration-300 transform hover:scale-105 shadow-lg"
            >
              <span>🤖</span>
              <span>AI Assistant</span>
            </button>
          )}
        </div>
        
        <div className="mb-6">
          {/* Use fallback text if filenames are undefined */}
          <p className="text-gray-700">
            Model: <span className="font-semibold">{uploadResult.modelFilename || 'Not specified'}</span>
          </p>
          <p className="text-gray-700">
            Data: <span className="font-semibold">{uploadResult.dataFilename || 'Not specified'}</span>
          </p>
          
          {/* Re-upload button */}
          <button
            onClick={onClearResults}
            className="mt-4 bg-gray-500 hover:bg-gray-600 text-white font-bold py-2 px-4 rounded-lg transition-colors"
          >
            Analyze New Files
          </button>
        </div>

        <div className="border-t border-gray-400 border-dotted w-4/5 mx-auto my-14"></div>

        {/* Summary Section - Only show if summary exists */}
        {/* Graph Components */}
        <div className="flex flex-col items-center gap-8">
          {/* Summary Section - Only show if summary exists */}
          {uploadResult.summary && (
            <div className="w-full max-w-4xl p-6 bg-blue-50 border border-blue-200 rounded-lg">
              <h3 className="font-semibold text-blue-800 mb-3 text-lg">AI Analysis Summary</h3>
              <p className="text-blue-700">{uploadResult.summary}</p>
            </div>
          )}
          
          {/* Waterfall Plot Container */}
          <div className="w-full max-w-4xl rounded-2xl border border-gray-300 p-8 shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1 bg-gradient-to-br from-blue-100 to-purple-100">
            <div className="text-center mb-6">
              <h3 className="text-2xl font-bold text-gray-800 mb-2">Waterfall Plot</h3>
              <p className="text-gray-600 text-sm">Feature importance visualization</p>
            </div>
            <Graph 
              src={uploadResult.waterfall}
              alt="SHAP Waterfall Plot"
              className="w-full h-auto rounded-lg"
              containerClassName="flex justify-center"
            />
          </div>
          
          {/* Bar Plot Container */}
          <div className="w-full max-w-4xl rounded-2xl border border-gray-300 p-8 shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1 bg-gradient-to-br from-blue-100 to-purple-100">
            <div className="text-center mb-6">
              <h3 className="text-2xl font-bold text-gray-800 mb-2">Bar Plot</h3>
              <p className="text-gray-600 text-sm">Feature contribution analysis</p>
            </div>
            <Graph 
              src={uploadResult.bar}
              alt="SHAP Bar Plot" 
              className="w-full h-auto rounded-lg"
              containerClassName="flex justify-center"
            />
          </div>
        </div>
      </div>

      {/* ChatBot Component - Only render when we have valid data */}
      {hasValidAnalysisData && (
        <ChatBot analysisData={uploadResult} />
      )}

      {/* Chat Trigger Button - Only show when we have valid data */}
      {hasValidAnalysisData && <ChatTrigger />}
    </>
  );
}