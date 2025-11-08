'use client';

import Analyzer from "../components/analyzer"; // Adjust the import path as needed

export default function UploadPage() {
  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-6xl mx-auto px-4"> {/* Increased max-width for analyzer */}
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">SHAP File Analysis</h1>
          <p className="text-xl text-gray-600">
            Upload model and data files for SHAP analysis and visualization
          </p>
        </div>

        {/* Analyzer Component */}
        <div className="bg-white rounded-2xl shadow-lg p-8">
          <Analyzer />
        </div>

        {/* Additional Info */}
        <div className="mt-8 bg-white rounded-2xl shadow-lg p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">About SHAP Analysis</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm text-gray-600">
            <div>
              <h4 className="font-medium text-gray-700 mb-2">Supported Files</h4>
              <p>Machine learning model files and corresponding data files for SHAP value computation</p>
            </div>
            <div>
              <h4 className="font-medium text-gray-700 mb-2">Analysis Output</h4>
              <p>Interactive waterfall plots and bar charts showing feature importance and contributions</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}