'use client';

import Analyzer from "../components/analyzer"; // Adjust the import path as needed

export default function UploadPage() {
  return (
    <div className="min-h-screen bg-gray-50 pb-1">
      <div className="relative overflow-hidden bg-linear-to-r from-blue-600 to-purple-600 text-white">
        <div className="absolute inset-0 bg-black opacity-5"></div>
        <div className="relative max-w-6xl mx-auto px-8 py-20 text-center space-y-6">
          <h2 className="text-4xl font-bold text-gray-100 mb-10">
            SHAP File Analysis
          </h2>
          <p className="text-xl text-gray-100 max-w-2xl mx-auto font-medium">
            Upload model and data files for SHAP analysis and visualization
          </p>
        </div>
      </div>

      <div className="max-w-6xl mx-auto"> {/* Increased max-width for analyzer */}

        {/* Analyzer Component */}
        <div className="bg-white rounded-2xl shadow-lg p-8 my-15">
          <Analyzer />
        </div>

        {/* Additional Info */}
        <div className="my-10 bg-white rounded-2xl shadow-lg p-6">
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