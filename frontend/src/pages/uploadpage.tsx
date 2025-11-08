'use client';

import Upload from "../components/upload"; // Adjust the import path as needed

export default function UploadPage() {
  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-3xl mx-auto px-4">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">File Upload</h1>
          <p className="text-xl text-gray-600">
            Upload and manage your files with ease
          </p>
        </div>

        {/* Upload Component */}
        <div className="bg-white rounded-2xl shadow-lg p-8">
          <Upload />
        </div>

        {/* Additional Info */}
        <div className="mt-8 bg-white rounded-2xl shadow-lg p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Need Help?</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-600">
            <div>
              <h4 className="font-medium text-gray-700 mb-2">Supported Files</h4>
              <p>Images, documents, videos, and most common file types up to 100MB</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}