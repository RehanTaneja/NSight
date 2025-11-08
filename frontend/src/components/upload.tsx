'use client';

import React, { useState } from "react";
import { useUpload } from '../hooks/useUpload';

export default function Upload() {
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [dataFile, setDataFile] = useState<File | null>(null);
  const { setUploadResult, setIsUploading } = useUpload();

  const handleModelFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setModelFile(e.target.files[0]);
    }
  };

  const handleDataFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setDataFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!modelFile || !dataFile) {
      alert('Please select both model and data files');
      return;
    }
    
    const formData = new FormData();
    formData.append('model', modelFile);
    formData.append('data', dataFile);

    try {
      setIsUploading(true);
      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (response.ok) {
        console.log('Upload successful:', result);
        // Store the result in context for other components to use
        setUploadResult({
          waterfall: result.waterfall,
          bar: result.bar,
          modelFilename: modelFile.name,
          dataFilename: dataFile.name
        });
        alert('Files uploaded and processed successfully!');
      } else {
        console.error('Upload failed:', result.error);
        alert(`Upload failed: ${result.error}`);
        setUploadResult(null);
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('Error uploading file - make sure the Flask server is running');
      setUploadResult(null);
    } finally {
      setIsUploading(false);
    }
  };

  const isUploadDisabled = !modelFile || !dataFile;

  return (
    <div className="max-w-md mx-auto p-6 bg-white rounded-lg shadow-md flex flex-col space-y-4">
      <h2 className="text-xl font-bold text-gray-800">Upload Model and Data Files</h2>

      {/* Model File Input */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Model File:
        </label>
        <input
          type="file"
          onChange={handleModelFileChange}
          className="border border-gray-200 rounded-md p-2 w-full"
        />
        {modelFile && (
          <p className="text-gray-700 mt-1">
            Selected model: <span className="font-semibold">{modelFile.name}</span>
          </p>
        )}
      </div>

      {/* Data File Input */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Data File:
        </label>
        <input
          type="file"
          onChange={handleDataFileChange}
          className="border border-gray-200 rounded-md p-2 w-full"
        />
        {dataFile && (
          <p className="text-gray-700 mt-1">
            Selected data: <span className="font-semibold">{dataFile.name}</span>
          </p>
        )}
      </div>

      <button
        onClick={handleUpload}
        className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded-lg transition-colors disabled:opacity-50"
        disabled={isUploadDisabled}
      >
        Upload and Analyze
      </button>
    </div>
  );
}