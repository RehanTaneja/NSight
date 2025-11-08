'use client';

import React, { useState } from 'react';
import { useUpload } from '../hooks/useUpload';
import Graph from './graph'; // Adjust import path as needed

export default function Analyzer() {
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [dataFile, setDataFile] = useState<File | null>(null);
  const { uploadResult, setUploadResult, setIsUploading, isUploading } = useUpload();

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

const handleUploadAndAnalyze = async () => {
  if (!modelFile || !dataFile) {
    alert('Please select both model and data files');
    return;
  }
  
  const formData = new FormData();
  formData.append('model', modelFile);
  formData.append('data', dataFile);

  try {
    setIsUploading(true);
    console.log('Sending request to backend...'); // Debug
    
    const response = await fetch('/api/upload', {
      method: 'POST',
      body: formData,
    });

    console.log('Response status:', response.status); // Debug
    
    const result = await response.json();
    console.log('Response data:', result); // Debug

    if (response.ok) {
      console.log('Upload successful:', result);
      setUploadResult({
        waterfall: result.waterfall,
        bar: result.bar,
        modelFilename: modelFile.name,
        dataFilename: dataFile.name
      });
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

  if (isUploading) {
    return (
      <div className="max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-md">
        <h2 className="text-xl font-bold text-gray-800 mb-4">Analyzing Files...</h2>
        <div className="animate-pulse flex space-x-4">
          <div className="flex-1 space-y-4 py-1">
            <div className="h-4 bg-gray-300 rounded w-3/4"></div>
            <div className="space-y-2">
              <div className="h-4 bg-gray-300 rounded"></div>
              <div className="h-4 bg-gray-300 rounded w-5/6"></div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!uploadResult) {
    return (
      <div className="max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-md">
        <h2 className="text-xl font-bold text-gray-800 mb-4">SHAP Analysis</h2>
        
        {/* File Upload Section */}
        <div className="space-y-4 mb-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Model File:
            </label>
            <input
              type="file"
              onChange={handleModelFileChange}
              className="border border-gray-200 rounded-md p-2 w-full"
              accept=".onnx"
            />
            {modelFile && (
              <p className="text-gray-700 mt-1">
                Selected model: <span className="font-semibold">{modelFile.name}</span>
              </p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Data File:
            </label>
            <input
              type="file"
              onChange={handleDataFileChange}
              className="border border-gray-200 rounded-md p-2 w-full"
              accept=".csv"
            />
            {dataFile && (
              <p className="text-gray-700 mt-1">
                Selected data: <span className="font-semibold">{dataFile.name}</span>
              </p>
            )}
          </div>

          <button
            onClick={handleUploadAndAnalyze}
            className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded-lg transition-colors disabled:opacity-50 w-full"
            disabled={isUploadDisabled}
          >
            Upload and Analyze
          </button>
        </div>

        <p className="text-gray-600 text-center">Please upload model (.onnx) and data (.csv) files to see the analysis results.</p>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-md">
      <h2 className="text-xl font-bold text-gray-800 mb-4">SHAP Analysis Results</h2>
      
      <div className="mb-6">
        <p className="text-gray-700">
          Model: <span className="font-semibold">{uploadResult.modelFilename}</span>
        </p>
        <p className="text-gray-700">
          Data: <span className="font-semibold">{uploadResult.dataFilename}</span>
        </p>
        
        {/* Re-upload button */}
        <button
          onClick={() => {
            setUploadResult(null);
            setModelFile(null);
            setDataFile(null);
          }}
          className="mt-4 bg-gray-500 hover:bg-gray-600 text-white font-bold py-2 px-4 rounded-lg transition-colors"
        >
          Analyze New Files
        </button>
      </div>

      <div className="border-t border-gray-400 border-dotted w-4/5 mx-auto my-14"></div>

      {/* Graph Components */}
      <div className="flex flex-col items-center gap-8">
        {/* Waterfall Plot Container */}
        <div className="w-full max-w-4xl rounded-2xl border border-gray-300 p-8 shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1 bg-linear-to-br from-blue-100 to-purple-100">
          <div className="text-center mb-6 ">
            <h3 className="text-2xl font-bold text-gray-800 mb-2">Waterfall Plot</h3>
            <p className="text-gray-600 text-sm">Feature importance visualization</p>
            <Graph 
              src={uploadResult.waterfall}
              alt="SHAP Waterfall Plot"
              className="w-full h-auto rounded-lg"
              containerClassName="flex justify-center"
            />
          </div>
        </div>
        
        {/* Bar Plot Container */}
        <div className="w-full max-w-4xl rounded-2xl border border-gray-300 p-8 shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1 bg-linear-to-br from-blue-100 to-purple-100">
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
  );
}