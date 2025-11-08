'use client';

import React, { useState } from "react";

export default function Upload() {
  const [file, setFile] = useState<File | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = () => {
    if (!file) return;
    console.log("Uploading file:", file);
    alert(`Uploading file: ${file.name}`);
  };

  return (
    <div className="max-w-md mx-auto p-6 bg-white rounded-lg shadow-md flex flex-col space-y-4">
      <h2 className="text-xl font-bold text-gray-800">Upload a File</h2>

      <input
        type="file"
        onChange={handleFileChange}
        className="border border-gray-200 rounded-md p-2"
      />

      {file && (
        <p className="text-gray-700">
          Selected file: <span className="font-semibold">{file.name}</span>
        </p>
      )}

      <button
        onClick={handleUpload}
        className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded-lg transition-colors disabled:opacity-50"
        disabled={!file}
      >
        Upload
      </button>
    </div>
  );
}
