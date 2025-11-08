'use client';

import "../index.css"

interface SkeletonLoaderProps {
  readonly message?: string;
  readonly showGraphs?: boolean;
}

export default function SkeletonLoader({ 
  message = "Analyzing Files...",
  showGraphs = true 
}: SkeletonLoaderProps) {
  return (
    <div className="max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-md">
      <h2 className="text-xl font-bold text-gray-800 mb-4">{message}</h2>
      
      <div className="animate-pulse space-y-6">
        {/* File Info Skeleton */}
        <div className="space-y-2">
          <div className="h-4 bg-gray-200 rounded w-1/4"></div>
          <div className="h-4 bg-gray-200 rounded w-1/3"></div>
        </div>

        {/* Graph Skeletons */}
        {showGraphs && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Waterfall Plot Skeleton */}
            <div className="space-y-3">
              <div className="h-6 bg-gray-200 rounded w-1/2"></div>
              <div className="h-64 bg-gray-200 rounded-lg"></div>
            </div>
            
            {/* Bar Plot Skeleton */}
            <div className="space-y-3">
              <div className="h-6 bg-gray-200 rounded w-1/3"></div>
              <div className="h-64 bg-gray-200 rounded-lg"></div>
            </div>
          </div>
        )}

        <div className="w-full max-w-md space-y-3 mx-auto">
          <div className="flex justify-between items-center">
            <span className="text-sm font-medium text-gray-700">Processing analysis...</span>
            <div className="flex space-x-1">
              <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
              <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
              <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
            </div>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
            <div className="h-2 bg-blue-500 rounded-full animate-pulse"></div>
          </div>
          <p className="text-xs text-gray-500 text-center">Please wait while we process your files</p>
        </div>
      </div>
    </div>
  );
}