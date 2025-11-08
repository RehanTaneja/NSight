'use client';

import React from "react";
import Description from "../components/description";
import Upload from "../components/upload";

export default function HomePage() {
  const features = [
    "Generate text summaries instantly",
    "Analyze data and visualize graphs",
    "Answer questions using AI models",
    "Integrates with multiple platforms"
  ];

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center p-8 space-y-12">
      {/* Header */}
      <header className="text-center space-y-4">
        <h1 className="text-5xl font-bold text-gray-800">AI Tool Hub</h1>
        <Description text="Welcome to your all-in-one AI tool. Generate, analyze, and visualize data quickly and effortlessly." />
      </header>

      {/* Features */}
      <section className="w-full max-w-4xl grid grid-cols-1 md:grid-cols-2 gap-6">
        {features.map((feature, idx) => (
          <div
            key={idx}
            className="p-6 bg-white rounded-lg shadow hover:shadow-lg transition-shadow"
          >
            <h3 className="text-xl font-semibold mb-2">{feature}</h3>
            <Description text={`Use the AI to ${feature.toLowerCase()}.`} />
          </div>
        ))}
      </section>

      {/* Upload Section */}
      <section className="w-full max-w-4xl">
        <h2 className="text-2xl font-bold mb-4 text-gray-800">Upload Your Data</h2>
        <Upload />
      </section>

      {/* Call to Action */}
      <section className="text-center">
        <button className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-3 px-6 rounded-lg transition-colors">
          Try the AI Tool Now
        </button>
      </section>
    </div>
  );
}
