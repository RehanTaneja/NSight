'use client';

import React from 'react';
import { useChat } from '../hooks/useChat';

export default function ChatTrigger() {
  const { openChat, isChatOpen } = useChat();

  if (isChatOpen) return null;

  return (
    <button
      onClick={openChat}
      className="fixed bottom-6 right-6 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-full p-4 shadow-2xl hover:shadow-3xl transition-all duration-300 transform hover:scale-110 z-40 group"
      aria-label="Open AI Assistant"
    >
      <div className="relative">
        <span className="text-2xl">🤖</span>
        <div className="absolute -top-2 -right-2 w-3 h-3 bg-green-400 rounded-full animate-ping"></div>
        <div className="absolute -top-2 -right-2 w-3 h-3 bg-green-500 rounded-full"></div>
      </div>
      
      {/* Tooltip */}
      <div className="absolute bottom-full right-0 mb-2 hidden group-hover:block">
        <div className="bg-gray-900 text-white text-sm rounded py-1 px-2 whitespace-nowrap">
          Ask AI about your analysis
        </div>
      </div>
    </button>
  );
}