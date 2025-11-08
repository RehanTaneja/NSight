'use client';

import React from 'react';

interface ButtonProps {
  readonly children: React.ReactNode;
  readonly onClick?: () => void;
  readonly type?: 'button' | 'submit' | 'reset';
  readonly disabled?: boolean;
  readonly className?: string;
}

export default function Button({ 
  children, 
  onClick, 
  type = 'button', 
  disabled = false,
  className = ''
}: ButtonProps) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`
        bg-white text-blue-600 border-2 border-blue-500 
        rounded-xl px-6 py-3 font-semibold text-lg
        transition-all duration-300 ease-in-out
        hover:bg-blue-500 hover:text-white
        focus:outline-none focus:ring-4 focus:ring-blue-200
        active:bg-blue-600 active:transform active:scale-95
        disabled:bg-gray-100 disabled:text-gray-400 disabled:border-gray-300 disabled:cursor-not-allowed
        ${className}
      `}
    >
      {children}
    </button>
  );
}