'use client';

import React from "react";

interface LayoutProps {
  readonly children: React.ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  return (
    <div className="flex min-h-screen bg-gray-100">
      {/* Main content area */}
      <main className="flex-1 py-6 px-6 md:px-12 overflow-auto">
        {children}
      </main>
    </div>
  );
}
