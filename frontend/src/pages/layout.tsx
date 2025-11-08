'use client';

import React from "react";

export default function Layout({ children }: { readonly children: React.ReactNode}){
  return (
    <div className="flex min-h-screen bg-gray-100">
      {/* Main content area */}
      <main className="flex-1 py-6 px-4 overflow-auto">
        {children}
      </main>
    </div>
  );
}