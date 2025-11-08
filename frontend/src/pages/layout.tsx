'use client';

import React from "react";

export default function Layout({ children }: { readonly children: React.ReactNode}){
  return (
    <html lang="en" className="scrollbar-hide">
      <body className="">
        <div className="flex min-h-screen">
          <main className="flex-1 bg-gray-100 py-6 overflow-auto">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}