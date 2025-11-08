'use client';

import React from "react";
import NavBar from "../layout/navbar"; // Adjust import path as needed
import "../layout/layoutStyles.css";
import "../index.css"
interface LayoutProps {
  readonly children: React.ReactNode;
  readonly currentPage: string;
  readonly setCurrentPage: (page: string) => void;
}

export default function Layout({ children, currentPage, setCurrentPage }: LayoutProps) {
  return (
    <div className="min-h-screen bg-gray-100 scrollbar-hide no-scrollbar scrollbar-hide">
      <NavBar currentPage={currentPage} setCurrentPage={setCurrentPage} />
      
      {/* Main content area */}
      <main className="mx-auto max-w-7xl">
        {children}
      </main>
      <footer className="bg-gray-800 text-white py-15">
        <div className="max-w-6xl mx-auto px-8 text-center">
          <p className="text-gray-400">
            Contact us: xxxx@gmail.com
          </p>
        </div>
      </footer>
    </div>
  );
}