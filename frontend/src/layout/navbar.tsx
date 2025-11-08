'use client';

import React from 'react';

export default function Navbar() {
  return (
    <nav className="bg-blue-600 text-white shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center">
            <a href="/" className="text-xl font-bold">
              MyBrand
            </a>
          </div>

          {/* Navigation Links */}
          <div className="flex items-center space-x-8">
            <a href="#" className="hover:text-blue-200 transition">
              Home
            </a>
            <a href="#" className="hover:text-blue-200 transition">
              About
            </a>
            <a href="#" className="hover:text-blue-200 transition">
              Services
            </a>
            <a href="#" className="hover:text-blue-200 transition">
              Contact
            </a>
            <button className="bg-white text-blue-600 px-4 py-2 rounded-lg font-semibold hover:bg-blue-50 transition">
              Get Started
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
}