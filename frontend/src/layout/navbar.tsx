'use client';

import Button from '../components/button'; // Adjust import path as needed

interface NavBarProps {
  readonly currentPage: string;
  readonly setCurrentPage: (page: string) => void;
}

export default function NavBar({ currentPage, setCurrentPage }: NavBarProps) {
  return (
    <nav className="bg-blue-600 shadow-lg py-4">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-20">
          <div className="flex items-center space-x-8">
            {/* Logo/Brand */}
            <div className="flex-shrink-0 flex items-center">
              <span className="text-white text-4xl font-bold">NSight</span>
            </div>
            
            {/* Navigation Buttons */}
            <div className="flex space-x-4">
              <Button
                onClick={() => setCurrentPage('home')}
                className={`
                  inline-flex items-center px-6 py-3 rounded-lg text-lg font-semibold transition-all duration-300
                  ${currentPage === 'home' 
                    ? '!bg-white !text-blue-600 !border-white shadow-md' 
                    : 'bg-white text-blue-600 border-blue-500 hover:bg-blue-500 hover:text-white hover:border-blue-500'
                  }
                `}
              >
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
                </svg>
                Home
              </Button>
              
              <Button
                onClick={() => setCurrentPage('upload')}
                className={`
                  inline-flex items-center px-6 py-3 rounded-lg text-lg font-semibold transition-all duration-300
                  ${currentPage === 'upload' 
                    ? '!bg-white !text-blue-600 !border-white shadow-md' 
                    : 'bg-white text-blue-600 border-blue-500 hover:bg-blue-500 hover:text-white hover:border-blue-500'
                  }
                `}
              >
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                Upload
              </Button>
            </div>
          </div>

          {/* Optional: User menu or additional buttons */}
          <div className="flex items-center space-x-4">
            <Button className="!bg-blue-700 !text-white !border-blue-700 hover:!bg-blue-800 hover:!border-blue-800">
              Settings
            </Button>
          </div>
        </div>
      </div>
    </nav>
  );
}