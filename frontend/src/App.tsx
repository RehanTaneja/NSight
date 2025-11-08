import { useState } from "react";
import Layout from "./pages/layout";
import HomePage from "./pages/home";
import UploadPage from "./pages/uploadpage";
import { UploadProvider } from "./contexts/uploadProvider"; 
export default function App() {
  const [currentPage, setCurrentPage] = useState('home');

  const renderPage = () => {
    switch (currentPage) {
      case 'home':
        return <HomePage />;
      case 'upload':
        return <UploadPage />;
      default:
        return <HomePage />;
    }
  };

return (
    <div className="[scrollbar-width:none] [-ms-overflow-style:none] [-webkit-overflow-scrolling:touch]">
      {/* Webkit requires custom CSS */}
      <style>
        {`::-webkit-scrollbar { display: none; }`}
      </style>
      
      <UploadProvider>
        <Layout currentPage={currentPage} setCurrentPage={setCurrentPage}>
          {renderPage()}
        </Layout>
      </UploadProvider>
    </div>
  );
}