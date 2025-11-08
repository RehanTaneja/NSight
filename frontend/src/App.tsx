import { useState } from "react";
import Layout from "./pages/layout";
import HomePage from "./pages/home";
import UploadPage from "./pages/uploadpage"

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
    <Layout currentPage={currentPage} setCurrentPage={setCurrentPage}>
      {renderPage()}
    </Layout>
  );
}