import React from 'react';
import ReactMarkdown from 'react-markdown';

// Define the props for the MarkdownViewer component
interface MarkdownViewerProps {
  markdownContent: string;
}

const MarkdownViewer: React.FC<MarkdownViewerProps> = ({ markdownContent }) => {
  return (
    <div>
      <h1>Markdown Viewer</h1>
      <ReactMarkdown>{markdownContent}</ReactMarkdown>
    </div>
  );
};

export default MarkdownViewer;
