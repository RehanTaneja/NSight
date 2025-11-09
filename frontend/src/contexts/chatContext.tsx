import { createContext } from 'react';

export interface ChatMessage {
  id: number;
  content: string;
  role: 'user' | 'assistant';
  timestamp: string;
  isError?: boolean;
}

// Update AnalysisData to match UploadResult structure
export interface AnalysisData {
  waterfall: string;
  bar: string;
  summary?: string;
  modelFilename?: string;
  dataFilename?: string;
  // Remove the graphs nested structure since UploadResult has waterfall/bar directly
}

export interface ChatContextType {
  isChatOpen: boolean;
  messages: ChatMessage[];
  isLoading: boolean;
  sendMessage: (message: string, analysisData: AnalysisData) => Promise<void>;
  clearChat: () => void;
  openChat: () => void;
  closeChat: () => void;
}

export const ChatContext = createContext<ChatContextType | undefined>(undefined);