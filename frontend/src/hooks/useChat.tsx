import { useContext } from 'react';
import { ChatContext } from '../contexts/chatContext';
import type { ChatContextType } from '../contexts/chatContext';

export const useChat = (): ChatContextType => {
  const context = useContext(ChatContext);
  if (context === undefined) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
};