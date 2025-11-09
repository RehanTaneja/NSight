'use client';

import React, { useState, useMemo, useCallback, type ReactNode } from 'react';
import { ChatContext, type ChatContextType, type ChatMessage, type AnalysisData } from './chatContext';

export const ChatProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState<string | null>(null);

  // Memoized setter functions
  const memoizedSetIsChatOpen = useCallback((open: boolean) => {
    setIsChatOpen(open);
  }, []);

  const memoizedSetIsLoading = useCallback((loading: boolean) => {
    setIsLoading(loading);
  }, []);

  // Chat actions
  const sendMessage = useCallback(async (message: string, analysisData: AnalysisData): Promise<void> => {
    const userMessage: ChatMessage = {
      id: Date.now(),
      content: message,
      role: 'user',
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    memoizedSetIsLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message,
          analysisData,
          conversationId
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to get response');
      }

      const assistantMessage: ChatMessage = {
        id: Date.now() + 1,
        content: data.response,
        role: 'assistant',
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev, assistantMessage]);
      
      if (data.conversationId) {
        setConversationId(data.conversationId);
      }

    } catch (error: unknown) {
      // Handle the error properly instead of leaving it unused
      console.error('Chat API error:', error);
      
      const errorMessage: ChatMessage = {
        id: Date.now() + 1,
        content: 'Sorry, I encountered an error. Please try again.',
        role: 'assistant',
        isError: true,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      memoizedSetIsLoading(false);
    }
  }, [conversationId, memoizedSetIsLoading]);

  const clearChat = useCallback((): void => {
    setMessages([]);
    setConversationId(null);
  }, []);

  const openChat = useCallback((): void => {
    memoizedSetIsChatOpen(true);
  }, [memoizedSetIsChatOpen]);

  const closeChat = useCallback((): void => {
    memoizedSetIsChatOpen(false);
  }, [memoizedSetIsChatOpen]);

  // Memoize the context value
  const contextValue = useMemo((): ChatContextType => ({
    isChatOpen,
    messages,
    isLoading,
    sendMessage,
    clearChat,
    openChat,
    closeChat
  }), [
    isChatOpen,
    messages,
    isLoading,
    sendMessage,
    clearChat,
    openChat,
    closeChat
  ]);

  return (
    <ChatContext.Provider value={contextValue}>
      {children}
    </ChatContext.Provider>
  );
};