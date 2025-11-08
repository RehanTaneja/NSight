'use client';

import React, { useState, useMemo, useCallback, type ReactNode } from 'react';
import { UploadContext, type UploadContextType } from './uploadContext';

export const UploadProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [uploadResult, setUploadResult] = useState<UploadContextType['uploadResult']>(null);
  const [isUploading, setIsUploading] = useState(false);

  // Memoize the setter functions
  const memoizedSetUploadResult = useCallback((result: UploadContextType['uploadResult']) => {
    setUploadResult(result);
  }, []);

  const memoizedSetIsUploading = useCallback((uploading: boolean) => {
    setIsUploading(uploading);
  }, []);

  // Memoize the context value
  const contextValue = useMemo((): UploadContextType => ({
    uploadResult,
    setUploadResult: memoizedSetUploadResult,
    isUploading,
    setIsUploading: memoizedSetIsUploading
  }), [uploadResult, isUploading, memoizedSetUploadResult, memoizedSetIsUploading]);

  return (
    <UploadContext.Provider value={contextValue}>
      {children}
    </UploadContext.Provider>
  );
};