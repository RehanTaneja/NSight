import { createContext } from 'react';

export interface UploadResult {
  waterfall: string;
  bar: string;
  summary?: string; // Add this line
  modelFilename?: string;
  dataFilename?: string;
}

export interface UploadContextType {
  uploadResult: UploadResult | null;
  setUploadResult: (result: UploadResult | null) => void;
  isUploading: boolean;
  setIsUploading: (uploading: boolean) => void;
}

export const UploadContext = createContext<UploadContextType | undefined>(undefined);