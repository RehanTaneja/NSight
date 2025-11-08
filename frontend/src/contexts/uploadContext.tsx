import { createContext } from 'react';

export interface UploadResult {
  waterfall: string;
  bar: string;
  modelFilename: string;  // Remove optional
  dataFilename: string;   // Remove optional
}

export interface UploadContextType {
  uploadResult: UploadResult | null;
  setUploadResult: (result: UploadResult | null) => void;
  isUploading: boolean;
  setIsUploading: (uploading: boolean) => void;
}

export const UploadContext = createContext<UploadContextType | undefined>(undefined);