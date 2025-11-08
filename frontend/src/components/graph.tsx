'use client';

import React from "react";

interface GraphProps {
  readonly src: string;          // URL or path to the PNG image
  readonly alt?: string;         // optional alt text
  readonly width?: number;       // optional width
  readonly height?: number;      // optional height
}

export default function Graph({
  src,
  alt = "Graph image",
  width,
  height
}: GraphProps) {
  return (
    <div className="flex justify-center items-center">
      <img
        src={src}
        alt={alt}
        width={width}
        height={height}
        className="rounded-md shadow-md"
      />
    </div>
  );
}
