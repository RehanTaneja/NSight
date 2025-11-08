'use client';

import React from "react";

interface DescriptionProps {
  readonly text: string;
}

export default function Description({ text }: DescriptionProps) {
  return (
    <p
      className={`
        text-gray-700 text-base leading-relaxed
        break-words
      `}
    >
      {text}
    </p>
  );
}
