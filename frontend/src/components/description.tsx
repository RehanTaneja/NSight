// components/description.tsx
'use client';

interface DescriptionProps {
  readonly text: string;
}

export default function Description({ text }: DescriptionProps) {
  return (
    <div className="flex justify-center">
      <p className="text-gray-700 text-base leading-relaxed wrap-break-words text-center">
        {text}
      </p>
    </div>
  );
}