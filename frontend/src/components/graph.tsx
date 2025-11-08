'use client';

interface GraphProps {
  readonly src: string;
  readonly alt?: string;
  readonly width?: number;
  readonly height?: number;
  readonly className?: string;
  readonly containerClassName?: string;
}

export default function Graph({
  src,
  alt = "Graph image",
  width,
  height,
  className = "",
  containerClassName = "flex justify-center items-center"
}: GraphProps) {
  return (
    <div className={containerClassName}>
      <img
        src={src}
        alt={alt}
        width={width}
        height={height}
        className={`rounded-md ${className}`}
      />
    </div>
  );
}