import type { NextConfig } from "next";

/** @type {import('next').NextConfig} */
const nextConfig: NextConfig = {
  eslint: {
    // WARNING: this will allow any lint errors to slip through your build
    ignoreDuringBuilds: true,
  },
};

export default nextConfig;
