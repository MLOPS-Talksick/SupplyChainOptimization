import type { NextConfig } from "next";

/** @type {import('next').NextConfig} */
const nextConfig: NextConfig = {
  eslint: {
    // WARNING: this will allow any lint errors to slip through your build
    ignoreDuringBuilds: true,
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `http://34.8.2.47/:path*`,
      },
    ];
  },
};

export default nextConfig;
