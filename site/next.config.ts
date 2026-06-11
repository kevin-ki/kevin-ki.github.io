import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  images: { unoptimized: true },
  experimental: {
    optimizePackageImports: ["@lobehub/icons"],
  },
};

export default nextConfig;
