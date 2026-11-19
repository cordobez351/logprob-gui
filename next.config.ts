import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  basePath: "/logprob-gui",
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
