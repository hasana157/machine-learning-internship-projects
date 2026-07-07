/** @type {import('next').NextConfig} */
const nextConfig = {
  // React strict mode for development
  reactStrictMode: true,

  // Image optimization
  images: {
    domains: [
      'image.tmdb.org',
      'm.media-amazon.com',
      'localhost',
      'localhost:8000',
    ],
    formats: ['image/webp', 'image/avif'],
  },

  // Environment variables
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
    NEXT_PUBLIC_GRADIO_URL: process.env.NEXT_PUBLIC_GRADIO_URL || 'http://localhost:7860',
  },

  // Custom headers
  async headers() {
    return [
      {
        source: '/api/:path*',
        headers: [
          {
            key: 'Access-Control-Allow-Credentials',
            value: 'true',
          },
          {
            key: 'Access-Control-Allow-Origin',
            value: '*',
          },
          {
            key: 'Access-Control-Allow-Methods',
            value: 'GET,OPTIONS,PATCH,DELETE,POST,PUT',
          },
          {
            key: 'Access-Control-Allow-Headers',
            value: 'X-CSRF-Token, X-Forwarded-Host, X-URL-SCHEME, x-requested-with, Content-Type, Authorization',
          },
        ],
      },
    ];
  },

  // Rewrites for API proxy
  async rewrites() {
    return {
      beforeFiles: [
        {
          source: '/api/:path*',
          destination: `${process.env.NEXT_PUBLIC_API_URL}/:path*`,
        },
      ],
    };
  },

  // Compression and performance
  compress: true,
  swcMinify: true,

  // TypeScript
  typescript: {
    ignoreBuildErrors: false,
  },

  // Webpack config
  webpack: (config) => {
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false,
      net: false,
      tls: false,
    };
    return config;
  },
};

module.exports = nextConfig;
