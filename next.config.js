/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  swcMinify: true,
  reactStrictMode: true,
  poweredByHeader: false,
  
  // 앱 아이콘 설정
  images: {
    favicon: '/icon.svg',
    icon: '/icon.svg',
    apple: '/icon.svg'
  }
}

module.exports = nextConfig
