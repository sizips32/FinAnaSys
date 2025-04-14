/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  swcMinify: true,
  reactStrictMode: true,
  poweredByHeader: false,
  
  images: {
    domains: [], // 외부 이미지 도메인이 필요한 경우 여기에 추가
    remotePatterns: [] // 필요한 경우 원격 이미지 패턴 설정
  }
}

module.exports = nextConfig
