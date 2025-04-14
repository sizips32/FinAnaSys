import { Html, Head, Main, NextScript } from 'next/document'

export default function Document() {
  return (
    <Html lang="ko">
      <Head>
        {/* SVG 파비콘 - 모든 크기에 대응 */}
        <link rel="icon" type="image/svg+xml" href="/icon.svg" />
        <link rel="apple-touch-icon" href="/icon.svg" />
        <link rel="mask-icon" href="/icon.svg" color="#FFC0CB" />
        
        {/* PWA 매니페스트 */}
        <link rel="manifest" href="/manifest.json" />
        
        {/* 브라우저 테마 */}
        <meta name="theme-color" content="#FFC0CB" />
        <meta name="msapplication-TileColor" content="#FFC0CB" />
      </Head>
      <body>
        <Main />
        <NextScript />
      </body>
    </Html>
  )
} 
