import { spawn } from 'child_process';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Method not allowed' });
  }

  const { type } = req.query;
  
  // Python 스크립트 매핑
  const scriptMap = {
    'wordcloud': 'word_cloud.py',
    'dividend': 'stream_dividend.py',
    'portfolio': 'stream_portfolio.py',
    'quantum': 'stream_quantum.py',
    'volatility': 'stream_volatility.py',
    'technical': 'AI_Technical_Analysis.py'
  };

  const script = scriptMap[type];
  if (!script) {
    return res.status(400).json({ message: '잘못된 분석 유형입니다.' });
  }

  try {
    // 이미 실행 중인 스트림릿 프로세스 확인
    const checkPort = spawn('lsof', ['-i', ':8501']);
    
    checkPort.stdout.on('data', (data) => {
      // 이미 실행 중인 프로세스가 있다면 종료
      if (data.toString().includes('streamlit')) {
        const pid = data.toString().split(/\s+/)[1];
        process.kill(parseInt(pid));
      }
    });

    checkPort.on('close', () => {
      // 새로운 스트림릿 프로세스 시작
      const streamlit = spawn('streamlit', ['run', script], {
        detached: true,
        stdio: 'ignore'
      });
      
      streamlit.unref();

      // 응답 전송 (URL 제거)
      res.status(200).json({
        message: '분석이 시작되었습니다.'
      });
    });
  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ message: '분석 실행 중 오류가 발생했습니다.' });
  }
} 
