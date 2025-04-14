import { exec } from 'child_process';
import { promisify } from 'util';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const execAsync = promisify(exec);

const ANALYSIS_CONFIG = {
  machine: {
    script: 'stream_machine.py',
    port: 8507
  },
  dividend: {
    script: 'stream_dividend.py',
    port: 8501
  },
  portfolio: {
    script: 'stream_portfolio.py',
    port: 8502
  },
  quantum: {
    script: 'stream_quantum.py',
    port: 8503
  },
  volatility: {
    script: 'stream_volatility.py',
    port: 8504
  },
  technical: {
    script: 'AI_Technical_Analysis.py',
    port: 8505
  },
  wordcloud: {
    script: 'stream_wordcloud.py',
    port: 8506
  }
};

async function killProcessOnPort(port) {
  try {
    if (process.platform === 'win32') {
      await execAsync(`netstat -ano | findstr :${port} | findstr LISTENING`);
    } else {
      await execAsync(`lsof -i :${port} | grep LISTEN | awk '{print $2}' | xargs kill -9`);
    }
  } catch (error) {
    // 프로세스가 없는 경우 무시
    console.log(`No process found on port ${port}`);
  }
}

async function checkFileExists(filePath) {
  try {
    await fs.promises.access(filePath, fs.constants.F_OK);
    return true;
  } catch {
    return false;
  }
}

export default async function handler(req, res) {
  const { type } = req.query;

  if (req.method === 'POST') {
    try {
      const config = ANALYSIS_CONFIG[type];
      
      if (!config) {
        return res.status(400).json({ 
          success: false,
          error: '지원하지 않는 분석 타입입니다.' 
        });
      }

      // 스크립트 파일 경로 확인
      const scriptPath = path.join(process.cwd(), config.script);
      const fileExists = await checkFileExists(scriptPath);

      if (!fileExists) {
        return res.status(404).json({
          success: false,
          error: `분석 스크립트를 찾을 수 없습니다: ${config.script}`,
          details: '시스템 관리자에게 문의하세요.'
        });
      }

      // 기존 프로세스 종료
      await killProcessOnPort(config.port);

      // Streamlit 실행
      const command = `streamlit run ${config.script} --server.port ${config.port} --server.address localhost`;
      const { stdout, stderr } = await execAsync(command, {
        detached: true,
        cwd: process.cwd()
      });

      if (stderr && stderr.includes('Error:')) {
        throw new Error(stderr);
      }

      // 프로세스 시작 대기
      await new Promise(resolve => setTimeout(resolve, 3000));

      res.status(200).json({ 
        success: true, 
        message: `${type} 분석이 시작되었습니다.`,
        url: `http://localhost:${config.port}`
      });
    } catch (error) {
      console.error('Analysis Error:', error);
      
      let errorMessage = '분석 시작 중 오류가 발생했습니다.';
      let errorDetails = error.message;

      if (error.message.includes('ENOENT')) {
        errorMessage = 'Streamlit이 설치되어 있지 않습니다.';
        errorDetails = 'pip install streamlit 명령어로 Streamlit을 설치해주세요.';
      } else if (error.message.includes('EADDRINUSE')) {
        errorMessage = '포트가 이미 사용 중입니다.';
        errorDetails = '잠시 후 다시 시도해주세요.';
      }

      res.status(500).json({ 
        success: false,
        error: errorMessage,
        details: errorDetails
      });
    }
  } else {
    res.status(405).json({ 
      success: false,
      error: '허용되지 않는 메소드입니다.' 
    });
  }
} 
