import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export default async function handler(req, res) {
  const { type } = req.query;

  if (req.method === 'POST') {
    try {
      if (type === 'machine') {
        // Streamlit 실행
        await execAsync('streamlit run stream_machine.py', { detached: true });
        res.status(200).json({ 
          success: true, 
          message: 'Streamlit 앱이 실행되었습니다.',
          url: 'http://localhost:8501'
        });
      } else if (type === 'wordcloud') {
        // 워드 클라우드 Streamlit 실행
        await execAsync('streamlit run word_cloud.py', { detached: true });
        res.status(200).json({ 
          success: true, 
          message: '워드 클라우드 분석이 시작되었습니다.',
          url: 'http://localhost:8501'
        });
      } else {
        // 다른 분석 타입들에 대한 처리
        res.status(200).json({ message: `${type} 분석이 시작되었습니다.` });
      }
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  } else {
    res.status(405).json({ error: 'Method not allowed' });
  }
} 
