import type { NextApiRequest, NextApiResponse } from 'next';
import { spawn } from 'child_process';
import path from 'path';

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Method not allowed' });
  }

  const { moduleId } = req.body;

  if (!moduleId) {
    return res.status(400).json({ message: 'Module ID is required' });
  }

  // Map module IDs to their corresponding Streamlit files
  const streamlitFiles: { [key: string]: string } = {
    technical: 'technical_analysis.py',
    dividend: 'dividend_analysis.py',
    machine: 'machine_learning.py',
    portfolio: 'portfolio_analysis.py',
    quantum: 'quantum_analysis.py',
    volatility: 'volatility_analysis.py',
    wordcloud: 'wordcloud_analysis.py',
  };

  const fileName = streamlitFiles[moduleId];
  if (!fileName) {
    return res.status(400).json({ message: 'Invalid module ID' });
  }

  try {
    // Run the Streamlit process
    const streamlit = spawn('streamlit', ['run', fileName], {
      detached: true, // 프로세스를 부모로부터 분리
      stdio: 'ignore' // 표준 입출력 무시
    });
    
    // 부모 프로세스와 분리
    streamlit.unref();

    // 즉시 성공 응답 반환
    res.status(200).json({
      message: 'Streamlit app started successfully',
      success: true
    });
  } catch (error) {
    console.error('Error running Streamlit:', error);
    res.status(500).json({ message: 'Failed to start Streamlit app' });
  }
} 
