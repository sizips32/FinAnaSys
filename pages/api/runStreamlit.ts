import type { NextApiRequest, NextApiResponse } from 'next'
import { spawn } from 'child_process'

type Data = {
  success: boolean
  message: string
  port?: number
}

const STREAMLIT_SCRIPTS = {
  technical: 'AI_Technical_Analysis.py',
  portfolio: 'stream_portfolio.py',
  machine: 'stream_machine.py',
  quantum: 'stream_quantum.py',
  volatility: 'stream_volatility.py',
  wordcloud: 'stream_wordcloud.py',
}

const PORT_MAP = {
  technical: 8501,
  portfolio: 8502,
  machine: 8503,
  quantum: 8504,
  volatility: 8505,
  wordcloud: 8506,
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<Data>
) {
  if (req.method !== 'POST') {
    return res.status(405).json({ success: false, message: 'Method not allowed' })
  }

  const { type } = req.body

  if (!type || !STREAMLIT_SCRIPTS[type as keyof typeof STREAMLIT_SCRIPTS]) {
    return res.status(400).json({ success: false, message: 'Invalid analysis type' })
  }

  const scriptPath = `../../python/${STREAMLIT_SCRIPTS[type as keyof typeof STREAMLIT_SCRIPTS]}`
  const port = PORT_MAP[type as keyof typeof PORT_MAP]

  try {
    const streamlit = spawn('streamlit', ['run', scriptPath, '--server.port', port.toString()])

    streamlit.stdout.on('data', (data) => {
      console.log(`Streamlit output: ${data}`)
    })

    streamlit.stderr.on('data', (data) => {
      console.error(`Streamlit error: ${data}`)
    })

    // 프로세스가 시작되면 성공 응답을 보냅니다
    return res.status(200).json({
      success: true,
      message: `Successfully started ${type} analysis`,
      port: port
    })
  } catch (error) {
    console.error('Failed to start Streamlit:', error)
    return res.status(500).json({
      success: false,
      message: 'Failed to start analysis'
    })
  }
} 
