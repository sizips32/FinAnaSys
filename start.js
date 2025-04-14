const { spawn } = require('child_process');
const path = require('path');

// Python 실행 파일 경로
const pythonFile = path.join(__dirname, 'app.py');

// Python 프로세스 실행
const pythonProcess = spawn('python', [pythonFile]);

// Python 프로세스의 출력을 콘솔에 표시
pythonProcess.stdout.on('data', (data) => {
  console.log(`${data}`);
});

pythonProcess.stderr.on('data', (data) => {
  console.error(`${data}`);
});

pythonProcess.on('close', (code) => {
  console.log(`Python process exited with code ${code}`);
});

// Ctrl+C 시그널 처리
process.on('SIGINT', () => {
  pythonProcess.kill();
  process.exit();
});
