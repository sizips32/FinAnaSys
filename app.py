from flask import Flask, render_template, request, jsonify, url_for
import subprocess
import sys
import time
import os
import webbrowser
from threading import Timer
import logging
import psutil  # 추가된 모듈
import PyPDF2  # PyPDF2 임포트 추가
import markdown  # markdown 임포트 추가
from analysis import (
    perform_dividend_analysis,
    perform_portfolio_optimization,
    perform_quantum_analysis,
    perform_volatility_analysis,
    perform_technical_analysis
)
from word_cloud import TextAnalyzer  # TextAnalyzer 임포트 추가
import requests

app = Flask(__name__,
            static_url_path='/static',
            static_folder='static',
            template_folder='templates')

from flask_cors import CORS  # 추가할 코드
CORS(app)  # CORS 설정 활성화

# 로깅 설정
if __name__ != '__main__':
    app.logger.disabled = True
    log = logging.getLogger('werkzeug')
    log.disabled = True
else:
    logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)s - %(levelname)s - %(message)s')

# 분석 타입 설정
ANALYSIS_TYPES = {
    'dividend': {'port': 8501, 'script': 'stream_dividend.py'},
    'portoptima': {'port': 8502, 'script': 'stream_portfolio.py'},
    'quantum': {'port': 8503, 'script': 'stream_quantum.py'},
    'voltrade': {'port': 8504, 'script': 'stream_volatility.py'},
    'technical': {'port': 8505, 'script': 'AI_Technical_Analysis.py'},
    'wordcloud': {'port': 8506, 'script': 'word_cloud.py'}
}

def open_browser():
    """메인 페이지용 브라우저 오프너"""
    webbrowser.open_new('http://localhost:5050')


def open_analysis_page(type):
    """분석 페이지용 브라우저 오프너"""
    ports = {
        'dividend': 8501,
        'portoptima': 8502,
        'quantum': 8503,
        'voltrade': 8504,
        'technical': 8505,
        'wordcloud': 8506  # word cloud analysis port 추가
    }
    if type in ports:
        webbrowser.open(f'http://localhost:{ports[type]}')
        return True
    return False


def kill_process_on_port(port):
    """지정된 포트의 프로세스 종료"""
    try:
        if sys.platform.startswith('win'):
            subprocess.run(['taskkill', '/F', '/PID', 
                          f"$(netstat -ano | findstr :{port} | awk '{{print $5}}'"],
                         shell=True)
        else:
            subprocess.run(['kill', '-9', 
                          f"$(lsof -t -i:{port})"], shell=True)
    except Exception as e:
        logging.error(f"프로세스 종료 실패 (포트 {port}): {e}")


@app.route('/')
def index():
    return render_template('index.html')


def wait_for_streamlit(port, timeout=30):
    """Streamlit 서버가 준비될 때까지 대기"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f'http://localhost:{port}')
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            time.sleep(1)
    return False


@app.route('/run/<analysis_type>', methods=['POST'])
def run_analysis(analysis_type):
    try:
        if analysis_type not in ANALYSIS_TYPES:
            return jsonify({
                'success': False,
                'error': '지원하지 않는 분석 유형입니다.'
            })

        details = ANALYSIS_TYPES[analysis_type]
        streamlit_script = details['script']
        port = details['port']
        
        # 기존 프로세스 종료
        kill_process_on_port(port)
        
        logging.info(f"Streamlit 실행 준비: {streamlit_script} (포트: {port})")
        
        process = subprocess.Popen([
            sys.executable, '-m', 'streamlit', 'run',
            streamlit_script,
            '--server.port', str(port),
            '--server.address', 'localhost'
        ])
        
        # 서버 준비 상태 확인
        if not wait_for_streamlit(port):
            process.kill()
            logging.error("Streamlit 서버 시작 시간 초과")
            return jsonify({
                'success': False,
                'error': '분석 서버 시작에 실패했습니다. 잠시 후 다시 시도해주세요.'
            })
        
        if process.poll() is not None:
            logging.error("Streamlit 프로세스가 비정상 종료되었습니다.")
            return jsonify({
                'success': False,
                'error': '분석 서버가 비정상 종료되었습니다.'
            })
        
        logging.info(f"분석 시작 성공: {analysis_type}")
        
        return jsonify({
            'success': True,
            'url': f'http://localhost:{port}',
            'message': '분석이 시작되었습니다. 새 탭이 열립니다.'
        })

    except Exception as e:
        logging.error(f"분석 실행 실패: {str(e)}")
        return jsonify({
            'success': False,
            'error': '분석을 시작할 수 없습니다. 시스템 관리자에게 문의하세요.'
        })


@app.route('/analyze', methods=['POST'])
def analyze_text():
    try:
        if 'file' not in request.files:
            text = request.form.get('text', '')
            if not text:
                return jsonify({
                    'success': False,
                    'error': '분석할 텍스트를 입력해주세요.'
                })
        else:
            file = request.files['file']
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': '파일을 선택해주세요.'
                })
            
            try:
                # 파일 내용 읽기
                if file.filename.endswith('.pdf'):
                    text = load_pdf(file)
                elif file.filename.endswith('.md'):
                    text = load_md(file)
                else:
                    text = file.read().decode('utf-8')
            except UnicodeDecodeError:
                return jsonify({
                    'success': False,
                    'error': '파일 인코딩이 올바르지 않습니다. UTF-8 형식의 텍스트 파일을 사용해주세요.'
                })
            except Exception as e:
                logging.error(f"파일 읽기 오류: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': '파일을 읽는 중 오류가 발생했습니다. 파일이 손상되었거나 지원하지 않는 형식일 수 있습니다.'
                })
        
        # 텍스트 분석 수행
        try:
            analyzer = TextAnalyzer()
            wordcloud_img, top_words = analyzer.analyze_text(text)
            
            if not wordcloud_img:
                return jsonify({
                    'success': False,
                    'error': '분석할 텍스트가 충분하지 않습니다. 더 많은 텍스트를 입력해주세요.'
                })
            
            return jsonify({
                'success': True,
                'wordcloud': wordcloud_img,
                'top_words': top_words
            })
        except Exception as e:
            logging.error(f"텍스트 분석 오류: {str(e)}")
            return jsonify({
                'success': False,
                'error': '텍스트 분석 중 오류가 발생했습니다. 입력된 텍스트를 확인해주세요.'
            })
            
    except Exception as e:
        logging.error(f"요청 처리 중 오류 발생: {str(e)}")
        return jsonify({
            'success': False,
            'error': '요청을 처리할 수 없습니다. 잠시 후 다시 시도해주세요.'
        })

def load_md(file):
    content = file.read().decode("utf-8")
    return markdown.markdown(content)

def load_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


import socket

def is_port_in_use(port):
    """포트 사용 여부 확인"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def free_port(port):
    """포트 강제 종료"""
    try:
        for proc in psutil.process_iter(['pid', 'connections']):
            for conn in proc.info['connections']:
                if conn.status == 'LISTEN' and conn.laddr.port == port:
                    proc.kill()
                    logging.info(f"포트 {port}를 사용 중인 프로세스를 종료했습니다.")
    except Exception as e:
        logging.warning(f"포트 {port} 종료 실패: {e}")

if __name__ == '__main__':
    Timer(1.0, open_browser).start()
    PORT = 5050

    # 포트 점유 해제 처리
    if is_port_in_use(PORT):
        logging.warning(f"포트 {PORT}가 사용 중입니다. 프로세스를 종료합니다.")
        free_port(PORT)

# 서버 실행
    app.run(
        host='0.0.0.0',
        port=PORT,
        debug=False,
        use_reloader=False
    )

import streamlit as st
import os
import sys

# 현재 디렉토리의 경로를 시스템 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from word_cloud import TextAnalyzer

# 나머지 app.py 코드
# ...existing code...
