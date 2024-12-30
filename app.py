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


def open_browser():
    """메인 페이지용 브라우저 오프너"""
    webbrowser.open_new('http://localhost:5000')


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
    """포트를 점유 중인 프로세스 종료"""
    try:
        for proc in psutil.process_iter(['pid', 'connections']):
            for conn in proc.info['connections']:
                if conn.status == 'LISTEN' and conn.laddr.port == port:  # 'LISTEN' 상태만 종료
                    proc.kill()
                    logging.info(f"포트 {port}의 프로세스를 종료했습니다.")
    except Exception as e:
        logging.warning(f"포트 {port} 종료 실패: {e}")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/run/<analysis_type>', methods=['POST'])
def run_analysis(analysis_type):
    try:
        # 분석 매핑 수정
        analysis_mapping = {
            'dividend': {'port': 8501, 'file': 'stream_dividend.py'},
            'portoptima': {'port': 8502, 'file': 'stream_portfolio.py'},
            'quantum': {'port': 8503, 'file': 'stream_quantum.py'},
            'voltrade': {'port': 8504, 'file': 'stream_volatility.py'},
            'technical': {'port': 8505, 'file': 'AI_Technical_Analysis.py'},
            'wordcloud': {'port': 8506, 'file': 'word_cloud.py'}
        }

        # 분석 타입 검증 및 로깅 추가
        if analysis_type == 'wordcloud':
            logging.info(f"워드클라우드 분석 시작: {analysis_type}")
            
        details = analysis_mapping.get(analysis_type)
        if not details:
            logging.error(f"지원하지 않는 분석 유형: {analysis_type}")
            return jsonify({
                'success': False,
                'error': f'지원하지 않는 분석 유형입니다: {analysis_type}'
            })

        current_dir = os.path.dirname(os.path.abspath(__file__))
        streamlit_script = os.path.join(current_dir, details['file'])
        
        # 상세 로깅 추가
        logging.info(f"스크립트 경로: {streamlit_script}")
        logging.info(f"현재 디렉토리: {current_dir}")

        if not os.path.isfile(streamlit_script):
            logging.error(f"파일을 찾을 수 없음: {streamlit_script}")
            return jsonify({
                'success': False,
                'error': f'분석 스크립트를 찾을 수 없습니다: {details["file"]}'
            })

        port = details['port']
        kill_process_on_port(port)

        # Streamlit 프로세스 시작 전 로깅
        logging.info(f"Streamlit 실행 준비: {streamlit_script} (포트: {port})")
        
        process = subprocess.Popen([
            sys.executable, '-m', 'streamlit', 'run',
            streamlit_script,
            '--server.port', str(port),
            '--server.address', 'localhost'
        ])

        # 서버 준비 시간 증가 및 상태 확인
        time.sleep(3)
        
        if process.poll() is not None:
            logging.error("Streamlit 프로세스가 비정상 종료되었습니다.")
            return jsonify({
                'success': False,
                'error': 'Streamlit 실행 실패'
            })

        # 성공 로깅
        logging.info(f"분석 시작 성공: {analysis_type}")
        
        return jsonify({
            'success': True,
            'url': f'http://localhost:{port}',
            'message': f'분석이 시작되었습니다. 새 탭이 열립니다.'
        })

    except Exception as e:
        logging.error(f"분석 실행 실패: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/analyze', methods=['POST'])
def analyze_text():
    try:
        if 'file' not in request.files:
            text = request.form.get('text', '')
            if not text:
                return jsonify({
                    'success': False,
                    'error': '분석할 텍스트가 없습니다.'
                })
        else:
            file = request.files['file']
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': '파일이 선택되지 않았습니다.'
                })
            
            # 파일 내용 읽기
            if file.filename.endswith('.pdf'):
                text = load_pdf(file)
            elif file.filename.endswith('.md'):
                text = load_md(file)
            else:
                text = file.read().decode('utf-8')
        
        # 텍스트 분석 수행
        analyzer = TextAnalyzer()
        wordcloud_img, top_words = analyzer.analyze_text(text)  # 수정된 반환값 처리
        
        if not wordcloud_img:
            return jsonify({
                'success': False,
                'error': '워드클라우드 생성에 실패했습니다.'
            })
        
        return jsonify({
            'success': True,
            'wordcloud': wordcloud_img,  # 이미지 데이터
            'top_words': top_words  # 상위 단어 목록
        })
            
    except Exception as e:
        logging.error(f"요청 처리 중 오류 발생: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'요청 처리 중 오류가 발생했습니다: {str(e)}'
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
