import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import PyPDF2
import markdown
from collections import Counter
import pandas as pd
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import io
import base64
import os

# konlpy import 예외 처리
try:
    from konlpy.tag import Okt
    USE_KONLPY = True
except ImportError:
    st.warning("""
    한글 형태소 분석을 위한 konlpy 패키지가 설치되지 않았습니다.
    터미널에서 다음 명령어를 실행하세요:
    
    pip install konlpy
    pip install JPype1
    
    만약 macOS나 Linux를 사용중이라면:
    brew install java
    또는
    sudo apt-get install default-jdk python3-dev
    """)
    USE_KONLPY = False

# NLTK 리소스 초기화
@st.cache_resource
def initialize_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('stopwords')

initialize_nltk()

class TextAnalyzer:
    def __init__(self):
        self.okt = Okt() if USE_KONLPY else None
        self.default_words = {
            '분석': 5, '결과': 4, '데이터': 3, '텍스트': 2, '단어': 1
        }
        self.font_paths = [
            'font/NanumGothic.ttf',
            '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
            '/System/Library/Fonts/AppleGothic.ttf',
            'C:/Windows/Fonts/malgun.ttf',
            '/Users/soonjaekim/Desktop/AGIs/나눔 글꼴/나눔명조/NanumFontSetup_OTF_MYUNGJO/NanumMyeongjo.otf'
        ]
        self.font_path = self._get_valid_font_path()

    def _get_valid_font_path(self):
        for path in self.font_paths:
            if os.path.exists(path):
                return path
        return None

def load_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        st.error(f"PDF 파일 처리 중 오류 발생: {str(e)}")
        return ""

def load_md(file):
    try:
        content = file.read().decode("utf-8")
        return markdown.markdown(content)
    except Exception as e:
        st.error(f"Markdown 파일 처리 중 오류 발생: {str(e)}")
        return ""

def load_text_file(file):
    try:
        return file.read().decode('utf-8')
    except UnicodeDecodeError:
        try:
            file.seek(0)
            return file.read().decode('cp949')
        except:
            file.seek(0)
            return file.read().decode('utf-8', errors='ignore')

# 필요한 함수들을 export
__all__ = ['TextAnalyzer', 'load_pdf', 'load_md', 'load_text_file']

if __name__ == '__main__':
    st.title("Word Cloud Generator")
    
    uploaded_file = st.file_uploader("텍스트 파일 업로드 (.txt, .md, .pdf)", type=["txt", "md", "pdf"])
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".md"):
            text = load_md(uploaded_file)
        elif uploaded_file.name.endswith(".pdf"):
            text = load_pdf(uploaded_file)
        elif uploaded_file.name.endswith(".txt"):
            text = load_text_file(uploaded_file)

        if text:
            try:
                # 폰트 경로 설정
                font_paths = [
                    './font/NanumGothic.ttf',
                    '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
                    '/System/Library/Fonts/AppleGothic.ttf',
                    'C:/Windows/Fonts/malgun.ttf',
                    '/Users/soonjaekim/Desktop/AGIs/나눔 글꼴/나눔명조/NanumFontSetup_OTF_MYUNGJO/NanumMyeongjo.otf'
                ]

                # 사용 가능한 폰트 찾기
                font_path = None
                for path in font_paths:
                    if os.path.exists(path):
                        font_path = path
                        break

                if font_path is None:
                    # 폰트를 찾지 못한 경우 기본 폰트 사용
                    wordcloud = WordCloud(width=800, height=400, 
                                       background_color='white').generate(text)
                    st.warning("한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")
                else:
                    wordcloud = WordCloud(
                        font_path=font_path,
                        width=800, 
                        height=400,
                        background_color='white'
                    ).generate(text)

                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)

            except Exception as e:
                st.error(f"워드 클라우드 생성 중 오류가 발생했습니다: {str(e)}")
                # 디버깅을 위한 상세 오류 정보
                st.error("상세 오류 정보:")
                st.error(e)
