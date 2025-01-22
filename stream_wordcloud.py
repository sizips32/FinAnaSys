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
import matplotlib.font_manager as fm

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

# KoNLPy 초기화를 위한 캐시된 함수
@st.cache_resource
def initialize_okt():
    try:
        return Okt()
    except Exception as e:
        st.warning(f"KoNLPy 초기화 중 오류 발생: {str(e)}")
        return None

# konlpy import 예외 처리
try:
    from konlpy.tag import Okt
    USE_KONLPY = True
    okt_instance = initialize_okt()
except ImportError:
    USE_KONLPY = False
    okt_instance = None
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

class TextAnalyzer:
    def __init__(self):
        self.okt = okt_instance  # 전역 인스턴스 사용
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
    
    # matplotlib 전역 폰트 설정
    plt.rcParams['font.family'] = ['NanumGothic', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # TextAnalyzer 인스턴스 생성
    analyzer = TextAnalyzer()
    font_path = analyzer.font_path
    
    if font_path:
        # 폰트 등록
        font_name = fm.FontProperties(fname=font_path).get_name()
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = [font_name, 'sans-serif']
    
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
                if font_path is None:
                    st.warning("한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")
                    wordcloud = WordCloud(width=800, height=400, 
                                       background_color='white').generate(text)
                else:
                    wordcloud = WordCloud(
                        font_path=font_path,
                        width=800, 
                        height=400,
                        background_color='white'
                    ).generate(text)

                # 워드클라우드 출력
                fig_wordcloud = plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(fig_wordcloud)
                plt.close(fig_wordcloud)

                # 단어 빈도 계산
                words = text.split()
                word_counts = Counter(words)
                top_10_words = word_counts.most_common(10)
                
                # 상위 10개 단어 출력 및 그래프 표시를 위한 컬럼 분할
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("상위 10개 단어별 언급 횟수:")
                    for word, count in top_10_words:
                        st.write(f"{word}: {count}")
                
                with col2:
                    # 파이차트 생성
                    words, counts = zip(*top_10_words)
                    fig_pie = plt.figure(figsize=(10, 10))
                    plt.pie(counts, labels=words, autopct='%1.1f%%', 
                           startangle=140, colors=plt.cm.Paired.colors)
                    plt.title('상위 10개 단어별 언급 비율', pad=20)
                    st.pyplot(fig_pie)
                    plt.close(fig_pie)

            except Exception as e:
                st.error(f"워드 클라우드 생성 중 오류가 발생했습니다: {str(e)}")
                # 디버깅을 위한 상세 오류 정보
                st.error("상세 오류 정보:")
                st.error(e)
