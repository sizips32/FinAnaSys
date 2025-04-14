const analysisTools = {
    dividend: {
        title: '배당금 분석',
        placeholder: '분석할 주식 코드를 입력하세요 (예: 005930)',
    },
    portoptima: {
        title: '포트폴리오 최적화',
        placeholder: '포트폴리오 구성 종목을 쉼표로 구분하여 입력하세요',
    },
    quantum: {
        title: '퀀텀 모델',
        placeholder: '분석 파라미터를 입력하세요',
    },
    voltrade: {
        title: '변동성 분석',
        placeholder: '분석할 종목 코드를 입력하세요',
    },
    technical: {
        title: '기술적 분석',
        placeholder: '분석할 종목 정보를 입력하세요',
    }
};

async function runAnalysis(type) {
    const inputElement = document.getElementById(`${type}-input`);
    const outputElement = document.getElementById(`${type}-output`);
    const button = document.querySelector(`#${type}-input`).nextElementSibling;

    try {
        // 실행 전 UI 업데이트
        button.disabled = true;
        outputElement.innerHTML = `
            <div class="loading-container">
                <div class="loading"></div>
                <span>분석 중입니다...</span>
            </div>
        `;

        const response = await fetch(`/run/${type}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                parameters: inputElement.value.trim()
            })
        });

        const result = await response.json();

        // 결과 처리
        if (result.success) {
            displaySuccess(outputElement, result.output);
        } else {
            displayError(outputElement, result.error);
        }
    } catch (error) {
        displayError(outputElement, '서버 연결 오류가 발생했습니다.');
    } finally {
        button.disabled = false;
    }
}

function displaySuccess(element, output) {
    element.innerHTML = `
        <div class="success-message">
            <h3>분석 완료</h3>
            <div class="output-content">${formatOutput(output)}</div>
        </div>
    `;
}

function displayError(element, message) {
    element.innerHTML = `
        <div class="error-message">
            <h3>오류 발생</h3>
            <div class="error-content">${message}</div>
        </div>
    `;
}

function formatOutput(output) {
    // 결과값이 JSON인 경우 포맷팅
    try {
        const parsed = JSON.parse(output);
        return `<pre>${JSON.stringify(parsed, null, 2)}</pre>`;
    } catch {
        // JSON이 아닌 경우 그대로 출력
        return `<pre>${output}</pre>`;
    }
}

// 입력창 자동 크기 조절
document.querySelectorAll('.code-input').forEach(textarea => {
    textarea.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight + 2) + 'px';
    });
});

// 실행 단축키 설정 (Ctrl + Enter)
document.querySelectorAll('.code-input').forEach(textarea => {
    textarea.addEventListener('keydown', function(e) {
        if (e.ctrlKey && e.key === 'Enter') {
            e.preventDefault();
            const type = this.id.replace('-input', '');
            runAnalysis(type);
        }
    });
});

// 초기 플레이스홀더 설정
document.addEventListener('DOMContentLoaded', () => {
    Object.entries(analysisTools).forEach(([type, config]) => {
        const input = document.getElementById(`${type}-input`);
        if (input) {
            input.placeholder = config.placeholder;
        }
    });
});
