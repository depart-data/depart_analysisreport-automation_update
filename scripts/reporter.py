from jinja2 import Environment, FileSystemLoader
import os
import re

def translate_gender(data):
    if isinstance(data, dict):
        return {k: translate_gender(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [translate_gender(i) for i in data]
    elif isinstance(data, str):
        # 차트 이미지(base64)·SVG 문자열은 건드리지 않음 (부분 치환 시 이미지가 깨짐)
        stripped = data.lstrip()
        if stripped.startswith("data:") or stripped.startswith("<svg"):
            return data
        res = re.sub(r"\b[Ff]emale\b", "여성", data)
        res = re.sub(r"\b[Mm]ale\b", "남성", res)
        return res
    return data

def generate_html(context):

    # 렌더링 직전에 데이터 성별 번역 적용
    translated_context = translate_gender(context)

    # [절대 경로 설정] 현재 reporter.py 위치를 기준으로 상위 폴더의 templates를 찾습니다.
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    template_dir = os.path.join(base_dir, 'templates')

    # 확실하게 template_dir 변수를 사용하여 로더를 초기화합니다.
    file_loader = FileSystemLoader(template_dir)
    env = Environment(loader=file_loader)
    template = env.get_template('template.html')

    # 템플릿에 데이터 채우기
    output = template.render(translated_context)

    # 결과 html 파일 저장 경로 설정 (프로젝트 루트 폴더)
    output_path = os.path.join(base_dir, "report.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)

    return os.path.abspath(output_path)
