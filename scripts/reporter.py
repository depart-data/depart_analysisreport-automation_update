from jinja2 import Environment, FileSystemLoader
import os

def generate_html(context):

    def translate_gender(data):
        if isinstance(data, dict):
            return {k: translate_gender(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [translate_gender(i) for i in data]
        elif isinstance(data, str):
            res = data.replace("female", "여성").replace("Female", "여성")
            res = res.replace("male", "남성").replace("Male", "남성")
            return res
        return data

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