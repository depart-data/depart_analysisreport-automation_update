# scripts/db_connector.py 예시
from sqlalchemy import create_engine

def get_engine():
    # 실제 접속 정보로 수정하세요
    db_url = "postgresql://violet:violetarasterized@deplan-analysis.cpmiuu620ld9.ap-northeast-2.rds.amazonaws.com:5432/depart_data"
    return create_engine(db_url)
