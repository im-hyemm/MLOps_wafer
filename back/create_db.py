import psycopg2
import random
from config.db import DB_CONFIG

# 초기 접속용 (postgres 기본 DB)
INIT_DB_CONFIG = DB_CONFIG.copy()
INIT_DB_CONFIG["dbname"] = "postgres"

# 실제 사용할 DB 이름
DB_NAME = DB_CONFIG["dbname"]

# 공정 단계별 Tool / Recipe 후보
process_info = {
    "웨이퍼 제조": {
        "tools": ["CRYSTAL-01", "CRYSTAL-02"],
        "recipes": ["WAFER_STD_V1", "WAFER_FAST_V2"]
    },
    "산화": {
        "tools": ["OXI-01", "OXI-02", "OXI-03"],
        "recipes": ["OXI_DRY_V1", "OXI_WET_V2"]
    },
    "포토": {
        "tools": ["PHOTO-01", "PHOTO-02"],
        "recipes": ["PHOTO_STD_V1", "PHOTO_HP_V2"]
    },
    "식각": {
        "tools": ["ETCH-12A", "ETCH-08C"],
        "recipes": ["ETCH_STD_V1", "ETCH_DEEP_V2"]
    },
    "증착": {
        "tools": ["CVD-07B", "CVD-02A"],
        "recipes": ["CVD_LOWK_V1", "CVD_OXIDE_V2"]
    },
    "이온 주입": {
        "tools": ["IMPL-01", "IMPL-02"],
        "recipes": ["IMPL_BORON_V1", "IMPL_ARSENIC_V2"]
    },
    "금속 배선": {
        "tools": ["METAL-01", "METAL-02"],
        "recipes": ["METAL_AL_V1", "METAL_CU_V2"]
    }
}

# ------------------------------
# 1. DB 생성 (기존에 있으면 드롭 후 생성)
# ------------------------------
init_conn = psycopg2.connect(**INIT_DB_CONFIG)
init_conn.autocommit = True
init_cur = init_conn.cursor()

# 기존 연결을 강제로 끊고 DB 삭제
init_cur.execute(f"""
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = %s;
""", (DB_NAME,))
init_cur.execute(f"DROP DATABASE IF EXISTS {DB_NAME};")
init_cur.execute(f"CREATE DATABASE {DB_NAME};")
print(f"✅ Database '{DB_NAME}' 새로 생성 완료")

init_cur.close()
init_conn.close()

# ------------------------------
# 2. 테이블 생성 (기존 테이블 드롭 후 생성)
# ------------------------------
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

cur.execute("DROP TABLE IF EXISTS lot_process_run CASCADE;")
cur.execute("DROP TABLE IF EXISTS lot CASCADE;")

cur.execute("""
CREATE TABLE lot (
    lot_id   SERIAL PRIMARY KEY,
    lot_code VARCHAR(40) UNIQUE NOT NULL
);
""")

cur.execute("""
CREATE TABLE lot_process_run (
    run_id     SERIAL PRIMARY KEY,
    lot_id     INT NOT NULL REFERENCES lot(lot_id) ON DELETE CASCADE,
    step_name  VARCHAR(32) NOT NULL,
    tool       VARCHAR(64) NOT NULL,
    recipe     VARCHAR(64) NOT NULL
);
""")
conn.commit()
print("✅ 테이블 재생성 완료")

# ------------------------------
# 3. LOT + 공정 데이터 삽입
# ------------------------------
for i in range(1, 201):
    lot_code = f"lot{i}"
    cur.execute("INSERT INTO lot (lot_code) VALUES (%s) RETURNING lot_id;", (lot_code,))
    lot_id = cur.fetchone()[0]

    for step_name, info in process_info.items():
        tool = random.choice(info["tools"])
        recipe = random.choice(info["recipes"])
        cur.execute("""
            INSERT INTO lot_process_run (lot_id, step_name, tool, recipe)
            VALUES (%s, %s, %s, %s);
        """, (lot_id, step_name, tool, recipe))

conn.commit()
cur.close()
conn.close()

print("✅ LOT 200개와 공정별 이력 데이터 삽입 완료")
