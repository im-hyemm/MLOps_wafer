# WaferGuard Frontend

> 반도체 웨이퍼 품질 관리 AI 서비스의 Vue 대시보드

프런트엔드는 팀원이 담당한 영역이며, 통합 서비스의 사용자 흐름과 백엔드 연동을 보여주기 위해 저장소에 포함했습니다. 이 포트폴리오에서 중점적으로 다루는 데이터 분석·모델링과 백엔드는 아래 문서에서 확인할 수 있습니다.

- [모델링과 실험 결과](../modeling/README.md)
- [FastAPI와 모델 운영 파이프라인](../back/README.md)
- [프로젝트 전체 소개](../README.md)

## 제공 화면

- 단일 웨이퍼 결함 유형과 신뢰도 확인
- 단일 Lot의 다중 웨이퍼 분석
- 여러 Lot의 결함률 비교와 상세 결과 조회
- 라벨 데이터 업로드를 통한 현재 모델 평가와 재학습 요청
- Lot 공정 이력과 결함 유형별 통계 시각화
- 결함률이 높은 Lot의 LLM 기반 원인·점검 항목 확인

## 기술 스택

- Vue 3
- Vite
- Vue Router
- Axios
- Chart.js
- Bootstrap / BootstrapVueNext

## 로컬 실행

백엔드를 먼저 `http://localhost:8001`에서 실행한 후 다음 명령을 사용합니다.

```bash
cd front
npm install
npm run dev
```

기본 개발 서버는 `http://localhost:5173`에서 확인할 수 있습니다.
