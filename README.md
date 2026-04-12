# 1 LLM Agent Project

LangGraph 기반 Supervisor 멀티 에이전트 + RAG + MCP + SSE 스트리밍 예제입니다.

## 현재 아키텍처
- `app/main.py`: FastAPI 웹앱, 로그인/세션, SSE 스트리밍
- `agent/supervisor_graph.py`: Supervisor 라우팅 그래프
- `rag/retriever_tool.py`: Elasticsearch 기반 문서 검색 도구
- `langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver`: 대화 체크포인트 영속 저장

## 핵심 동작
- 로그인 사용자별 `thread_id`를 유지하여 재로그인 후에도 대화 이력 복원
- `clear-chat` 시 사용자의 `thread_id`만 새로 발급해 해당 사용자 대화 초기화
- Supervisor가 질의를 `rag_worker` 또는 `mcp_worker`로 라우팅
- `/chat/stream`에서 토큰 단위 스트리밍 응답 제공

## 실행
1. 환경변수 설정

`.env` 예시:

```env
OPENAI_API_KEY=...
LANGSMITH_API_KEY=...
ELASTICSEARCH_URL=http://host.docker.internal:9200
SESSION_SECRET_KEY=long_random_secret
```

2. 의존성 설치

```bash
pip install -r requirements.txt
```

3. 문서 인덱싱

```bash
python rag/ingest.py
```

4. 서버 실행

```bash
uvicorn app.main:app --reload --port 8000
```

## 참고
- 체크포인트 DB 파일은 `data/checkpoints.sqlite`에 저장됩니다.
- MCP 서버 URL은 `agent/supervisor_graph.py`의 `MultiServerMCPClient` 설정을 사용합니다.
