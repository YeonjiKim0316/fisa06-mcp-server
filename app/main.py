import os
import json
from contextlib import asynccontextmanager
from typing import Dict, List
from uuid import uuid4

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent.supervisor_graph import create_supervisor_app as create_agent_app

load_dotenv()

# LangSmith 설정 (선택)
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
if langsmith_api_key and langsmith_api_key.strip():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = langsmith_api_key  # docker compose down / docker compose up -d
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "LLM Agent with LangGraph")
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

app = FastAPI(title="LLM Agent API")
templates = Jinja2Templates(directory="templates")
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET_KEY", "dev-only-change-me"),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs("data", exist_ok=True)
    sqlite_checkpointer_cm = AsyncSqliteSaver.from_conn_string("data/checkpoints.sqlite")
    sqlite_checkpointer = await sqlite_checkpointer_cm.__aenter__()
    await sqlite_checkpointer.setup()
    app.state.sqlite_checkpointer_cm = sqlite_checkpointer_cm
    app.state.sqlite_checkpointer = sqlite_checkpointer
    app.state.agent_app = await create_agent_app(checkpointer=sqlite_checkpointer)
    try:
        yield
    finally:
        await sqlite_checkpointer_cm.__aexit__(None, None, None)


app.router.lifespan_context = lifespan

def get_thread_id(request: Request) -> str | None:
    username = request.session.get("username")
    if not username:
        return None

    user_threads: Dict[str, str] = request.session.get("user_threads", {})
    thread_id = user_threads.get(username)
    if not thread_id:
        thread_id = f"{username}:{uuid4().hex}"
        user_threads[username] = thread_id
        request.session["user_threads"] = user_threads

    request.session["thread_id"] = thread_id
    return thread_id


async def ensure_agent_app(request: Request):
    agent_app = getattr(request.app.state, "agent_app", None)
    if agent_app is not None:
        return agent_app

    sqlite_checkpointer = getattr(request.app.state, "sqlite_checkpointer", None)
    if sqlite_checkpointer is None:
        raise HTTPException(status_code=503, detail="Agent is not initialized")

    agent_app = await create_agent_app(checkpointer=sqlite_checkpointer)
    request.app.state.agent_app = agent_app
    return agent_app

async def get_history(request: Request) -> List[BaseMessage]:
    agent_app = getattr(request.app.state, "agent_app", None)
    if agent_app is None:
        return []

    thread_id = get_thread_id(request)
    if not thread_id:
        return []
    
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = await agent_app.aget_state(config)
        messages = state.values.get("messages", [])
        # 사용자, AI가 보낸 메시지만 필터링하여 프론트로 전달
        history = [m for m in messages if m.type in ["human", "ai"]]
        return history
    except Exception:
        return []

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if request.session.get("username"):
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse(
        request,
        "login.html",
        {"request": request, "error": ""},
    )

@app.post("/login", response_class=HTMLResponse)
async def login(request: Request, username: str = Form(...)):
    username = username.strip()
    if not username:
        return templates.TemplateResponse(
            request,
            "login.html",
            {"request": request, "error": "사용자 이름을 입력하세요."},
        )

    user_threads: Dict[str, str] = request.session.get("user_threads", {})
    thread_id = user_threads.get(username)
    if not thread_id:
        thread_id = f"{username}:{uuid4().hex}"
        user_threads[username] = thread_id

    request.session["user_threads"] = user_threads
    request.session["username"] = username
    request.session["thread_id"] = thread_id
    return RedirectResponse(url="/", status_code=303)

@app.post("/logout")
async def logout(request: Request):
    request.session.pop("username", None)
    request.session.pop("thread_id", None)
    return RedirectResponse(url="/login", status_code=303)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    if not request.session.get("username"):
        return RedirectResponse(url="/login", status_code=303)

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "chat_history": await get_history(request),
            "username": request.session.get("username", ""),
            "success": True,
            "error": "",
        },
    )

@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, query: str = Form(...)): # Legacy fallback
    if not request.session.get("username"):
        return RedirectResponse(url="/login", status_code=303)

    agent_app = await ensure_agent_app(request)

    thread_id = get_thread_id(request)
    if not thread_id:
        return RedirectResponse(url="/login", status_code=303)

    # messages: 현재 사용자 질문만 보냄. 기존 문맥은 checkpointer가 관리.
    messages_to_send = [HumanMessage(content=query)]
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = await agent_app.ainvoke({"messages": messages_to_send}, config=config)
        success = True
        error = ""
    except Exception as e:
        success = False
        error = str(e)

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "chat_history": await get_history(request),
            "username": request.session.get("username", ""),
            "success": success,
            "error": error,
        },
    )

@app.get("/chat/stream")
async def chat_stream(request: Request, query: str):
    async def event_generator():
        if not request.session.get("username"):
            yield f"data: {json.dumps({'type': 'error', 'content': '로그인이 필요합니다.'})}\n\n"
            return

        agent_app = await ensure_agent_app(request)

        thread_id = get_thread_id(request)
        if not thread_id:
            yield f"data: {json.dumps({'type': 'error', 'content': '세션이 유효하지 않습니다.'})}\n\n"
            return

        config = {"configurable": {"thread_id": thread_id}}
        messages_to_send = [HumanMessage(content=query)]
        emitted_token = False
        
        try:
            async for event in agent_app.astream_events({"messages": messages_to_send}, version="v2", config=config):
                kind = event["event"]
                # 텍스트 스트리밍
                if kind == "on_chat_model_stream":
                    # supervisor의 구조화 라우팅(JSON) 스트림은 사용자에게 노출하지 않음
                    if event.get("metadata", {}).get("langgraph_node") == "supervisor":
                        continue
                    content = event["data"]["chunk"].content
                    if content:
                        emitted_token = True
                        yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
                # 도구 사용 표시
                elif kind == "on_tool_start":
                    tool_name = event["name"]
                    yield f"data: {json.dumps({'type': 'tool_start', 'content': f'도구 사용: {tool_name}'})}\n\n"
                elif kind == "on_tool_end":
                    tool_name = event["name"]
                    yield f"data: {json.dumps({'type': 'tool_end', 'content': f'도구 완료: {tool_name}'})}\n\n"

            # clarify_node처럼 토큰 스트림이 없는 경우 최종 AI 메시지를 보강 전송
            if not emitted_token:
                state = await agent_app.aget_state(config)
                messages = state.values.get("messages", [])
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) or getattr(msg, "type", "") == "ai":
                        content = getattr(msg, "content", "")
                        if content:
                            yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
                        break
            
            yield f"data: {json.dumps({'type': 'finish'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/clear-chat")
async def clear_chat(request: Request):
    username = request.session.get("username")
    if not username:
        return JSONResponse({"ok": False, "error": "로그인이 필요합니다."}, status_code=401)

    # 새 thread_id를 발급해 현재 세션 대화를 초기화합니다.
    new_thread_id = f"{username}:{uuid4().hex}"
    user_threads: Dict[str, str] = request.session.get("user_threads", {})
    user_threads[username] = new_thread_id
    request.session["user_threads"] = user_threads
    request.session["thread_id"] = new_thread_id
    return JSONResponse({"ok": True})

@app.get("/chat-history")
async def get_chat_history(request: Request):
    if not request.session.get("username"):
        return {"chat_history": []}

    history_messages = await get_history(request)
    chat_history = []
    for msg in history_messages:
        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
            chat_history.append({"type": "user", "content": getattr(msg, "content", "")})
        elif isinstance(msg, AIMessage) or getattr(msg, "type", "") == "ai":
            chat_history.append({"type": "ai", "content": getattr(msg, "content", "")})
    return {"chat_history": chat_history}

@app.get("/api")
async def root():
    return {"message": "LLM Agent API is running"}