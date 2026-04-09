# docker compose up --build -d

# 1. mcp 서버에 tool 역할을 할 수 있는 함수를 하나 추가하고, 
# 2. agent 서버가 제대로 그 tool을 호출하는지 확인해보세요.
# @app.get("/add_int", operation_id="add_int")

# 2. agent 서버가 다른 워커를 만들거나, 
#  rag system(엘라스틱서치)에 다른 index로 데이터를 적재해서, 해당 인덱스의 정보를 검색하도록 코드를 수정해 보세요.
from typing import Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState

from langchain_mcp_adapters.client import MultiServerMCPClient
from rag.retriever_tool import retriever_tool

load_dotenv()

# System prompts - 
SUPERVISOR_PROMPT = """당신은 스마트 라우터(Supervisor)입니다. 
- "rag_worker": Langchain, Langgraph, RAG 등 문서나 패키지 사용법 등 개발/도큐먼트와 관련된 질문일 때
- "mcp_worker": 날씨, 주식 정보 등 실시간 외부 데이터를 요구하거나 정수끼리 더할 때
- "clarify_worker": 질문이 너무 짧거나 모호해서 어떤 워커로 보낼지 확신이 없을 때
- "dummy_worker": 사용자가 **짱구**라는 단어를 말했을 때
사용자의 질문을 분석하여 다음 네 워커 중 하나에게 질문을 전달하세요:
답변을 직접 생성하지 말고, 반드시 워커 역할 중 하나를 골라 next_worker 필드로 출력하세요.
"""

# 라우팅 응답 스키마  
class RouteResponse(TypedDict):
    next_worker: Literal["rag_worker", "mcp_worker", "clarify_worker", "dummy_worker"]

# MCP 클라이언트
client = None

try:
    client = MultiServerMCPClient({
        "fisa-mcp": {
            # "url": "http://localhost:7860/mcp",
            "url": "https://mcp-server-fisa06.onrender.com/mcp",
            "transport": "streamable_http"
        }
    })
    print("✅ MCP 클라이언트 초기화 완료 (Supervisor)")
except Exception as e:
    print(f"⚠️ MCP 클라이언트 초기화 실패 (Supervisor): {e}")

# 에이전트 생성
async def create_supervisor_app(checkpointer=None):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 1. RAG 워커 생성 (도구: retriever_tool)
    rag_agent = create_agent(
        llm, 
        tools=[retriever_tool], 
        system_prompt="당신은 RAG 시스템 질문에 답변하는 전문가 워커입니다. 주어진 retriever 도구를 최대한 활용해 답변하세요."
    )

    # Langgraph Node 함수에서는 ainvoke를 써야 비동기 스트리밍이 매끄럽습니다.
    async def rag_node(state):
        result = await rag_agent.ainvoke(state)
        return {"messages": result["messages"][-1:]}

    # 2. MCP 워커 생성 (도구: stock, weather API)
    mcp_tools = []
    if client:
        try:
            mcp_tools = await client.get_tools() or []
            print(mcp_tools)
        except Exception as e:
            print("MCP 툴 로드 실패:", e)

    mcp_agent = create_agent(
        llm, 
        tools=mcp_tools,
        system_prompt="당신은 외부 API(주식, 날씨)를 사용하여 실시간 답변을 제공하는 전문 워커입니다. 도구 사용 결과에만 의존하여 답변하세요."
    )

    async def mcp_node(state):
        result = await mcp_agent.ainvoke(state)
        return {"messages": result["messages"][-1:]}

    async def clarify_node(state):
        return {
            "messages": [
                AIMessage(
                    content=(
                        "질문이 조금 모호합니다. 아래 중 하나로 구체화해 주세요:\n"
                        "1) 문서/개발 개념 질문(RAG)\n"
                        "2) 실시간 데이터 질문(날씨/주식)\n"
                        "예: '서울 오늘 날씨 알려줘' 또는 'LangGraph의 StateGraph 개념 설명해줘'"
                    )
                )
            ]
        }


    async def dummy_node(state):
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "더미 노드입니다!!"
                        )
                    )
                ]
            }

    # 3. Supervisor 노드
    supervisor_chain = llm.with_structured_output(RouteResponse)

    async def supervisor_node(state):
        messages = state["messages"]
        last_human_msg = next((m.content for m in reversed(messages) if m.type == "human"), "")
        
        decision = await supervisor_chain.ainvoke([
            {"role": "system", "content": SUPERVISOR_PROMPT},
            {"role": "user", "content": last_human_msg}
        ])
        
        next_worker = decision["next_worker"]
        return {"next": next_worker}

    # 강제 분기를 위해 State 확장
    class MultiAgentState(MessagesState):
        next: str

    # 4. StateGraph 연결
    builder = StateGraph(MultiAgentState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("rag_worker", rag_node)
    builder.add_node("mcp_worker", mcp_node)
    builder.add_node("clarify_worker", clarify_node)
    builder.add_node("dummy_worker", dummy_node)

    builder.add_edge(START, "supervisor")

    def route_to_worker(state: MultiAgentState):
        if state.get("next") == "mcp_worker":
            return "mcp_worker"
        if state.get("next") == "clarify_worker":
            return "clarify_worker"
        if state.get("next") == "dummy_worker":
            return "dummy_worker"
        return "rag_worker"

    builder.add_conditional_edges("supervisor", route_to_worker)
    
    # 워커 완료 후 최종 종료
    builder.add_edge("rag_worker", END)
    builder.add_edge("mcp_worker", END)
    builder.add_edge("clarify_worker", END)
    builder.add_edge("dummy_worker", END)

    # 그래프 컴파일 (외부에서 주입된 checkpointer 사용)
    graph = builder.compile(checkpointer=checkpointer)
    return graph
