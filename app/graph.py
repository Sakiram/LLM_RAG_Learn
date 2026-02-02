from typing import TypedDict, Sequence, Annotated
from operator import add as add_messages
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def build_graph(llm, tools):

    tools_dict = {t.name: t for t in tools}

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        return hasattr(last, "tool_calls") and len(last.tool_calls) > 0

    def call_llm(state: AgentState):
        system = SystemMessage(
            content="Answer questions using the PDF. Cite content."
        )
        response = llm.invoke([system] + list(state["messages"]))
        return {"messages": [response]}

    def take_action(state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        results = []

        for call in tool_calls:
            tool = tools_dict[call["name"]]
            result = tool.invoke(call["args"]["query"])
            results.append(
                ToolMessage(
                    tool_call_id=call["id"],
                    name=call["name"],
                    content=result
                )
            )

        return {"messages": results}

    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("tools", take_action)

    graph.add_conditional_edges(
        "llm",
        should_continue,
        {True: "tools", False: END}
    )

    graph.add_edge("tools", "llm")
    graph.set_entry_point("llm")

    return graph.compile()