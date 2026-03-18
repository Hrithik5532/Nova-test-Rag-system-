import os
import json
import datetime
from typing import TypedDict, Annotated, Literal

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent

from rag_module import query_knowledge_base
from main import (
    lookup_order,
    process_return,
    get_product_recommendations,
    escalate_to_human,
    search_product_knowledge,
    MOCK_DB,
    SYSTEM_PROMPT,
)


class NOVAState(TypedDict):
    messages: list
    intent: str
    customer_id: str
    audit_trail: list
    final_response: str


TRACES_FILE = os.path.join(os.path.dirname(__file__), "nova_traces.json")


def log_trace(state: NOVAState, node: str, action: str, details: dict) -> NOVAState:
    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "node": node,
        "action": action,
        "details": details,
    }
    state["audit_trail"].append(entry)
    return state


model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)


def router_node(state: NOVAState) -> NOVAState:
    user_message = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage) or (isinstance(msg, dict) and msg.get("role") == "user"):
            user_message = msg.content if isinstance(msg, HumanMessage) else msg["content"]

    classification_prompt = f"""Classify the following customer message into exactly ONE of these intents:
- order_status: asking about order tracking, delivery, shipping status
- return: requesting a return, refund, or exchange
- product_query: asking about product ingredients, sizing, compatibility, details
- recommendation: looking for product suggestions or recommendations
- escalation: frustrated, angry, demanding a manager, reporting adverse reaction

Think step by step:
1. What is the customer's primary concern?
2. Are there any escalation signals (anger, health issues, legal threats)?
3. What category best fits?

Customer message: "{user_message}"

Respond with ONLY the intent label (one of: order_status, return, product_query, recommendation, escalation). Nothing else."""

    response = model.invoke([SystemMessage(content="You are an intent classifier."), HumanMessage(content=classification_prompt)])
    intent = response.content.strip().lower()

    valid_intents = {"order_status", "return", "product_query", "recommendation", "escalation"}
    if intent not in valid_intents:
        intent = "product_query"

    state["intent"] = intent
    state = log_trace(state, "router", "intent_classified", {
        "user_message": user_message,
        "classified_intent": intent,
    })

    return state


support_agent = create_react_agent(
    model=model,
    tools=[lookup_order, process_return],
    prompt=SYSTEM_PROMPT + "\n\nYou are handling an ORDER or RETURN query. Use the lookup_order tool to find order details and process_return to handle returns. Be helpful and empathetic.",
)


def support_node(state: NOVAState) -> NOVAState:
    result = support_agent.invoke({"messages": state["messages"]})
    ai_msg = result["messages"][-1].content

    state["final_response"] = ai_msg
    state = log_trace(state, "support_agent", "query_handled", {
        "intent": state["intent"],
        "response_length": len(ai_msg),
    })
    return state


rag_agent = create_react_agent(
    model=model,
    tools=[search_product_knowledge],
    prompt=SYSTEM_PROMPT + "\n\nYou are handling a PRODUCT KNOWLEDGE query. Use the search_product_knowledge tool to find relevant product information from our knowledge base. Always ground your answers in the retrieved information.",
)


def rag_node(state: NOVAState) -> NOVAState:
    result = rag_agent.invoke({"messages": state["messages"]})
    ai_msg = result["messages"][-1].content

    state["final_response"] = ai_msg
    state = log_trace(state, "rag_agent", "knowledge_retrieved", {
        "intent": state["intent"],
        "response_length": len(ai_msg),
    })
    return state


rec_agent = create_react_agent(
    model=model,
    tools=[get_product_recommendations, search_product_knowledge],
    prompt=SYSTEM_PROMPT + "\n\nYou are handling a PRODUCT RECOMMENDATION query. Use get_product_recommendations with the customer ID to find personalized suggestions. You can also use search_product_knowledge for additional product details.",
)


def recommendation_node(state: NOVAState) -> NOVAState:
    messages = list(state["messages"])

    if state.get("customer_id"):
        messages.append(HumanMessage(
            content=f"[System: Customer ID is {state['customer_id']}. Use this with the get_product_recommendations tool.]"
        ))

    result = rec_agent.invoke({"messages": messages})
    ai_msg = result["messages"][-1].content

    state["final_response"] = ai_msg
    state = log_trace(state, "recommendation_agent", "recommendations_generated", {
        "customer_id": state.get("customer_id", "unknown"),
        "response_length": len(ai_msg),
    })
    return state


esc_agent = create_react_agent(
    model=model,
    tools=[escalate_to_human],
    prompt=SYSTEM_PROMPT + "\n\nYou are handling an ESCALATION. The customer is frustrated or has a critical issue. Use the escalate_to_human tool to transfer them to a human agent. Provide a brief, empathetic response while creating the escalation.",
)


def escalation_node(state: NOVAState) -> NOVAState:
    messages = list(state["messages"])

    if state.get("customer_id"):
        messages.append(HumanMessage(
            content=f"[System: Customer ID is {state['customer_id']}. Escalate this case — the customer needs human assistance.]"
        ))

    result = esc_agent.invoke({"messages": messages})
    ai_msg = result["messages"][-1].content

    state["final_response"] = ai_msg
    state = log_trace(state, "escalation_agent", "escalated_to_human", {
        "customer_id": state.get("customer_id", "unknown"),
        "reason": state["intent"],
    })
    return state


def route_by_intent(state: NOVAState) -> str:
    intent = state.get("intent", "product_query")
    routing = {
        "order_status": "support",
        "return": "support",
        "product_query": "rag",
        "recommendation": "recommendation",
        "escalation": "escalation",
    }
    return routing.get(intent, "rag")


def build_nova_graph():
    graph = StateGraph(NOVAState)

    graph.add_node("router", router_node)
    graph.add_node("support", support_node)
    graph.add_node("rag", rag_node)
    graph.add_node("recommendation", recommendation_node)
    graph.add_node("escalation", escalation_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_by_intent,
        {
            "support": "support",
            "rag": "rag",
            "recommendation": "recommendation",
            "escalation": "escalation",
        },
    )

    graph.add_edge("support", END)
    graph.add_edge("rag", END)
    graph.add_edge("recommendation", END)
    graph.add_edge("escalation", END)

    return graph.compile()


def save_traces(traces: list):
    with open(TRACES_FILE, "w") as f:
        json.dump(traces, f, indent=2)


nova_graph = build_nova_graph()


def run_query(user_message: str, customer_id: str = "") -> dict:
    initial_state: NOVAState = {
        "messages": [HumanMessage(content=user_message)],
        "intent": "",
        "customer_id": customer_id,
        "audit_trail": [],
        "final_response": "",
    }

    result = nova_graph.invoke(initial_state)

    return {
        "response": result.get("final_response", "I'm sorry, I couldn't process your request."),
        "intent": result.get("intent", "unknown"),
        "audit_trail": result.get("audit_trail", []),
    }


if __name__ == "__main__":
    all_traces = []

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            save_traces(all_traces)
            break

        result = run_query(user_input, customer_id="CUST001")
        print(f"\nNOVA AI: {result['response']}")
        print(f"  [Intent: {result['intent']}]\n")
        all_traces.extend(result["audit_trail"])
