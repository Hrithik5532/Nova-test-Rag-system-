import os
import json
import datetime
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from rag_module import query_knowledge_base as _rag_query

DB_PATH = os.path.join(os.path.dirname(__file__), "nova_mock_db.json")

with open(DB_PATH, "r") as f:
    MOCK_DB = json.load(f)

AUDIT_LOG_FILE = os.path.join(os.path.dirname(__file__), "audit_log.jsonl")


def log_audit(action: str, details: dict):
    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "action": action,
        "details": details,
    }
    with open(AUDIT_LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


@tool
def lookup_order(order_id: str) -> str:
    """
    Look up an order by its ID. Returns order status, items, tracking,
    and delivery information.
    Args:
        order_id: The order ID (e.g., 'ORD-10001')
    """
    log_audit("lookup_order", {"order_id": order_id})

    for order in MOCK_DB.get("orders", []):
        if order["order_id"] == order_id:
            customer_name = "Unknown"
            for c in MOCK_DB.get("customers", []):
                if c["id"] == order["customer_id"]:
                    customer_name = c["name"]
                    break

            result = {
                "order_id": order["order_id"],
                "customer": customer_name,
                "items": [f"{item['name']} (x{item['quantity']}, ${item['price']})" for item in order["items"]],
                "total": f"${order['total']:.2f}",
                "status": order["status"],
                "shipping_address": order.get("shipping_address", "N/A"),
                "tracking_number": order.get("tracking_number", "Not yet available"),
            }

            if order.get("delivered_at"):
                result["delivered_at"] = order["delivered_at"]
            if order.get("shipped_at"):
                result["shipped_at"] = order["shipped_at"]
            if order.get("return_reason"):
                result["return_reason"] = order["return_reason"]

            return json.dumps(result, indent=2)

    return f"Order '{order_id}' not found. Please verify the order ID and try again."


@tool
def process_return(order_id: str, reason: str) -> str:
    """
    Process a return request for an order. Checks eligibility based on
    NOVA's return policy and initiates the return process.
    Args:
        order_id: The order ID to return (e.g., 'ORD-10001')
        reason: The reason for the return
    """
    log_audit("process_return", {"order_id": order_id, "reason": reason})

    for order in MOCK_DB.get("orders", []):
        if order["order_id"] == order_id:
            if order["status"] not in ("delivered", "return_requested"):
                return f"Order {order_id} has status '{order['status']}'. Returns can only be processed for delivered orders."

            if order.get("delivered_at"):
                delivered = datetime.datetime.fromisoformat(order["delivered_at"].replace("Z", "+00:00"))
                now = datetime.datetime.now(datetime.timezone.utc)
                days_since = (now - delivered).days
                if days_since > 30:
                    return f"Return window has expired. Order {order_id} was delivered {days_since} days ago. NOVA's return policy is 30 days from delivery."

            non_returnable = []
            for item in order["items"]:
                name_lower = item["name"].lower()
                if "earring" in name_lower:
                    non_returnable.append(item["name"])

            if non_returnable:
                return f"The following items are final sale and cannot be returned: {', '.join(non_returnable)}. Earrings are non-returnable for hygiene reasons."

            is_defective = any(word in reason.lower() for word in ["damaged", "broken", "defective", "cracked", "leaked", "wrong"])
            refund_type = "full refund to original payment method" if is_defective else "store credit"

            result = {
                "status": "return_approved",
                "order_id": order_id,
                "refund_type": refund_type,
                "reason": reason,
                "next_steps": [
                    "A return shipping label has been emailed to the customer",
                    "Ship the item back within 7 days",
                    f"{'Refund' if is_defective else 'Store credit'} will be processed within 5-7 business days after we receive the item",
                ],
            }
            return json.dumps(result, indent=2)

    return f"Order '{order_id}' not found."


@tool
def get_product_recommendations(customer_id: str) -> str:
    """
    Get personalized product recommendations based on a customer's
    profile, skin type, preferences, and purchase history.
    Args:
        customer_id: The customer ID (e.g., 'CUST001')
    """
    log_audit("get_product_recommendations", {"customer_id": customer_id})

    customer = None
    for c in MOCK_DB.get("customers", []):
        if c["id"] == customer_id:
            customer = c
            break

    if not customer:
        return f"Customer '{customer_id}' not found."

    purchased_ids = set()
    for order in MOCK_DB.get("orders", []):
        if order["customer_id"] == customer_id:
            for item in order["items"]:
                purchased_ids.add(item["product_id"])

    recommendations = []
    for product in MOCK_DB.get("products", []):
        if product["product_id"] in purchased_ids:
            continue
        if not product.get("in_stock", False):
            continue

        score = 0
        skin_types = product.get("skin_types", [])
        if customer.get("skin_type") in skin_types or "all" in skin_types:
            score += 2
        if product.get("category") in customer.get("preferences", []):
            score += 3

        if score > 0:
            recommendations.append({
                "product_id": product["product_id"],
                "name": product["name"],
                "category": product["category"],
                "price": f"${product['price']:.2f}",
                "rating": product.get("rating", "N/A"),
                "match_score": score,
                "reason": f"Matches your {customer.get('skin_type', '')} skin type" if score >= 2 else f"In your preferred category: {product.get('category', '')}",
            })

    recommendations.sort(key=lambda x: x["match_score"], reverse=True)
    top_recs = recommendations[:5]

    if not top_recs:
        return "No personalized recommendations available at this time. Browse our full catalog at nova.com/shop!"

    result = {
        "customer": customer["name"],
        "skin_type": customer.get("skin_type", "unknown"),
        "recommendations": top_recs,
    }
    return json.dumps(result, indent=2)


@tool
def escalate_to_human(customer_id: str, reason: str, conversation_summary: str) -> str:
    """
    Escalate a support case to a human agent. Use this when the customer
    is frustrated, requests a manager, reports an adverse reaction, or
    when the issue is beyond AI capabilities.
    Args:
        customer_id: The customer ID
        reason: Why this is being escalated
        conversation_summary: Summary of the conversation so far
    """
    log_audit("escalate_to_human", {
        "customer_id": customer_id,
        "reason": reason,
        "summary": conversation_summary,
    })

    result = {
        "status": "escalated",
        "ticket_created": f"ESC-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
        "priority": "high" if any(word in reason.lower() for word in ["allergic", "reaction", "legal", "angry", "furious"]) else "medium",
        "message": "Your case has been escalated to a senior support specialist. A human agent will contact you within 15 minutes. We sincerely apologize for the inconvenience.",
        "context_transferred": True,
    }
    return json.dumps(result, indent=2)


@tool
def search_product_knowledge(question: str) -> str:
    """
    Search the NOVA product knowledge base using RAG. Use this for questions
    about product ingredients, sizing, compatibility, shipping, returns policy,
    or any product-related information.
    Args:
        question: The product-related question to search for
    """
    log_audit("search_product_knowledge", {"question": question})
    return _rag_query(question)


tools = [
    lookup_order,
    process_return,
    get_product_recommendations,
    escalate_to_human,
    search_product_knowledge,
]

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "support_system_prompt.txt")

with open(PROMPT_PATH, "r") as f:
    SYSTEM_PROMPT = f.read()

model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

agent = create_react_agent(
    model=model,
    tools=tools,
    prompt=SYSTEM_PROMPT,
)


def main():
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        try:
            response = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]}
            )
            ai_message = response["messages"][-1].content
            print(f"\nNOVA AI: {ai_message}\n")

            log_audit("agent_response", {
                "user_input": user_input,
                "response_length": len(ai_message),
            })

        except Exception as e:
            print(f"\n Error: {e}\n")
            log_audit("agent_error", {"user_input": user_input, "error": str(e)})


if __name__ == "__main__":
    main()