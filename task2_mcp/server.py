

import json
import uuid
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
from mcp.server.fastmcp import FastMCP

DB_PATH        = Path(os.getenv("NOVA_DB_PATH",      "nova_mock_db.json"))
CATALOG_PATH   = Path(os.getenv("NOVA_CATALOG_PATH", "product_catalog.txt"))
AUDIT_LOG_PATH = Path(os.getenv("NOVA_AUDIT_LOG",    "audit_log.jsonl"))


def _load_db() -> dict:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"nova_mock_db.json not found at {DB_PATH}")
    return json.loads(DB_PATH.read_text())


def _load_catalog() -> str:
    if CATALOG_PATH.exists():
        return CATALOG_PATH.read_text()
    return ""


DB      = _load_db()
CATALOG = _load_catalog()


def _audit(tool: str, params: dict, result: dict, call_id: str = None) -> str:
    call_id = call_id or str(uuid.uuid4())
    entry = {
        "call_id":    call_id,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "tool":       tool,
        "params":     params,
        "result_summary": {
            "status":  result.get("status", "ok"),
            "keys":    list(result.keys()),
        },
    }
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return call_id



mcp = FastMCP("nova-backend")



@mcp.tool()
def lookup_order(order_id: str) -> dict:
    """
    Look up a NOVA order by order ID.

    Returns order status, items, tracking number, delivery date,
    shipping address, and customer details.

    Args:
        order_id: The NOVA order ID (e.g. 'ORD-10001')
    """
    params = {"order_id": order_id}

    # Find order
    order = next((o for o in DB["orders"] if o["order_id"] == order_id), None)
    if not order:
        result = {
            "status":  "not_found",
            "message": f"No order found with ID '{order_id}'. Please verify the order number.",
        }
        _audit("lookup_order", params, result)
        return result

    # Enrich with customer info
    customer = next((c for c in DB["customers"] if c["id"] == order["customer_id"]), {})

    result = {
        "status":           "found",
        "order_id":         order["order_id"],
        "order_status":     order["status"],
        "customer_name":    customer.get("name", "Unknown"),
        "customer_id":      order["customer_id"],
        "items":            order["items"],
        "total":            order["total"],
        "placed_at":        order["placed_at"],
        "shipping_address": order["shipping_address"],
        "tracking_number":  order.get("tracking_number"),
        "delivered_at":     order.get("delivered_at"),
        "shipped_at":       order.get("shipped_at"),
        "return_reason":    order.get("return_reason"),
    }
    _audit("lookup_order", params, result)
    return result



@mcp.tool()
def query_knowledge_base(query: str, skin_type: str = None) -> dict:
    """
    Query the NOVA product knowledge base for ingredient info,
    skin type suitability, sizing guides, FAQs, and policies.

    Args:
        query:     Natural language question about a product or policy
        skin_type: Optional skin type filter (oily/dry/combination/sensitive/normal)
    """
    params = {"query": query, "skin_type": skin_type}
    q = query.lower()

    matched_products = []
    for product in DB["products"]:
        name = product["name"].lower()
        desc = product.get("description", "").lower()
        ingr = " ".join(product.get("ingredients", [])).lower()
        skin = product.get("skin_types", [])

        # Match by query keywords
        score = 0
        for word in q.split():
            if word in name: score += 3
            if word in desc: score += 2
            if word in ingr: score += 1

        # Skin type filter boost
        if skin_type and skin_type.lower() in skin:
            score += 5

        if score > 0:
            matched_products.append({
                "score":       score,
                "product_id":  product["product_id"],
                "name":        product["name"],
                "category":    product["category"],
                "price":       product["price"],
                "description": product["description"],
                "ingredients": product.get("ingredients", []),
                "skin_types":  product.get("skin_types", []),
                "in_stock":    product["in_stock"],
                "rating":      product["rating"],
            })

    matched_products.sort(key=lambda x: x["score"], reverse=True)
    top_products = matched_products[:3]

    # Check FAQs
    matched_faqs = []
    for faq in DB.get("faqs", []):
        faq_text = (faq["question"] + " " + faq["answer"]).lower()
        if any(word in faq_text for word in q.split() if len(word) > 3):
            matched_faqs.append(faq)

    # Check catalog text for detailed ingredient/usage info
    catalog_snippet = ""
    if CATALOG:
        lines = CATALOG.split("\n")
        relevant_lines = []
        capture = False
        for line in lines:
            if any(word in line.lower() for word in q.split() if len(word) > 3):
                capture = True
            if capture:
                relevant_lines.append(line)
            if capture and len(relevant_lines) > 30:
                break
        catalog_snippet = "\n".join(relevant_lines[:30])

    result = {
        "status":           "ok",
        "query":            query,
        "products_found":   len(top_products),
        "products":         top_products,
        "faqs":             matched_faqs[:2],
        "catalog_context":  catalog_snippet[:800] if catalog_snippet else "",
        "return_policy":    DB.get("return_policy", {}),
    }
    _audit("query_knowledge_base", params, result)
    return result



@mcp.tool()
def process_return(order_id: str, reason: str, items: list[str]) -> dict:
    """
    Initiate and validate a return request for a NOVA order.

    Checks eligibility against return policy and generates a
    return authorisation number and prepaid label.

    Args:
        order_id: The NOVA order ID to return
        reason:   Reason for return (e.g. 'damaged', 'wrong_item', 'changed_mind')
        items:    List of product IDs or names to return
    """
    params = {"order_id": order_id, "reason": reason, "items": items}

    order = next((o for o in DB["orders"] if o["order_id"] == order_id), None)
    if not order:
        result = {
            "status":  "not_found",
            "message": f"Order '{order_id}' not found.",
        }
        _audit("process_return", params, result)
        return result

    # Eligibility checks
    policy   = DB.get("return_policy", {})
    ineligible_reasons = []
    is_defective = reason.lower() in ("damaged", "defective", "wrong_item", "broken")

    # Check final-sale items
    final_sale_keywords = ["earring", "intimate", "underwear", "bra"]
    for item_id in items:
        item_name = item_id.lower()
        if any(kw in item_name for kw in final_sale_keywords):
            ineligible_reasons.append(
                f"'{item_id}' is a final-sale item and cannot be returned (earrings/intimate apparel)."
            )

    # Check order status
    non_returnable_statuses = ["return_requested", "returned", "processing"]
    if order["status"] in non_returnable_statuses:
        ineligible_reasons.append(
            f"Order is currently '{order['status']}' — return already initiated or not yet shipped."
        )

    if ineligible_reasons:
        result = {
            "status":    "ineligible",
            "order_id":  order_id,
            "reasons":   ineligible_reasons,
            "policy":    policy.get("conditions", []),
        }
        _audit("process_return", params, result)
        return result

    # Generate return authorisation
    rma_number  = f"RMA-{uuid.uuid4().hex[:8].upper()}"
    label_url   = f"https://returns.nova.com/label/{rma_number}"
    refund_type = "full_refund_to_original_payment" if is_defective else "store_credit"
    refund_timeline = "5–7 business days after item received"

    result = {
        "status":          "approved",
        "order_id":        order_id,
        "rma_number":      rma_number,
        "return_label_url": label_url,
        "items_to_return": items,
        "return_reason":   reason,
        "refund_type":     refund_type,
        "refund_timeline": refund_timeline,
        "instructions":    [
            f"Your return authorisation number is {rma_number}.",
            f"A prepaid return label has been emailed to you.",
            "Please ship items back within 7 days.",
            f"Refund ({refund_type.replace('_', ' ')}) processed {refund_timeline}.",
        ],
    }
    _audit("process_return", params, result)
    return result



@mcp.tool()
def get_product_recommendations(
    customer_id: str,
    skin_type: str = None,
    preferences: list[str] = None,
    budget_max: float = None,
) -> dict:
    """
    Generate personalised product recommendations for a NOVA customer.

    Uses purchase history, skin type, category preferences, and budget
    to surface the most relevant products.

    Args:
        customer_id:  NOVA customer ID (e.g. 'CUST001') or 'guest'
        skin_type:    Skin type override (oily/dry/combination/sensitive/normal)
        preferences:  List of preferred categories (skincare/makeup/hair/apparel)
        budget_max:   Maximum price per item in USD
    """
    params = {
        "customer_id": customer_id,
        "skin_type":   skin_type,
        "preferences": preferences,
        "budget_max":  budget_max,
    }

    # Resolve customer profile
    customer = next((c for c in DB["customers"] if c["id"] == customer_id), None)
    resolved_skin = skin_type or (customer["skin_type"] if customer else "normal")
    resolved_prefs = preferences or (customer["preferences"] if customer else ["skincare"])

    # Get already-purchased product IDs for this customer
    purchased_ids = set()
    if customer:
        for order in DB["orders"]:
            if order["customer_id"] == customer_id:
                for item in order["items"]:
                    purchased_ids.add(item["product_id"])

    # Score products
    scored = []
    for product in DB["products"]:
        if not product["in_stock"]:
            continue
        if budget_max and product["price"] > budget_max:
            continue
        if product["product_id"] in purchased_ids:
            continue   # don't re-recommend purchased items

        score = product["rating"]   # base score from rating

        # Skin type match
        skin_types = product.get("skin_types", [])
        if resolved_skin in skin_types or "all" in skin_types:
            score += 2.0

        # Category preference match
        if product["category"] in resolved_prefs:
            score += 1.5

        scored.append({
            "product_id":  product["product_id"],
            "name":        product["name"],
            "category":    product["category"],
            "price":       product["price"],
            "rating":      product["rating"],
            "description": product["description"],
            "skin_types":  skin_types,
            "score":       round(score, 2),
            "reason":      _build_rec_reason(product, resolved_skin, resolved_prefs),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:4]

    result = {
        "status":          "ok",
        "customer_id":     customer_id,
        "customer_name":   customer["name"] if customer else "Guest",
        "resolved_skin":   resolved_skin,
        "resolved_prefs":  resolved_prefs,
        "recommendations": top,
        "total_found":     len(scored),
    }
    _audit("get_product_recommendations", params, result)
    return result


def _build_rec_reason(product: dict, skin_type: str, prefs: list) -> str:
    reasons = []
    if skin_type in product.get("skin_types", []):
        reasons.append(f"ideal for {skin_type} skin")
    if product["category"] in prefs:
        reasons.append(f"matches your {product['category']} preference")
    if product["rating"] >= 4.7:
        reasons.append(f"top-rated ({product['rating']}/5)")
    return "; ".join(reasons) if reasons else "highly recommended"



@mcp.tool()
def escalate_to_human(
    customer_id: str,
    reason: str,
    conversation_summary: str,
    priority: str = "normal",
) -> dict:
    """
    Escalate a customer issue to a human support agent.

    Generates a ticket, assigns an agent, and provides estimated wait time.
    Always call this when: customer requests human, reports allergic reaction,
    makes legal threats, uses abusive language, or issue is unresolvable.

    Args:
        customer_id:           NOVA customer ID or 'unknown'
        reason:                Short escalation reason (e.g. 'allergic_reaction')
        conversation_summary:  Full summary of the conversation so far
        priority:              Ticket priority — 'normal' | 'high' | 'critical'
    """
    params = {
        "customer_id":          customer_id,
        "reason":               reason,
        "priority":             priority,
        "conversation_summary": conversation_summary[:200] + "…",
    }

    # Assign agent and wait time by priority
    agent_pool = {
        "critical": {"agent": "Sarah Mitchell",  "wait_min": 2,  "team": "Senior Support"},
        "high":     {"agent": "James Okafor",    "wait_min": 5,  "team": "Priority Support"},
        "normal":   {"agent": "Leila Nouri",     "wait_min": 12, "team": "Standard Support"},
    }
    assignment  = agent_pool.get(priority, agent_pool["normal"])
    ticket_id   = f"ESC-{uuid.uuid4().hex[:6].upper()}"

    customer    = next((c for c in DB["customers"] if c["id"] == customer_id), None)
    customer_name = customer["name"] if customer else "Customer"

    result = {
        "status":             "escalated",
        "ticket_id":          ticket_id,
        "customer_id":        customer_id,
        "customer_name":      customer_name,
        "assigned_agent":     assignment["agent"],
        "agent_team":         assignment["team"],
        "estimated_wait_min": assignment["wait_min"],
        "priority":           priority,
        "reason":             reason,
        "summary_logged":     True,
        "message_to_customer": (
            f"You've been connected to {assignment['agent']} from our "
            f"{assignment['team']} team (ticket {ticket_id}). "
            f"Estimated wait: {assignment['wait_min']} minutes. "
            f"Your full conversation history has been shared with them."
        ),
    }
    _audit("escalate_to_human", params, result)
    return result


if __name__ == "__main__":
    print("🚀 NOVA MCP Server starting (stdio transport)…", file=sys.stderr)
    mcp.run(transport="stdio")