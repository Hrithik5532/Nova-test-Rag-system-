import json
from task5_nova_platform import run_query, save_traces


def main():
    scenarios = [
        {
            "name": "Order Status Inquiry",
            "message": "Hi, can you tell me the status of my order ORD-10002? When will it arrive?",
            "customer_id": "CUST002",
        },
        {
            "name": "Return Request",
            "message": "I want to return order ORD-10005. The product was damaged during shipping.",
            "customer_id": "CUST001",
        },
        {
            "name": "Product Knowledge Query",
            "message": "What ingredients are in the HydraGlow Vitamin C Serum? Is it safe to use with retinol?",
            "customer_id": "CUST003",
        },
        {
            "name": "Product Recommendation",
            "message": "I have dry skin and I'm looking for a good moisturizer. Can you recommend something?",
            "customer_id": "CUST002",
        },
        {
            "name": "Escalation — Frustrated Customer",
            "message": "I used your CalmSkin Sensitive Cleanser and got a rash! This is UNACCEPTABLE! I want to speak to a manager RIGHT NOW!",
            "customer_id": "CUST004",
        },
        {
            "name": "Sizing Question",
            "message": "What size UrbanFlex Joggers should I get? My waist is 33 inches.",
            "customer_id": "CUST005",
        },
        {
            "name": "Shipping Query",
            "message": "Do you ship to India? How long does international shipping take?",
            "customer_id": "CUST001",
        },
    ]

    all_traces = []
    results = []

    for i, scenario in enumerate(scenarios, 1):
        try:
            result = run_query(
                scenario["message"],
                customer_id=scenario["customer_id"],
            )
            print(f"\nIntent: {result['intent']}")
            print(f"\nResponse:\n")
            for line in result["response"].split("\n"):
                print(f"     {line}")

            all_traces.extend(result["audit_trail"])
            results.append({
                "scenario": scenario["name"],
                "intent": result["intent"],
                "response_length": len(result["response"]),
                "audit_entries": len(result["audit_trail"]),
                "status": "Success",
            })
        except Exception as e:
            print(f"\nError: {e}")
            results.append({
                "scenario": scenario["name"],
                "status": f"Error: {str(e)[:100]}",
            })

    for r in results:
        status = r.get("status", "unknown")
        intent = r.get("intent", "N/A")
        print(f"  {status} {r['scenario']} -> Intent: {intent}")

    save_traces(all_traces)

    demo_report = {
        "demo_run_at": "2026-03-18",
        "total_scenarios": len(scenarios),
        "successful": sum(1 for r in results if "Success" in r.get("status", "")),
        "results": results,
        "total_audit_entries": len(all_traces),
    }
    with open("demo_report.json", "w") as f:
        json.dump(demo_report, f, indent=2)


if __name__ == "__main__":
    main()
