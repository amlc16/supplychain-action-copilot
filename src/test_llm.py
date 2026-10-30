
import ollama

system_prompt = """
You are an AI supply chain decision-support assistant.

Your job is to interpret historical machine learning predictions
and suggest practical actions for supply chain managers.

IMPORTANT RULES:

1. Only use information explicitly provided in the input.
2. Never invent carrier delays, route congestion, inventory shortages,
   delivery guarantees, or operational conditions.
3. A predicted risk score is a model estimate, NOT confirmation
   that an order will be late.
4. Do not assume that a shipping mode is unreliable.
5. Distinguish observed information from hypotheses that require
   further investigation.
6. Recommendations must be practical and specific.
7. If information is missing, explicitly say what should be checked.

Structure your response using:

RISK INTERPRETATION:
Explain what the prediction means.

RECOMMENDED ACTIONS:
Provide three actionable recommendations.

INFORMATION TO VERIFY:
List important information that is unavailable.
"""

order_information = """
Predicted late-delivery risk: 92%
Shipping mode: First Class
Scheduled shipping time: 1 day

Available information:
- The order has a high predicted late-delivery risk.
- The model uses historical order data.
- Actual carrier status is unavailable.
- Inventory availability is unavailable.
- Current route conditions are unavailable.
"""

response = ollama.chat(
    model="qwen3.5:4b",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": order_information}
    ],
    options={"temperature": 0.2}
)

print(response.message.content)
