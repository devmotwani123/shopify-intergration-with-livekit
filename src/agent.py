import os
import logging

import httpx
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    inference,
)
from livekit.plugins import silero

# Load environment variables
load_dotenv()

logger = logging.getLogger("shopify-agent")


# --- SHOPIFY API HELPER ---
async def get_shopify_order_id(order_number: str):
    clean_number = order_number.replace("#", "")
    shop = os.getenv("SHOPIFY_SHOP")
    token = os.getenv("SHOPIFY_ACCESS_TOKEN")

    url = f"https://{shop}/admin/api/2024-01/orders.json"
    headers = {"X-Shopify-Access-Token": token}
    params = {"name": f"#{clean_number}", "status": "any"}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, params=params)
        orders = response.json().get("orders", [])
        return orders[0] if orders else None


# --- AGENT CLASS ---
class ShopifyAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful Shopify assistant. "
                "ONLY ask for Order Number. "
                "Speak in Hinglish. "
                "Use tools to check order status, items, or cancel orders."
            )
        )

    # --- TOOL: GET ORDER INFO ---
    @function_tool(description="Get order status and items for a specific order number.")
    async def get_order_info(self, context: RunContext, order_number: str) -> str:
        del context

        order = await get_shopify_order_id(order_number)

        if not order:
            return f"Maaf, order {order_number} nahi mila."

        status = order.get("fulfillment_status") or "Processing"

        items = [
            f"{i['quantity']}x {i['title']}"
            for i in order.get("line_items", [])
        ]

        return f"Order {order_number} ka status '{status}' hai. Items: {', '.join(items)}."

    # --- TOOL: CANCEL ORDER ---
    @function_tool(description="Cancel a specific Shopify order.")
    async def cancel_order(self, context: RunContext, order_number: str) -> str:
        del context

        order = await get_shopify_order_id(order_number)

        if not order:
            return "Order nahi mila."

        shop = os.getenv("SHOPIFY_SHOP")
        token = os.getenv("SHOPIFY_ACCESS_TOKEN")

        url = f"https://{shop}/admin/api/2024-01/orders/{order['id']}/cancel.json"

        async with httpx.AsyncClient() as client:
            res = await client.post(
                url,
                headers={"X-Shopify-Access-Token": token}
            )

            if res.status_code == 200:
                return "Order successfully cancel ho gaya."
            else:
                return "Cancel nahi ho paaya."


# --- ENTRYPOINT ---
async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        vad=silero.VAD.load(),

        stt=inference.STT(
            model=os.getenv("STT_MODEL", "deepgram/nova-3"),
            language="multi",
        ),

        llm=inference.LLM(
            model=os.getenv("LLM_MODEL", "openai/gpt-4.1-mini"),
        ),

        tts=inference.TTS(
            model=os.getenv("TTS_MODEL", "cartesia/sonic-3"),
            voice=os.getenv(
                "TTS_VOICE",
                "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
            ),
        ),
    )

    await session.start(agent=ShopifyAssistant(), room=ctx.room)

    await session.say(
        "Namaste! Main aapka assistant hoon. Kripya apna order number batayein.",
        allow_interruptions=True,
    )


# --- MAIN ---
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))