import os
import logging
import httpx
from datetime import datetime, timedelta
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

# ------------------ SETUP ------------------
load_dotenv()
logger = logging.getLogger("shopify-agent")


# ------------------ TTS ------------------
def build_tts():
    return inference.TTS(
        model="cartesia/sonic-3",
        voice=os.getenv("TTS_VOICE"),
        extra_kwargs={
            "speed": 0.9,
            "volume": 1.0,
        },
    )


# ------------------ SHOPIFY FETCH ------------------
async def get_shopify_order(order_number: str):
    clean_number = order_number.replace("#", "")
    shop = os.getenv("SHOPIFY_SHOP")
    token = os.getenv("SHOPIFY_ACCESS_TOKEN")

    url = f"https://{shop}/admin/api/2024-01/orders.json"

    params = {
        "name": f"#{clean_number}",
        "status": "any"
    }

    async with httpx.AsyncClient() as client:
        res = await client.get(
            url,
            headers={"X-Shopify-Access-Token": token},
            params=params
        )

        data = res.json()
        logger.info(f"Order Fetch: {data}")

        orders = data.get("orders", [])
        return orders[0] if orders else None


# ------------------ ACTIVE ORDERS ------------------
async def get_all_active_orders():
    shop = os.getenv("SHOPIFY_SHOP")
    token = os.getenv("SHOPIFY_ACCESS_TOKEN")

    url = f"https://{shop}/admin/api/2024-01/orders.json"

    params = {
        "status": "open",
        "limit": 10
    }

    async with httpx.AsyncClient() as client:
        res = await client.get(
            url,
            headers={"X-Shopify-Access-Token": token},
            params=params
        )

        data = res.json()
        logger.info(f"Active Orders: {data}")

        return data.get("orders", [])


# ------------------ STATUS ------------------
def get_order_status_text(order):
    if order.get("cancelled_at"):
        return "cancelled"

    fulfillment = order.get("fulfillment_status")

    if fulfillment == "fulfilled":
        return "delivered"

    if fulfillment is None:
        return "processing"

    return fulfillment or "processing"


# ------------------ DATE FORMAT ------------------
def format_order_date(created_at):
    if not created_at:
        return "unknown"

    try:
        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        return dt.strftime("%d %b %Y, %I:%M %p")
    except:
        return "unknown"


# ------------------ DELIVERY ESTIMATION 🔥 ------------------
def estimate_delivery_date(order):
    created_at = order.get("created_at")

    if not created_at:
        return "unknown"

    try:
        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

        if order.get("fulfillment_status") == "fulfilled":
            delivery = dt + timedelta(days=2)
        else:
            delivery = dt + timedelta(days=5)

        return delivery.strftime("%d %b %Y")
    except:
        return "unknown"


# ------------------ AGENT ------------------
class ShopifyAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly female Shopify assistant speaking Hinglish.\n\n"
                "RULES:\n"
                "- Always use tools\n"
                "- Never say 'data nahi hai'\n"
                "- If user asks order → get_order_info\n"
                "- If cancel → cancel_order\n"
                "- If active orders → get_active_orders\n"
                "- If delivery → estimate it\n"
                "- Speak like human (short + warm)\n"
            )
        )

    # -------- ORDER INFO --------
    @function_tool
    async def get_order_info(self, context: RunContext, order_number: str) -> str:
        del context

        order = await get_shopify_order(order_number)

        if not order:
            return f"Maaf kariye, order {order_number} nahi mila."

        status = get_order_status_text(order)
        formatted_date = format_order_date(order.get("created_at"))
        eta = estimate_delivery_date(order)

        items = [
            f"{i['quantity']}x {i['title']}"
            for i in order.get("line_items", [])
        ]

        if status == "cancelled":
            return (
                f"Aapka order {order_number} cancel ho chuka hai.\n"
                f"Ye order {formatted_date} ko place hua tha."
            )

        return (
            f"Haan ji 😊 ek second check karti hoon...\n\n"
            f"Order {order_number} {formatted_date} ko place hua tha.\n"
            f"Status: {status}\n"
            f"Expected delivery: {eta} 📦\n"
            f"Items: {', '.join(items)}."
        )

    # -------- CANCEL --------
    @function_tool
    async def cancel_order(self, context: RunContext, order_number: str) -> str:
        del context

        order = await get_shopify_order(order_number)

        if not order:
            return "Order nahi mila."

        if order.get("cancelled_at"):
            return "Ye order pehle se hi cancel hai."

        shop = os.getenv("SHOPIFY_SHOP")
        token = os.getenv("SHOPIFY_ACCESS_TOKEN")

        url = f"https://{shop}/admin/api/2024-01/orders/{order['id']}/cancel.json"

        async with httpx.AsyncClient() as client:
            res = await client.post(
                url,
                headers={"X-Shopify-Access-Token": token}
            )

            if res.status_code != 200:
                return "Cancel nahi ho paaya, please dobara try karein."

        updated = await get_shopify_order(order_number)

        if updated and updated.get("cancelled_at"):
            return "Aapka order successfully cancel ho gaya hai."
        else:
            return "Cancel request bhej di hai, thoda time lag sakta hai."

    # -------- ACTIVE ORDERS --------
    @function_tool
    async def get_active_orders(self, context: RunContext) -> str:
        del context

        orders = await get_all_active_orders()

        if not orders:
            return "Aapke paas koi active order nahi hai."

        result = []

        for o in orders:
            order_num = o.get("name")
            items = ", ".join([i["title"] for i in o.get("line_items", [])])
            result.append(f"{order_num} ({items})")

        return (
            "Haan ji 😊 aapke active orders ye hain:\n\n"
            + "\n".join(result)
        )


# ------------------ ENTRYPOINT ------------------
async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        vad=silero.VAD.load(),

        stt=inference.STT(
            model="deepgram/nova-3",
            language="multi",
        ),

        llm=inference.LLM(
            model="openai/gpt-4.1-mini",
        ),

        tts=build_tts(),
    )

    await session.start(agent=ShopifyAssistant(), room=ctx.room)

    await session.say(
        "Namaste! Main aapki assistant hoon 😊\n"
        "Order number batayein ya boliye 'mere active orders dikhao'.",
        allow_interruptions=True,
    )


# ------------------ MAIN ------------------
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
