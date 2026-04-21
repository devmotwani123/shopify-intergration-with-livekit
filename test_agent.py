import asyncio
import logging
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli, Agent
from livekit.plugins import openai, silero

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test-agent")

class TestAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a test assistant. Respond to any message with 'Hello! I received your message.'"
        )

async def test_entrypoint(ctx):
    logger.info("🎯 TEST AGENT STARTED!")
    try:
        await ctx.connect()
        logger.info("✅ Connected to room")

        # Create session for voice interaction
        session = AgentSession(ctx.room)

        # Create the assistant
        assistant = TestAssistant()

        # Start the session
        await session.start(
            agent=assistant,
            room_input_options={
                "echoed_audio": False,
            },
        )

        logger.info("🎉 Agent ready! Try speaking now...")

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    options = WorkerOptions(entrypoint_fnc=test_entrypoint)
    cli.run_app(options)