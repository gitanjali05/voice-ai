import asyncio
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, AutoSubscribe
from livekit.plugins import openai, silero

from api_data import CsvAgent

load_dotenv()

INSTRUCTIONS = (
    "You are a voice data analyst. Ask which CSV filename to use if it isn't provided. "
    "Use short spoken answers. For large outputs, summarize instead of reading every row."
)

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
#manages conversation flow
    session = AgentSession(
        vad=silero.VAD.load(), #voice activity detection
        stt=openai.STT(),
        llm=openai.LLM(),
        tts=openai.TTS(),
    )

    await session.start(
        room=ctx.room,
        agent=CsvAgent(instructions=INSTRUCTIONS),
    )

    await asyncio.sleep(1)
    session.say(
        "Hey! Put a CSV file in the data folder and tell me the filename. You can ask for an overview or summaries.",
        allow_interruptions=True,
    )

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
