import asyncio
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, AutoSubscribe
from livekit.plugins import openai, silero

from api import HomeAgent

load_dotenv()

INSTRUCTIONS = (
    "You are a voice assistant created by LiveKit. Your interface with users will be voice. "
    "Use short, concise responses and avoid unpronounceable punctuation."
)

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=openai.STT(),
        llm=openai.LLM(),
        tts=openai.TTS(),
    )

    await session.start(
        room=ctx.room,
        agent=HomeAgent(instructions=INSTRUCTIONS),
    )

    await asyncio.sleep(1)
    session.say("Hey, how can I help you today!", allow_interruptions=True)

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
