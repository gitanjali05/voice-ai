import enum
import logging
from typing import Any

from livekit.agents import Agent, RunContext, function_tool

logger = logging.getLogger("temperature-control")
logger.setLevel(logging.INFO)


class Zone(str, enum.Enum):
    LIVING_ROOM = "living_room"
    BEDROOM = "bedroom"
    KITCHEN = "kitchen"
    BATHROOM = "bathroom"
    OFFICE = "office"


class HomeAgent(Agent):
    def __init__(self, instructions: str):
        super().__init__(instructions=instructions)

        self.temperature_settings = {
            Zone.LIVING_ROOM: 22,
            Zone.BEDROOM: 20,
            Zone.KITCHEN: 24,
            Zone.BATHROOM: 23,
            Zone.OFFICE: 21,
        }

    @function_tool(description="Get the temperature in a specific room in the house.")
    async def get_temperature(self, context: RunContext, zone: Zone) -> dict[str, Any]:
        logger.info("get_temperature - zone=%s", zone)
        temp = self.temperature_settings[zone]
        return {"zone": zone.value, "temperature_c": temp, "message": f"The current temperature in the {zone.value} is {temp}°C."}
