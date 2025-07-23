import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import UserMessage
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    model = OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv("OPEN_AI_API_KEY"),
    )

    topic = "C# is better than Python for AI development."

    supporter = AssistantAgent(
        name="John",
        system_message=('You are a John, a supporter agent in a debate for'
         f' the topic: {topic}. You will be debatining againts Jack a critic agent.'
        ),
        model_client=model,
    )

    critic = AssistantAgent(
        name="Jack",
        system_message=('You are a Jack, a critic agent in a debate for'
         f' the topic: {topic}. You will be debating against John a supporter agent.'
        ),
        model_client=model,
    )

    team = RoundRobinGroupChat(
        participants=[supporter, critic],
        max_turns=4,
    )

    res = await team.run(task="Start the debate!")

    for message in res.messages:
        print('-'*80)
        print(f"{message.source}: {message.content}")

if __name__ == "__main__":
    asyncio.run(main())