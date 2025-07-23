import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import UserMessage
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    model = OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv("OPEN_AI_API_KEY"),
    )

    res = await model.create(
        messages=[
            UserMessage(content="Hello, how can I help you today?", source="user"),
        ]
    )
    print(res.content)

if __name__ == "__main__":
    asyncio.run(main())