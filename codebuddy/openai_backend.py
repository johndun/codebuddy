import os
from dataclasses import dataclass, field, asdict
from typing import List, Tuple

from openai import OpenAI

from codebuddy.backend import Backend
from codebuddy.utils import Message


import logging
logger = logging.getLogger(__name__)


@dataclass
class OpenaiBackend(Backend):
    """Backend for OpenAI chat completions API."""
    model: str = field(
        metadata={"help": "ID of the model to use."}, default="gpt-3.5-turbo-0125"
    )
    max_tokens: int = field(
        metadata={
            "help": "The maximum number of tokens that can be generated in the chat completion."
        },
        default=4096,
    )
    top_p: float = field(
        metadata={
            "help": (
                "An alternative to sampling with temperature, called nucleus sampling, where the "
                "model considers the results of the tokens with top_p probability mass. So 0.1 "
                "means only the tokens comprising the top 10% probability mass are considered."
            )
        },
        default=1,
    )
    temperature: float = field(
        metadata={
            "help": (
                "What sampling temperature to use, between 0 and 2. Higher values like 0.8 will "
                "make the output more random, while lower values like 0.2 will make it more "
                "focused and deterministic."
            )
        },
        default=0.1,
    )
    stop: List[str] = field(
        default_factory=lambda: [],
        metadata={
            "help": "Up to 4 sequences where the API will stop generating further tokens."
        },
    )

    def request_base(self):
        return {
            k: v for k, v in self.__dict__.items()
            if k in ["model", "max_tokens", "top_p", "temperature", "stop"]
        }

    def call_api(self, messages: List[Message], retries: int = 0) -> str:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        request = self.request_base()
        request["messages"] = [asdict(x) for x in messages]

        logger.debug(request)
        response = client.chat.completions.create(**request)
        logger.debug(response)

        self.input_tokens.append(response.usage.prompt_tokens)
        self.output_tokens.append(response.usage.completion_tokens)

        return response.choices[0].message.content


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    backend = OpenaiBackend()
    response = backend.call_api([Message("user", "Hello")])
    logger.info(response)
    logger.info(backend.tokens)
