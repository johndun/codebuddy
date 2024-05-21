import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Tuple

import boto3

from codebuddy.backend import Backend
from codebuddy.utils import Message


import logging
logger = logging.getLogger(__name__)


@dataclass
class BedrockBackend(Backend):
    """Backend for Claude models on the AWS Bedrock API."""
    model_id: str = field(
        metadata={"help": "The model ID"}, default="anthropic.claude-3-haiku-20240307-v1:0"
    )
    max_tokens: int = field(
        metadata={"help": "The maximum number of tokens to generate"}, default=32768
    )
    top_k: int = field(
        metadata={"help": "The number of highest probability tokens to keep for top-k sampling"},
        default=100
    )
    top_p: float = field(
        metadata={"help": "The cumulative probability for top-p sampling"},
        default=0.999
    )
    temperature: float = field(
        metadata={"help": "The sampling temperature to use for generation"}, default=0.1
    )
    stop_sequences: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "A list of sequences to stop generation when encountered"}
    )
    anthropic_version: str = field(
        metadata={"help": "The version of the Anthropic API to use"}, default="bedrock-2023-05-31"
    )

    def request_base(self):
        return {
            k: v
            for k, v in self.__dict__.items()
            if k in [
                "stop_sequences", "max_tokens", "top_k", "top_p", "temperature",
                "anthropic_version"
            ]
        }

    def call_api(self, messages: List[Message], retries: int = 0) -> str:
        bedrock = boto3.client(service_name="bedrock-runtime")
        body = self.request_base()
        if messages[0].role == "system":
            body["messages"] = [asdict(msg) for msg in messages[1:]]
            body["system"] = messages[0].content
        else:
            body["messages"] = [asdict(msg) for msg in messages]

        try:
            response = bedrock.invoke_model(body=json.dumps(body), modelId=self.model_id)
            response_body = json.loads(response.get("body").read())
            response_content = response_body.get("content")[0]["text"]

        except Exception as ex:
            if "ThrottlingException" in str(ex) and retries < 3:
                logger.info("Throttled. Sleeping for 5s and trying again.")
                time.sleep(5)
                return self.call_api(messages, retries=retries + 1)
            elif "ExpiredTokenException" in str(ex) and retries < 3:
                logger.info("Token expired. Refreshing and trying again.")
                from importlib import reload
                reload(boto3)
                return self.call_api(messages, retries=retries + 1)
            else:
                raise ex

        self.input_tokens.append(response_body["usage"]["input_tokens"])
        self.output_tokens.append(response_body["usage"]["output_tokens"])
        return response_content


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    backend = BedrockBackend()
    response = backend.call_api([Message("user", "Hello")])
    logger.info(response)
    logger.info(backend.tokens)
