import os
from dataclasses import dataclass, field, asdict
from typing import List

import yaml
from openai import OpenAI

from codebuddy.utils import Dialog
from codebuddy.utils import Message


import logging

logger = logging.getLogger(__name__)


@dataclass
class OpenaiCompletionArgs:
    """Inference arguments for OpenAI chat completions API."""

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
            "help": "An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered."
        },
        default=1,
    )
    temperature: float = field(
        metadata={
            "help": "What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic."
        },
        default=0.1,
    )
    stop: List[str] = field(
        default_factory=lambda: [],
        metadata={
            "help": "Up to 4 sequences where the API will stop generating further tokens."
        },
    )

    @property
    def completion_args(self):
        """Completions API arguments."""
        return {
            k: v
            for k, v in self.__dict__.items()
            if k in ["model", "max_tokens", "top_p", "temperature", "stop"]
        }


@dataclass
class OpenaiModule(OpenaiCompletionArgs):
    """A basic module for generating inferences using the OpenAI API."""

    name: str = field(metadata={"help": "A name for the module"}, default="")
    config_path: str = field(
        metadata={
            "help": "Path to a yaml file containing module fields. Will override fields provided in the constructor."
        },
        default="",
    )
    instruction: str = field(
        metadata={
            "help": "An optional instruction to include as the system message in a dialog"
        },
        default="",
    )
    messages: List[Message] = field(
        default_factory=lambda: [],
        metadata={"help": "A list of messages in the dialog"},
    )
    dialog_history: List[Dialog] = field(
        default_factory=lambda: [], metadata={"help": "A history of dialogs"}
    )
    input_tokens: List[int] = field(
        default_factory=lambda: [],
        metadata={"help": "A list of input token counts for each message"},
    )
    output_tokens: List[int] = field(
        default_factory=lambda: [],
        metadata={"help": "A list of output token counts for each message"},
    )

    @property
    def tokens(self):
        return {
            "llm_calls": len(self.input_tokens),
            "input_tokens": sum(self.input_tokens),
            "output_tokens": sum(self.output_tokens),
        }

    def __post_init__(self):
        if self.config_path:
            with open(self.config_path, "r") as file:
                module_data = yaml.safe_load(file)
            for field_name, value in module_data.items():
                setattr(self, field_name, value)

    def clear(self):
        """Clear the message history."""
        if self.messages:
            self.dialog_history.append(Dialog(self.messages))
            self.messages = []

    def __call__(
        self, msg: str, messages: List[Message] = None, clear: bool = False
    ) -> str:
        if messages is not None:
            self.messages = messages
        if clear:
            self.clear()
        return self.forward(msg)

    def get_gradio_interface(self, **kwargs):
        """Return a gradio chat interface."""
        import gradio as gr

        def predict(message, history):
            messages = []
            for human, assistant in history:
                messages.append(Message("user", human))
                messages.append(Message("assistant", assistant))
            yield self(message, messages=messages)

        return gr.ChatInterface(
            predict,
            analytics_enabled=False,
            chatbot=gr.Chatbot(label=self.name, height=600),
            **kwargs
        )

    def forward(self, message: str = "", depth: int = 0) -> str:
        """Generate a response to a user message."""
        logger.debug("Calling LLM")
        messages = self.messages
        messages.append(Message("user", message))
        if self.instruction:
            messages.insert(0, Message("system", self.instruction))

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        request = self.completion_args
        request["messages"] = [asdict(x) for x in messages]

        try:
            logger.debug(request)
            response = client.chat.completions.create(**request)
            logger.debug(response)
        except Exception as ex:
            raise ex

        self.input_tokens.append(response.usage.prompt_tokens)
        self.output_tokens.append(response.usage.completion_tokens)
        response_content = response.choices[0].message.content
        self.messages.append(Message("assistant", response_content))
        return response_content
