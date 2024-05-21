import os
from dataclasses import dataclass, field, asdict
from typing import List

import yaml

from codebuddy.backend import Backend
from codebuddy.openai_backend import OpenaiBackend
from codebuddy.bedrock_backend import BedrockBackend
from codebuddy.script import Script
from codebuddy.utils import Dialog, Message

import logging

logger = logging.getLogger(__name__)


@dataclass
class ChatModule(Backend):
    """A basic chat module."""

    name: str = field(metadata={"help": "A name for the module"}, default="")
    config_path: str = field(
        metadata={
            "help": (
                "Path to a yaml file containing module fields. Will override fields provided "
                "in the constructor."
            )
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

    def __post_init__(self):
        self.config_path = os.path.expanduser(self.config_path)
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

    def __call__(self, msg: str, messages: List[Message] = None, clear: bool = False) -> str:
        if messages is not None:
            self.messages = messages
        if clear:
            self.clear()
        for chunk in self.forward(msg):
            yield chunk
        logger.info(f"Total tokens: {self.tokens}")

    def get_gradio_interface(self, **kwargs):
        """Returns a gradio chat interface."""
        import gradio as gr

        def predict(history):
            messages = []
            history[-1][1] = ""
            for human, assistant in history[:-1]:
                messages.append(Message("user", human))
                messages.append(Message("assistant", assistant))
            for chunk in self(history[-1][0], messages=messages):
                history[-1][1] = chunk
                yield history

        def user(user_message, history):
            return "", history + [[user_message, None]]

        with gr.Blocks(analytics_enabled=False, **kwargs) as gui:
            chat = gr.Chatbot(label=self.name, height=500)
            with gr.Row():
                clear = gr.Button("Clear", variant="secondary", size="sm", min_width=60)
            with gr.Row():
                msg = gr.Textbox(
                    container=False,
                    show_label=False,
                    label="Message",
                    placeholder="Type a message...",
                    scale=7,
                    autofocus=True,
                )
                submit = gr.Button("Submit", variant="primary", scale=1, min_width=150)

            msg.submit(user, [msg, chat], [msg, chat], queue=False).then(predict, chat, chat)
            submit.click(user, [msg, chat], [msg, chat], queue=False).then(predict, chat, chat)
            clear.click(lambda: None, None, chat, queue=False)

        return gui

    def forward(self, message: str = "", depth: int = 0) -> str:
        """Generate a response to a user message."""
        self.messages.append(Message("user", message))
        response_content = self.call_api([Message("system", self.instruction)] + self.messages)
        self.messages.append(Message("assistant", response_content))
        yield response_content


@dataclass
class OpenaiChatModule(ChatModule, OpenaiBackend):
    """A chat module using OpenaiBackend."""


@dataclass
class BedrockChatModule(ChatModule, BedrockBackend):
    """A chat module using BedrockBackend."""


CHAT_MODULES = {
    "openai": OpenaiChatModule,
    "bedrock": BedrockChatModule
}

@dataclass
class ChatbotLauncher(Script):
    """Launch a OpenAI chatbot gradio gui."""
    backend: str = field(default="openai", metadata={"help": "Backend name"})
    prompt_name: str = field(default="omni", metadata={"help": "The name of the prompt config yaml file."})

    def run(self):
        prompt_basepath = os.path.dirname(os.path.dirname(__file__))
        prompt_path = os.path.join(prompt_basepath, "prompts", self.prompt_name + ".yaml")
        module = CHAT_MODULES[self.backend](config_path=prompt_path)
        gui = module.get_gradio_interface()
        gui.launch(share=False)


if __name__ == "__main__":
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO)
    load_dotenv()
    ChatbotLauncher.parse_args().run()
