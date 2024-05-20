import os
import re
from dataclasses import dataclass, asdict, field
from typing import Union

import yaml
import logging
from openai import OpenAI

from codebuddy.script import Script
from codebuddy.openai_module import OpenaiModule
from codebuddy.tmux import TmuxSession
from codebuddy.utils import (
    PromptTemplate,
    run_bash,
    Message,
    split_markdown,
    process_chunks,
    TRIPLE_BACKTICKS,
)

logger = logging.getLogger(__name__)


@dataclass
class TmuxModule(OpenaiModule):
    prompt_template: Union[str, PromptTemplate] = ""  #: The prompt template
    max_calls: int = 5  #: Maximum number of LLM API calls
    python_env: str = os.path.dirname(os.path.dirname(__file__)) + "/codebuddy-venv"  #: Path to the Python environment
    project_path: str = "~/demo"  #: Path to the project directory
    sleep_duration: int = 1  #: Maximum number of LLM API calls.
    prefix_break_token: str = "TMUX_BREAK"  #: Maximum number of LLM API calls.
    prompt: str = "$"  #: The command prompt string.
    session_height: int = 8192  #: Number of lines for the tmux history
    session_width: int = 256  #: Width of the tmux history
    terminal_session_id: str = "terminal-session"  #: tmux terminal session name
    python_session_id: str = "python-session"  #: tmux python session name
    terminal_session: TmuxSession = None
    python_session: TmuxSession = None

    def __post_init__(self):
        assert self.project_path
        if self.config_path:
            with open(self.config_path, "r") as file:
                module_data = yaml.safe_load(file)
            for field_name, value in module_data.items():
                setattr(self, field_name, value)

        if isinstance(self.prompt_template, str):
            self.prompt_template = PromptTemplate(self.prompt_template)

        self.project_path = os.path.expanduser(self.project_path).rstrip("/")
        self.python_env = os.path.expanduser(self.python_env)

        self._initialize_tmux_sessions()

    def _initialize_tmux_sessions(self):
        """Initializes the tmux sessions."""
        session_args = {
            "session_width": self.session_width,
            "session_height": self.session_height,
            "sleep_duration": self.sleep_duration,
            "prefix_break_token": self.prefix_break_token,
            "prompt": self.prompt,
            "python_env": self.python_env,
            "project_path": self.project_path,
        }
        self.terminal_session = TmuxSession(
            session_id=self.terminal_session_id, **session_args
        )
        self.python_session = TmuxSession(
            session_id=self.python_session_id, **session_args
        )
        self.python_session.prompt = ">>>"
        for cmd in ("python", f"print('{self.prefix_break_token}')"):
            self.python_session(cmd)

    @property
    def project_tree(self):
        """Run tree on the project."""
        cmd_str = f'tree -I "*.pyc|__pycache__" {self.project_path}'
        project_tree = run_bash(cmd_str).strip()
        project_tree = project_tree.splitlines()
        if len(project_tree) > 1:
            project_tree = "$PROJECT_PATH\n" + "\n".join(project_tree[1:])
        return project_tree

    def _update_prompt(self):
        """Update the prompt with current project tree."""
        self.instruction = self.prompt_template.format(project=self.project_tree)

    @property
    def functions(self):
        """File editing functions."""
        return ["OVERWRITE", "DELETE", "APPEND", "REPLACE"]

    def _get_file_path(self, text):
        """Returns a cleaned file path from a function call."""
        pattern = "|".join(self.functions)
        file_path = re.sub(f"({pattern}) ", "", text)
        file_path = (
            file_path.replace("$PROJECT_PATH", self.project_path)
            .replace("`", "")
            .replace("'", "")
            .replace('"', "")
        )
        return file_path

    def forward(self, message: str = "", depth: int = 0) -> str:
        """Generate a response to a user message."""
        self._update_prompt()
        logger.debug("Calling LLM")
        self.messages.append(Message("user", message))

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        request = self.completion_args
        messages = [asdict(x) for x in self.messages]

        request["messages"] = [{"role": "system", "content": self.instruction}] + messages

        try:
            response = client.chat.completions.create(**request)
            logger.debug(response)
        except Exception as ex:
            raise ex

        self.input_tokens.append(response.usage.prompt_tokens)
        self.output_tokens.append(response.usage.completion_tokens)
        response_content = response.choices[0].message.content
        self.messages.append(Message("assistant", response_content))
        messages.append({"role": "assistant", "content": response_content})

        parser_content = ""
        chunk_idx = 0
        chunks = process_chunks(split_markdown(response_content), self.functions)
        while chunk_idx < len(chunks):
            chunk_type, content = chunks[chunk_idx]["type"], chunks[chunk_idx]["content"],

            if chunk_type == "terminal":
                old_terminal = self.terminal_session.content
                terminal = self.terminal_session(content)
                parser_content += (
                    f"\n{TRIPLE_BACKTICKS}\n"
                    + terminal[len(old_terminal) :].strip("\n")
                    + f"\n{TRIPLE_BACKTICKS}\n"
                )
                yield messages + [{"role": "user", "content": parser_content.strip()}]

            elif chunk_type == "ipython":
                old_python = self.python_session.content
                python = self.python_session(content + "\n")
                parser_content += (
                    f"\n{TRIPLE_BACKTICKS}\n"
                    + python[len(old_python) :].strip("\n")
                    + f"\n{TRIPLE_BACKTICKS}\n"
                )
                yield messages + [{"role": "user", "content": parser_content.strip()}]

            elif chunk_type == "text" and content.startswith("OVERWRITE"):
                logger.info("OVERWRITE workflow")
                file_path = self._get_file_path(content)
                if not os.path.exists(file_path):
                    parser_content += f"\nFile {file_path} does not exist.\n"
                    yield messages + [{"role": "user", "content": parser_content.strip()}]
                    break
                with open(file_path, "w") as file:
                    file.write(chunks[chunk_idx + 1]["content"])
                parser_content += (
                    f"\nContents of {file_path} successfully overwritten.\n"
                )
                yield messages + [{"role": "user", "content": parser_content.strip()}]
                chunk_idx += 1

            elif chunk_type == "text" and content.startswith("APPEND"):
                logger.info("APPEND workflow")
                file_path = self._get_file_path(content)
                if not os.path.exists(file_path):
                    parser_content += f"\nFile {file_path} does not exist.\n"
                    yield messages + [{"role": "user", "content": parser_content.strip()}]
                    break
                with open(file_path, "a") as file:
                    file.write("\n" + chunks[chunk_idx + 1]["content"])
                parser_content += f"\nContents successfully append to {file_path}.\n"
                yield messages + [{"role": "user", "content": parser_content.strip()}]
                chunk_idx += 1

            elif chunk_type == "text" and content.startswith("DELETE"):
                logger.info("DELETE workflow")
                file_path = self._get_file_path(content)
                if not os.path.exists(file_path):
                    parser_content += f"\nFile {file_path} does not exist.\n"
                    yield messages + [{"role": "user", "content": parser_content.strip()}]
                    break
                with open(file_path, "r") as file:
                    file_contents = file.read()
                delete_content = chunks[chunk_idx + 1]["content"]
                if delete_content not in file_contents:
                    parser_content += f"\nContent to delete not found in {file_path}.\n"
                    yield messages + [{"role": "user", "content": parser_content.strip()}]
                    break
                file_contents = file_contents.replace(delete_content, "")
                with open(file_path, "w") as file:
                    file.write(file_contents)
                parser_content += f"\nContents successfully deleted from {file_path}.\n"
                yield messages + [{"role": "user", "content": parser_content.strip()}]
                chunk_idx += 1

            elif chunk_type == "text" and content.startswith("REPLACE"):
                logger.info("REPLACE workflow")
                file_path = self._get_file_path(content)
                if not os.path.exists(file_path):
                    parser_content += f"\nFile {file_path} does not exist.\n"
                    yield messages + [{"role": "user", "content": parser_content.strip()}]
                    break
                with open(file_path, "r") as file:
                    file_contents = file.read()
                old_content = chunks[chunk_idx + 1]["content"]
                new_content = chunks[chunk_idx + 2]["content"]
                if old_content not in file_contents:
                    parser_content += f"\nContent to replace not found in {file_path}.\n"
                    yield messages + [{"role": "user", "content": parser_content.strip()}]
                    break
                file_contents = file_contents.replace(old_content, new_content)
                with open(file_path, "w") as file:
                    file.write(file_contents)
                parser_content += f"\nContents successfully replaced in {file_path}.\n"
                yield messages + [{"role": "user", "content": parser_content.strip()}]
                chunk_idx += 2

            chunk_idx += 1

        parser_content = parser_content.strip()
        if parser_content:
            yield messages + [{"role": "user", "content": parser_content}]
            for chunk in self.forward(parser_content, depth + 1):
                yield chunk
        else:
            yield messages

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
                new_hist = []
                for idx in range(0, len(chunk), 2):
                    if idx + 1 < len(chunk):
                        new_hist.append((chunk[idx]["content"], chunk[idx + 1]["content"]))
                    else:
                        new_hist.append((chunk[idx]["content"], None))
                yield new_hist

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


@dataclass
class ChatbotLauncher(Script):
    """Launch a tmux module gradio gui."""
    prompt_name: str = field(default="codebuddy-openai", metadata={"help": "The name of the prompt config yaml file."})
    max_calls: int = field(default=5, metadata={"help": "Maximum number of LLM API calls."})
    python_env: str = field(
        default=os.path.dirname(os.path.dirname(__file__)) + "/codebuddy-venv",
        metadata={"help": "Path to the Python environment."}
    )
    project_path: str = field(default="~/demo", metadata={"help": "Path to the project directory."})
    share: bool = field(default=False, metadata={"help": "If True, launches public gradio."})

    def __post_init__(self):
        self.project_path = os.path.expanduser(self.project_path).rstrip("/")
        self.python_env = os.path.expanduser(self.python_env)
        if not os.path.exists(self.project_path):
            raise NotADirectoryError(f"Project path {self.project_path} does not exist.")
        if not os.path.exists(self.python_env):
            raise NotADirectoryError(f"Python env {self.python_env} does not exist.")

    def run(self):
        prompt_basepath = os.path.dirname(os.path.dirname(__file__))

        prompt_path = os.path.join(prompt_basepath, "prompts", self.prompt_name + ".yaml")
        module = TmuxModule(
            config_path=prompt_path,
            max_calls=self.max_calls,
            python_env=self.python_env,
            project_path=self.project_path
        )
        gui = module.get_gradio_interface()
        gui.launch(share=self.share)


if __name__ == "__main__":
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO)
    load_dotenv()
    ChatbotLauncher.parse_args().run()
