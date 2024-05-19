import os
import re
from dataclasses import dataclass, asdict
from typing import Union

import yaml
import logging
from openai import OpenAI

from codebuddy.openai_module import OpenaiModule
from codebuddy.tmux import TmuxSession
from codebuddy.utils import PromptTemplate, run_bash, Message, split_markdown, process_chunks, TRIPLE_BACKTICKS

logger = logging.getLogger(__name__)

@dataclass
class TmuxModule(OpenaiModule):
    prompt_template: Union[str, PromptTemplate] = ""  #: The prompt template
    max_calls: int = 20  #: Maximum number of LLM API calls
    session_height: int = 8192
    session_width: int = 256
    sleep_duration: int = 1
    prefix_break_token: str = "TMUX_BREAK"
    prompt: str = "$"
    python_env: str = "/myenv"
    project_path: str = "/demo"
    terminal_session_id: str = "terminal-session"  #: The tmux session id for the terminal
    python_session_id: str = "python-session"  #: The tmux session id for ipython
    terminal_session: TmuxSession = None
    python_session: TmuxSession = None

    def __post_init__(self):
        assert self.project_path
        if self.config_path:
            with open(self.config_path, 'r') as file:
                module_data = yaml.safe_load(file)
            for field_name, value in module_data.items():
                setattr(self, field_name, value)

        if isinstance(self.prompt_template, str):
            self.prompt_template = PromptTemplate(self.prompt_template)

        self.project_path = os.path.expanduser(self.project_path).rstrip("/")
        self.python_env = os.path.expanduser(self.python_env)

        self._initialize_tmux_sessions()

    def _initialize_tmux_sessions(self):
        session_args = {
            "session_width": self.session_width,
            "session_height": self.session_height,
            "sleep_duration": self.sleep_duration,
            "prefix_break_token": self.prefix_break_token,
            "prompt": self.prompt,
            "python_env": self.python_env,
            "project_path": self.project_path
        }
        self.terminal_session = TmuxSession(session_id=self.terminal_session_id, **session_args)
        self.python_session = TmuxSession(session_id=self.python_session_id, **session_args)
        self.python_session.prompt = ">>>"
        for cmd in ("python", f"print('{self.prefix_break_token}')"):
            self.python_session(cmd)

    @property
    def project_tree(self):
        cmd_str = f'tree -I "*.pyc|__pycache__" {self.project_path}'
        project_tree = run_bash(cmd_str).strip()
        project_tree = project_tree.splitlines()
        if len(project_tree) > 1:
            project_tree = "$PROJECT_PATH\n" + "\n".join(project_tree[1:])
        return project_tree

    def _update_prompt(self):
        self.instruction = self.prompt_template.format(
            project=self.project_tree
        )

    @property
    def functions(self):
        return ["OVERWRITE", "DELETE", "APPEND", "REPLACE"]

    def _get_file_path(self, text):
        pattern = "|".join(self.functions)
        file_path = re.sub(f"({pattern}) ", "", text)
        file_path = (
            file_path
            .replace("$PROJECT_PATH", self.project_path)
            .replace("`", "")
            .replace("'", "")
            .replace('"', "")
        )
        return file_path

    def forward(self, message: str = "", depth: int = 0) -> str:
        """Generate a response to a user message."""
        self._update_prompt()
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
        logger.debug(response_content)
        self.messages.append(Message("assistant", response_content))

        parser_content = ""
        chunk_idx = 0
        chunks = process_chunks(split_markdown(response_content), self.functions)
        while chunk_idx < len(chunks):
            chunk_type, content = chunks[chunk_idx]["type"], chunks[chunk_idx]["content"]

            if chunk_type == "terminal":
                old_terminal = self.terminal_session.content
                terminal = self.terminal_session(content)
                parser_content += f"\n{TRIPLE_BACKTICKS}\n" + terminal[len(old_terminal):].strip("\n") + f"\n{TRIPLE_BACKTICKS}\n"

            elif chunk_type == "ipython":
                old_python = self.python_session.content
                python = self.python_session(content + "\n")
                parser_content += f"\n{TRIPLE_BACKTICKS}\n" + python[len(old_python):].strip("\n") + f"\n{TRIPLE_BACKTICKS}\n"

            elif chunk_type == "text" and content.startswith("OVERWRITE"):
                file_path = self._get_file_path(content)
                if not os.path.exists(file_path):
                    parser_content += f"\nFile {file_path} does not exist.\n"
                    break
                with open(file_path, "w") as file:
                    file.write(chunks[chunk_idx + 1]["content"])
                parser_content += f"\nContents of {file_path} successfully overwritten.\n"
                chunk_idx += 1

            elif chunk_type == "text" and content.startswith("APPEND"):
                file_path = self._get_file_path(content)
                if not os.path.exists(file_path):
                    parser_content += f"\nFile {file_path} does not exist.\n"
                    break
                with open(file_path, "a") as file:
                    file.write("\n" + chunks[chunk_idx + 1]["content"])
                parser_content += f"\nContents successfully append to {file_path}.\n"
                chunk_idx += 1

            elif chunk_type == "text" and content.startswith("DELETE"):
                file_path = self._get_file_path(content)
                if not os.path.exists(file_path):
                    parser_content += f"\nFile {file_path} does not exist.\n"
                    break
                with open(file_path, "r") as file:
                    file_contents = file.read()
                delete_content = chunks[chunk_idx + 1]["content"]
                if delete_content not in file_contents:
                    parser_content += f"\nContent to delete not found in {file_path}.\n"
                    break
                file_contents = file_contents.replace(delete_content, "")
                with open(file_path, "w") as file:
                    file.write(file_contents)
                parser_content += f"\nContents successfully deleted from {file_path}.\n"
                chunk_idx += 1

            elif chunk_type == "text" and content.startswith("REPLACE"):
                file_path = self._get_file_path(content)
                if not os.path.exists(file_path):
                    parser_content += f"\nFile {file_path} does not exist.\n"
                    break
                with open(file_path, "r") as file:
                    file_contents = file.read()
                old_content = chunks[chunk_idx + 1]["content"]
                new_content = chunks[chunk_idx + 2]["content"]
                if old_content not in file_contents:
                    parser_content += f"\nContent to replace     not found in {file_path}.\n"
                    break
                file_contents = file_contents.replace(old_content, new_content)
                with open(file_path, "w") as file:
                    file.write(file_contents)
                parser_content += f"\nContents successfully replaced in {file_path}.\n"
                chunk_idx += 2

            chunk_idx += 1

        if parser_content:
            return self.forward(parser_content.strip(), depth + 1)
        else:
            return response_content
