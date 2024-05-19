import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass

import logging
from pathlib import Path

from codebuddy.utils import run_bash


logger = logging.getLogger(__name__)


def _check_command_complete(output, prompt="➜"):
    prompts = [prompt] if isinstance(prompt, str) else prompt
    if any(output.endswith(x) for x in prompts):
        return True
    output_lines = output.splitlines()
    if output_lines and re.match(r"In \[\d+]:", output_lines[-1]):
        return True
    return False


@dataclass
class TmuxSession:
    session_id: str = "terminal-session"
    session_height: int = 8192
    session_width: int = 256
    sleep_duration: int = 0.5
    prefix_break_token: str = "TMUX_BREAK"
    prompt: str = "$"
    python_env: str = "~/myenv"
    project_path: str = "~"
    content: str = ""

    def __post_init__(self):
        self.project_path = os.path.expanduser(self.project_path).rstrip("/")
        self.python_env = os.path.expanduser(self.python_env)
        run_bash(
            f"tmux kill-session -t {self.session_id} 2>/dev/null || true;"
            f"tmux new-session -s {self.session_id} -d;"
            f"tmux resize-window -t {self.session_id} -x {self.session_width} -y {self.session_height}"
        )
        startup_commands = [
            f'export PS1="%c%  {self.prompt} "',
            f'export PROJECT_PATH={self.project_path}',
            f'cd $PROJECT_PATH',
            f'export PYTHONPATH={self.project_path}',
            f'source {self.python_env}/bin/activate',
            f"echo '{self.prefix_break_token}'"
        ]
        self("\n".join(startup_commands), sleep_duration=0.1)

    def __call__(self, command, sleep_duration=None):
        sleep_duration = sleep_duration if sleep_duration is not None else self.sleep_duration
        # Escape single quotes in the command
        escaped_command = command.replace("'", r"'\''")
        for cmd in escaped_command.splitlines():
            # Construct the full command with escaped command
            full_command = f"tmux send-keys -t {self.session_id} '{cmd}' C-m"
            # Use shlex to split the command into a list
            command_list = shlex.split(full_command)
            # Run the command
            logger.info(command_list)
            subprocess.run(command_list, text=True)
        # Monitor the pane output
        output = ""
        buffer_file = f"/tmp/{self.session_id}_output.txt"
        while not _check_command_complete(output, prompt=self.prompt):
            if sleep_duration:
                time.sleep(sleep_duration)
            subprocess.run(f"tmux capture-pane -t {self.session_id} -p > {buffer_file}", shell=True)
            with open(buffer_file, 'r') as f:
                output = f.read().strip("\n")
        output = re.split(f"{self.prefix_break_token}", output)[-1].strip()
        if output.endswith(self.prompt):
            output = "\n".join(output.splitlines()[:-1])
        else:
            output = re.sub(r"^In \[\d+]:\s*$", "", output, flags=re.MULTILINE).strip()
        self.content = output
        return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    terminal_session = TmuxSession()
    response = terminal_session("pwd")
    logger.info(response)

    python_session = TmuxSession(session_id="python-session")
    for cmd in ("ipython", "print('TMUX_BREAK')"):
        response = python_session(cmd)
    response = python_session("print('success')")
    logger.info(response)
