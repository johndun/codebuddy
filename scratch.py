import logging
import os

from dotenv import load_dotenv

from codebuddy import TmuxModule, Dialog

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

current_file_path = os.path.abspath(__file__)
prompts_path = os.path.dirname(current_file_path) + "/prompts"

# run_bash(f"rm -rf ~/demo; mkdir ~/demo; cp {current_file_path}/utils.py ~/demo/utils.py")

codebuddy = TmuxModule(
    config_path=prompts_path + "/codebuddy-openai.yaml",
    python_env="~/myenv",
    project_path="~/demo",
    max_calls=5
)

cmds = [
    "Can you summarize your instructions please?",
    # "Initialize git, stage all files, and submit an initial commit to the main branch"
    # "Can you summarize what is in the utils.py file?",

    # "Create an empty file call fibonnaci.py"

    # "Can you create a new file called fibonnaci.py containing a minimal basic function that returns the nth fibonnaci number?",
    # "In the file fibonnaci.py, can you add a second implementation of a function that returns the nth fibonnaci number called fib2?",
    # "In the file fibonnaci.py, can you update the `fib2` function with a google-style docstring and typing hints?"

    # "In the file fibonnaci.py, can you delete the fib2 function?"
]

for cmd in cmds:
    response = codebuddy(cmd, clear=True)
    print(response)

codebuddy._update_prompt()
print(codebuddy.instruction)
print(Dialog(codebuddy.messages))
logger.info(codebuddy.tokens)
