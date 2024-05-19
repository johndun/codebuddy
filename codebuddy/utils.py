import re
from dataclasses import dataclass
import subprocess
from typing import List


TRIPLE_BACKTICKS = "` ` `".replace(" ", "")


def run_bash(command_str):
    command_str = re.sub(r'pip(?! --no-input)', r'pip --no-input', command_str)
    try:
        return subprocess.run(
            command_str,
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        ).stdout
    except:
        return None


@dataclass
class Message:
    """A message in a dialog."""
    role: str  #: The role of the message sender
    content: str  #: The content of the message

    def __repr__(self):
        return f"{self.role}: {self.content}"


@dataclass
class Dialog:
    """A dialog consisting of a list of messages."""
    turns: List[Message]  #: A list of messages in the dialog

    def __repr__(self):
        return "\n".join([repr(msg) for msg in self.turns])


@dataclass
class PromptTemplate:
    """A string prompt template with keys marked by double curly braces."""
    template: str  #: The prompt template string or list of strings

    def format(self, **kwargs) -> str:
        """Replace template keys with provided values."""
        template = self.template
        for k, v in kwargs.items():
            kk = "{{" + k + "}}"
            if kk in template:
                template = template.replace(kk, v)
        return template


def split_markdown(text):
    # Regular expression to match markdown code blocks
    code_block_pattern = re.compile(rf'{TRIPLE_BACKTICKS}(\w+)?\n(.*?){TRIPLE_BACKTICKS}', re.DOTALL)

    chunks = []
    last_index = 0

    # Find all code blocks
    for match in code_block_pattern.finditer(text):
        # Add the text before the code block
        if last_index < match.start():
            chunks.append({
                "type": "text",
                "content": text[last_index:match.start()]  # .strip()
            })

        # Add the code block
        language = match.group(1) if match.group(1) else "code"
        code_content = match.group(2)
        chunks.append({
            "type": language,
            "content": code_content
        })

        last_index = match.end()

    # Add the remaining text after the last code block
    if last_index < len(text):
        chunks.append({
            "type": "text",
            "content": text[last_index:].strip()
        })
    chunks = [x for x in chunks if x["content"]]
    return chunks


def filter_content(content, keywords):
    """
    Filters lines in the content that start with any of the specified keywords.

    Parameters:
    content (str): The content to be filtered.
    keywords (list of str): The keywords to filter by.

    Returns:
    str: The filtered content with lines starting with one of the keywords.
    """
    filtered_lines = [line for line in content.splitlines() if any(line.startswith(keyword) for keyword in keywords)]
    return '\n'.join(filtered_lines)


def process_chunks(chunks, keywords):
    """
    Processes and filters the chunks based on the specified criteria.

    Parameters:
    chunks (list of dict): A list of dictionaries with fields "type" and "content".
    keywords (list of str): A list of keywords to filter "content" in text-type chunks.

    Returns:
    list of dict: A filtered and processed list of chunks.
    """

    filtered_chunks = []
    for chunk in chunks:
        if chunk['type'] != "text":
            filtered_chunks.append(chunk)
        else:
            filtered_content = filter_content(chunk['content'], keywords)
            if filtered_content:
                filtered_chunks.append({
                    'type': chunk['type'],
                    'content': filtered_content
                })

    return filtered_chunks
