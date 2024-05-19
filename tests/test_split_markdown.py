import pytest
from codebuddy.utils import split_markdown, TRIPLE_BACKTICKS

def test_split_markdown_no_code_blocks():
    text = "This is a simple text without any code blocks."
    expected = [{"type": "text", "content": text}]
    assert split_markdown(text) == expected

def test_split_markdown_single_code_block():
    text = f"This is a text with a code block.\n{TRIPLE_BACKTICKS}python\nprint('Hello, world!'){TRIPLE_BACKTICKS}"
    expected = [
        {"type": "text", "content": "This is a text with a code block.\n"},
        {"type": "python", "content": "print('Hello, world!')"}
    ]
    assert split_markdown(text) == expected

def test_split_markdown_multiple_code_blocks():
    text = f"This is a text with multiple code blocks.\n{TRIPLE_BACKTICKS}python\nprint('Hello, world!'){TRIPLE_BACKTICKS}\nAnd some more text.\n{TRIPLE_BACKTICKS}bash\necho 'Hello, world!'{TRIPLE_BACKTICKS}"
    expected = [
        {"type": "text", "content": "This is a text with multiple code blocks.\n"},
        {"type": "python", "content": "print('Hello, world!')"},
        {"type": "text", "content": "\nAnd some more text.\n"},
        {"type": "bash", "content": "echo 'Hello, world!'"}
    ]
    assert split_markdown(text) == expected

def test_split_markdown_code_block_without_language():
    text = f"This is a text with a code block without a specified language.\n{TRIPLE_BACKTICKS}\nprint('Hello, world!'){TRIPLE_BACKTICKS}"
    expected = [
        {"type": "text", "content": "This is a text with a code block without a specified language.\n"},
        {"type": "code", "content": "print('Hello, world!')"}
    ]
    assert split_markdown(text) == expected

def test_split_markdown_text_after_code_block():
    text = f"Text before code block.\n{TRIPLE_BACKTICKS}python\nprint('Hello, world!'){TRIPLE_BACKTICKS}\nText after code block."
    expected = [
        {"type": "text", "content": "Text before code block.\n"},
        {"type": "python", "content": "print('Hello, world!')"},
        {"type": "text", "content": "Text after code block."}
    ]
    assert split_markdown(text) == expected
