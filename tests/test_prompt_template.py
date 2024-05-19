import pytest
from codebuddy.utils import PromptTemplate

def test_prompt_template_format_single_key():
    template = PromptTemplate("Hello, {{name}}!")
    result = template.format(name="World")
    assert result == "Hello, World!"

def test_prompt_template_format_multiple_keys():
    template = PromptTemplate("Hello, {{name}}! Welcome to {{place}}.")
    result = template.format(name="Alice", place="Wonderland")
    assert result == "Hello, Alice! Welcome to Wonderland."

def test_prompt_template_format_missing_key():
    template = PromptTemplate("Hello, {{name}}! Welcome to {{place}}.")
    result = template.format(name="Alice")
    assert result == "Hello, Alice! Welcome to {{place}}."

def test_prompt_template_format_no_keys():
    template = PromptTemplate("Hello, World!")
    result = template.format()
    assert result == "Hello, World!"

def test_prompt_template_format_key_not_in_template():
    template = PromptTemplate("Hello, World!")
    result = template.format(name="Alice")
    assert result == "Hello, World!"
