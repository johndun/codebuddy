import pytest
from codebuddy.utils import Dialog, Message

def test_dialog_initialization():
    messages = [Message(role="user", content="Hello"), Message(role="assistant", content="Hi there!")]
    dialog = Dialog(turns=messages)
    assert len(dialog.turns) == 2
    assert dialog.turns[0].role == "user"
    assert dialog.turns[0].content == "Hello"
    assert dialog.turns[1].role == "assistant"
    assert dialog.turns[1].content == "Hi there!"

def test_dialog_repr():
    messages = [Message(role="user", content="Hello"), Message(role="assistant", content="Hi there!")]
    dialog = Dialog(turns=messages)
    expected_repr = "user: Hello\nassistant: Hi there!"
    assert repr(dialog) == expected_repr
