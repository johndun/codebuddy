from codebuddy.utils import Message

def test_message_initialization():
    msg = Message(role="user", content="Hello, world!")
    assert msg.role == "user"
    assert msg.content == "Hello, world!"

def test_message_repr():
    msg = Message(role="user", content="Hello, world!")
    assert repr(msg) == "user: Hello, world!"
