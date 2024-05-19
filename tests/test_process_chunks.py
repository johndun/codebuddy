from codebuddy.utils import process_chunks


def test_process_chunks_no_text():
    chunks = [
        {"type": "code", "content": "print('Hello, World!')"},
        {"type": "code", "content": "print('Another code block')"},
    ]
    keywords = ["Hello"]
    result = process_chunks(chunks, keywords)
    assert result == chunks


def test_process_chunks_with_text():
    chunks = [
        {"type": "text", "content": "This is a test.\nHello World!"},
        {"type": "code", "content": "print('Hello, World!')"},
        {"type": "text", "content": "Another text block.\nHello again!"},
    ]
    keywords = ["Hello"]
    expected_result = [
        {"type": "text", "content": "Hello World!"},
        {"type": "code", "content": "print('Hello, World!')"},
        {"type": "text", "content": "Hello again!"},
    ]
    result = process_chunks(chunks, keywords)
    assert result == expected_result


def test_process_chunks_empty_text():
    chunks = [
        {"type": "text", "content": "This is a test.\nNo match here."},
        {"type": "code", "content": "print('Hello, World!')"},
    ]
    keywords = ["Hello"]
    expected_result = [{"type": "code", "content": "print('Hello, World!')"}]
    result = process_chunks(chunks, keywords)
    assert result == expected_result


def test_process_chunks_mixed():
    chunks = [
        {"type": "text", "content": "This is a test.\nHello World!"},
        {"type": "code", "content": "print('Hello, World!')"},
        {"type": "text", "content": "Another text block.\nHello again!"},
        {"type": "code", "content": "print('Another code block')"},
    ]
    keywords = ["Hello"]
    expected_result = [
        {"type": "text", "content": "Hello World!"},
        {"type": "code", "content": "print('Hello, World!')"},
        {"type": "text", "content": "Hello again!"},
        {"type": "code", "content": "print('Another code block')"},
    ]
    result = process_chunks(chunks, keywords)
    assert result == expected_result


def test_process_chunks_no_keywords():
    chunks = [
        {"type": "text", "content": "This is a test.\nHello World!"},
        {"type": "code", "content": "print('Hello, World!')"},
        {"type": "text", "content": "Another text block.\nHello again!"},
    ]
    keywords = []
    expected_result = [{"type": "code", "content": "print('Hello, World!')"}]
    result = process_chunks(chunks, keywords)
    assert result == expected_result
