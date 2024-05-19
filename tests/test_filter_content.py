from codebuddy.utils import filter_content, TRIPLE_BACKTICKS


def test_filter_content_single_keyword():
    content = "apple\nbanana\ncherry\napple pie\nbanana split"
    keywords = ["apple"]
    expected_output = "apple\napple pie"
    assert filter_content(content, keywords) == expected_output


def test_filter_content_multiple_keywords():
    content = "apple\nbanana\ncherry\napple pie\nbanana split"
    keywords = ["apple", "banana"]
    expected_output = "apple\nbanana\napple pie\nbanana split"
    assert filter_content(content, keywords) == expected_output


def test_filter_content_no_keywords():
    content = "apple\nbanana\ncherry\napple pie\nbanana split"
    keywords = []
    expected_output = ""
    assert filter_content(content, keywords) == expected_output


def test_filter_content_no_matching_keywords():
    content = "apple\nbanana\ncherry\napple pie\nbanana split"
    keywords = ["orange"]
    expected_output = ""
    assert filter_content(content, keywords) == expected_output


def test_filter_content_partial_match():
    content = "apple\nbanana\ncherry\napple pie\nbanana split"
    keywords = ["app"]
    expected_output = "apple\napple pie"
    assert filter_content(content, keywords) == expected_output


def test_filter_content_triple_backticks():
    content = f"apple\nbanana\n{TRIPLE_BACKTICKS}python\nprint('hello')\n{TRIPLE_BACKTICKS}\napple pie\nbanana split"
    keywords = ["apple"]
    expected_output = "apple\napple pie"
    assert filter_content(content, keywords) == expected_output
