from dataclasses import dataclass, field
from typing import List, Tuple

from codebuddy.utils import Message


@dataclass
class Backend:
    """Backend for LLM API."""
    input_tokens: List[int] = field(
        default_factory=lambda: [],
        metadata={"help": "A list of input token counts for each message"},
    )
    output_tokens: List[int] = field(
        default_factory=lambda: [],
        metadata={"help": "A list of output token counts for each message"},
    )

    @property
    def tokens(self):
        return {
            "llm_calls": len(self.input_tokens),
            "input_tokens": sum(self.input_tokens),
            "output_tokens": sum(self.output_tokens),
        }

    def request_base(self):
        """Returns a dictionary that can be used as a base for an API request."""
        raise NotImplementedError

    def call_api(self, messages: List[Message], retries: int = 0) -> str:
        "Returns API response and updates input and output token counts."
        raise NotImplementedError
