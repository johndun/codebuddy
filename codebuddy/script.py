from dataclasses import dataclass
from typing import List

from transformers import HfArgumentParser


@dataclass
class Script:
    """A base class to be used in an experiment script.

    Dataclass fields can be parsed from the command line arguments.

    ### Usage

    1. Define script arguments as subclass fields.
    2. Define run method.
    3. Call cls.parse_args().run() in main.

    ### Notes

    * Make sure your subclass is also a dataclass.
    * Arguments defined in subclass must have defaults and help metadata

    ### Examples

    ```
    @dataclass
    class Greeter(ScriptRunner):
        name: str = field(metadata={"help": "A name."}, default="John")

        def run(self):
            print(f"Hi {self.name}")

    if __name__ == "__main__":
        Greeter.parse_args().run()
    ```
    """

    def _subset_args(self, keys: List[str]) -> dict:
        """Returns a dictionary of keys with not None values."""
        return {k: v for k, v in self.__dict__.items() if k in keys and v is not None}

    @classmethod
    def _docstring(cls) -> str:
        """Returns the class docstring."""
        return cls.__doc__.strip() if cls.__doc__ else ""

    @classmethod
    def parse_args(cls) -> "Script":
        """Parses command-line arguments into a ScriptArgs instance."""
        parser = HfArgumentParser(cls)
        parser.description = cls._docstring()
        return parser.parse_args_into_dataclasses()[0]

    def run(self):
        """To be overridden run method."""
        raise NotImplementedError
