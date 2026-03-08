import json
from pathlib import Path

from .benchmark import SyntaxGymSingleTSE, SyntaxGym2020

with open(Path(__file__).parent / "test_suites.json") as json_file:
    test_suite_dict = json.load(json_file)


def build_syntaxgym_suite(identifier: str) -> SyntaxGymSingleTSE:
    if identifier not in test_suite_dict:
        raise ValueError(f"Unknown SyntaxGym suite '{identifier}'.")
    return SyntaxGymSingleTSE(identifier=identifier, suite_ref=test_suite_dict[identifier])


__all__ = ["SyntaxGymSingleTSE", "SyntaxGym2020", "build_syntaxgym_suite", "test_suite_dict"]
