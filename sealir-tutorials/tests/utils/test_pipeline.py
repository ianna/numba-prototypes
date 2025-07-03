from typing import (
    Any,
    Dict,
)

import pytest
from typing_extensions import TypedDict

from utils import Pipeline


def test_pipeline_structure_description():

    # Define TypedDict classes for return types
    class Step1Result(TypedDict):
        y: int
        step: str

    class Step2Result(TypedDict):
        z: int
        previous_step: str
        final: bool

    class OptionalResult(TypedDict, total=False):
        optional_field: str
        result: int

    # Valid functions with proper TypedDict annotations
    def step1(x: int) -> Step1Result:
        return {"y": x * 2, "step": "multiplication"}

    def step2(y: int, step: str) -> Step2Result:
        return {"z": y + 10, "previous_step": step, "final": True}

    def optional_step(x: int) -> OptionalResult:
        return {"result": x}  # optional_field is not required

    # Invalid function without TypedDict annotation
    def bad_step(x: int) -> Dict[str, Any]:
        return {"result": x}

    # Invalid function that returns wrong structure
    def wrong_structure(x: int) -> Step1Result:
        return {"wrong": x}  # Missing required 'y' and 'step' keys

    # Valid pipeline
    pipeline = Pipeline(step1, step2)
    result = pipeline(x=5)
    print(f"Result: {result}")
    print(f"Final value: {result.z}")

    # Visualize the pipeline structure (no initial kwargs needed)
    print("\n" + "=" * 60)
    print("PIPELINE STRUCTURE VISUALIZATION:")
    print(pipeline.describe())

    # Pipeline with optional fields
    optional_pipeline = Pipeline(optional_step)
    optional_result = optional_pipeline(x=42)
    print(f"Optional result: {optional_result}")

    # Visualize optional pipeline
    print("\n" + "=" * 60)
    print("OPTIONAL PIPELINE STRUCTURE:")
    print(optional_pipeline.describe())

    # Complex pipeline example
    class ProcessingResult(TypedDict):
        processed: str
        count: int

    def process_data(y: int, step: str) -> ProcessingResult:
        return {"processed": f"processed_{step}", "count": y // 2}

    complex_pipeline = Pipeline(step1, step2, process_data)
    print("\n" + "=" * 60)
    print("COMPLEX PIPELINE STRUCTURE:")
    print(complex_pipeline.describe())

    # Example of pipeline with external dependencies and optional parameters
    class ExternalInput(TypedDict):
        config: str
        multiplier: int

    def external_step(
        config: str, multiplier: int, y: int, debug: bool = False
    ) -> ExternalInput:
        return {"config": f"processed_{config}", "multiplier": multiplier * 2}

    external_pipeline = Pipeline(step1, external_step)
    print("\n" + "=" * 60)
    print("PIPELINE WITH EXTERNAL INPUTS:")
    print(external_pipeline.describe())

    with pytest.raises(TypeError) as raises:
        # This would raise TypeError at initialization:
        Pipeline(bad_step)  # Not a TypedDict return type
    raises.match("Function 'bad_step' must return a TypedDict subclass")

    # This would raise TypeError at runtime:
    wrong_pipeline = Pipeline(wrong_structure)
    with pytest.raises(TypeError) as raises:
        wrong_pipeline(x=5)  # Missing required keys
    raises.match("Function wrong_structure missing required keys")
