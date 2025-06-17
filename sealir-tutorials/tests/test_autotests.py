import importlib
import inspect
import linecache
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path

import pytest

# Assuming your script is saved as 'paste.py'
from .autotests import (
    _cache_source_in_linecache,
    exec_with_linecache,
    get_main_code_blocks,
)


def create_temp_module(
    temp_dir: Path, content: str, module_name: str | None = None
) -> str:
    """Create a temporary module file for testing."""
    if module_name is None:
        module_name = f"test_module_{hash(content) % 10000}"

    module_file = temp_dir / f"{module_name}.py"
    module_file.write_text(content)

    # Add the directory to sys.path
    if str(temp_dir) not in sys.path:
        sys.path.insert(0, str(temp_dir))

    return module_name


def test_single_main_block():
    """Test extraction of a single main block."""
    module_content = """
def hello():
    print("Hello, world!")

def add(a, b):
    return a + b

if __name__ == "__main__":
    print("This is the main block")
    hello()
    result = add(2, 3)
    print(f"2 + 3 = {result}")
"""

    with ExitStack() as stack:
        temp_dir = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        module_name = create_temp_module(temp_dir, module_content)
        mod = importlib.import_module(module_name)

        # Cleanup sys.path on exit
        stack.callback(
            lambda: (
                sys.path.remove(str(temp_dir))
                if str(temp_dir) in sys.path
                else None
            )
        )

        blocks = get_main_code_blocks(mod)

        assert len(blocks) == 1
        code_block, line_number, synthetic_filename = blocks[0]

        # Check content
        assert 'print("This is the main block")' in code_block
        assert "hello()" in code_block
        assert "add(2, 3)" in code_block

        # Check line number and filename
        assert isinstance(line_number, int) and line_number > 0
        assert synthetic_filename.startswith(f"<{module_name}_main_block_")

        # Test linecache integration
        assert synthetic_filename in linecache.cache


def test_multiple_main_blocks_with_functions():
    """Test multiple main blocks with function definitions."""
    module_content = '''
def module_func():
    return "module function"

if __name__ == "__main__":
    print("First block")

    def local_func():
        """Function defined in main block"""
        return "local function"

    result = local_func()
    print(result)

def module_func2():
    return 42

if __name__ == "__main__":
    print("Second block")
    x = module_func2()
    print(f"Value: {x}")
'''

    with ExitStack() as stack:
        temp_dir = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        module_name = create_temp_module(temp_dir, module_content)
        mod = importlib.import_module(module_name)

        # Cleanup sys.path on exit
        stack.callback(
            lambda: (
                sys.path.remove(str(temp_dir))
                if str(temp_dir) in sys.path
                else None
            )
        )

        blocks = get_main_code_blocks(mod)

        assert len(blocks) == 2

        # Check first block
        code1, line1, filename1 = blocks[0]
        assert "First block" in code1
        assert "def local_func():" in code1

        # Check second block
        code2, line2, filename2 = blocks[1]
        assert "Second block" in code2
        assert "x = module_func2()" in code2

        # Verify they're different
        assert line1 < line2
        assert filename1 != filename2

        # Test that we can execute and inspect functions
        namespace = {}
        exec_with_linecache(code1, filename1, namespace, namespace)

        if "local_func" in namespace:
            func = namespace["local_func"]
            # Test that inspect.getsource works
            source = inspect.getsource(func)
            assert "def local_func():" in source
            assert "Function defined in main block" in source


def test_line_cache():
    """Execute all main blocks sequentially."""

    # Main block 1
    source_code_1 = """
def my_function():
    return "Hello!"
print("Block 1 executed")
    """
    synthetic_filename_1 = "<test_module_main_block_1>"

    _cache_source_in_linecache(source_code_1, synthetic_filename_1)

    # Execute with linecache support
    ns = {}
    exec_with_linecache(source_code_1, synthetic_filename_1, ns, ns)
    # Ensure this works
    src = inspect.getsource(ns["my_function"])

    exec_ns = {}
    exec(src, exec_ns)
    assert exec_ns["my_function"]() == ns["my_function"]()
