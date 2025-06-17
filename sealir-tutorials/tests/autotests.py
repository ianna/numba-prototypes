import ast
import importlib
import linecache
import os
import sys
from collections import ChainMap
from contextlib import ExitStack
from textwrap import dedent
from types import ModuleType


def autotest_notebook(mod: ModuleType):
    with ExitStack() as raii:
        old_stdin = sys.stdin

        def restore_old_stdin(*args):
            sys.stdin = old_stdin

        sys.stdin = raii.enter_context(open(os.devnull, "r"))
        raii.push(restore_old_stdin)

        # Extract main block source codes
        main_blocks = get_main_code_blocks(mod)
        print(f"Found {len(main_blocks)} main blocks in {mod}")

        ns = {}
        ns.update(mod.__dict__)

        # Execute them manually
        for i, (code, lineno, synthetic_filename) in enumerate(main_blocks):
            print(f"Executing block {i+1} at {lineno}".center(80, "-"))
            print(code)
            exec_with_linecache(code, synthetic_filename, ns, ns)


def get_main_code_blocks(mod: ModuleType) -> list[tuple[str, int, str]]:
    """
    Extract code from 'if __name__ == "__main__"' blocks and return them as
    executable source strings.

    Returns:
        List of source code strings that can be executed with exec()
    """
    modname = mod.__name__
    file = mod.__file__
    with open(file, "r") as fin:
        source = fin.read()
    astree = ast.parse(source)

    source_lines = source.splitlines()
    main_code_blocks = []

    # Walk through the AST to find if __name__ == "__main__" blocks
    for node in ast.walk(astree):
        if isinstance(node, ast.If):
            # Check if this is an "if __name__ == '__main__'" condition
            if _is_main_guard(node.test):
                # Extract the source code for the body of the if statement
                code_block = _extract_source_from_body(node.body, source_lines)
                if code_block.strip():  # Only add non-empty blocks
                    # Create a synthetic filename for this block
                    synthetic_filename = (
                        f"<{modname}_main_block_{len(main_code_blocks)+1}>"
                    )

                    # Cache the source in linecache
                    _cache_source_in_linecache(code_block, synthetic_filename)

                    main_code_blocks.append(
                        (code_block, node.lineno, synthetic_filename)
                    )

    return main_code_blocks


def _cache_source_in_linecache(source_code: str, filename: str):
    """
    Cache source code in linecache for inspect.getsource compatibility.
    """
    # Ensure source ends with newline for proper line counting
    if not source_code.endswith("\n"):
        source_code += "\n"

    lines = source_code.splitlines(True)  # Keep line endings

    # Cache in linecache: (size, mtime, lines, fullname)
    linecache.cache[filename] = (
        len(source_code),
        None,  # mtime - None for synthetic files
        lines,
        filename,
    )


def exec_with_linecache(
    source_code: str,
    synthetic_filename: str,
    globals_dict: dict,
    locals_dict: dict,
):
    """
    Execute source code with linecache support for inspect.getsource.

    Args:
        source_code: The source code to execute
        synthetic_filename: The synthetic filename for linecache
        globals_dict: Global namespace
        locals_dict: Local namespace

    Returns:
        The locals_dict after execution (useful for capturing defined
        functions/variables)
    """
    # Compile with the synthetic filename
    compiled_code = compile(source_code, synthetic_filename, "exec")

    # Execute the compiled code
    exec(compiled_code, globals_dict, locals_dict)

    return locals_dict


def _is_main_guard(test_node) -> bool:
    """
    Check if the given test node represents 'if __name__ == "__main__"'
    """
    if isinstance(test_node, ast.Compare):
        # Check for __name__ == "__main__"
        if (
            isinstance(test_node.left, ast.Name)
            and test_node.left.id == "__name__"
            and len(test_node.ops) == 1
            and isinstance(test_node.ops[0], ast.Eq)
            and len(test_node.comparators) == 1
            and isinstance(test_node.comparators[0], ast.Constant)
            and test_node.comparators[0].value == "__main__"
        ):
            return True
    return False


def _extract_source_from_body(
    body_nodes: list[ast.stmt], source_lines: list[str]
) -> str:
    """
    Extract the original source code from AST body nodes.
    """
    if not body_nodes:
        return ""

    code_lines = []

    for stmt in body_nodes:
        if hasattr(stmt, "lineno"):
            start_line = stmt.lineno - 1  # Convert to 0-based indexing
            if hasattr(stmt, "end_lineno") and stmt.end_lineno:
                end_line = stmt.end_lineno
            else:
                end_line = start_line + 1

            # Extract the source lines for this statement
            for line_num in range(
                start_line, min(end_line, len(source_lines))
            ):
                original_line = source_lines[line_num]
                # Remove the original indentation and add consistent indentation
                stripped_line = original_line
                if stripped_line:  # Skip empty lines
                    code_lines.append(stripped_line)
                else:
                    code_lines.append(
                        ""
                    )  # Preserve empty lines for readability

    return dedent("\n".join(code_lines))
