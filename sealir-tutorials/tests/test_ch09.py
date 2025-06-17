import ch09_whole_program_compiler_driver
from ch09_whole_program_compiler_driver import *

from .autotests import autotest_notebook


def test_ch09_autotest():
    autotest_notebook(ch09_whole_program_compiler_driver)


def test_call_graph():

    source_file = "llm.py"

    try:
        with open(source_file, "r") as f:
            source_code = f.read()
    except FileNotFoundError:
        print(f"File not found: {source_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    # Create a NamespaceVisitor instance
    cgv = CallGraphVisitor(source_code, source_file)
    # Visit all nodes in the AST
    cgv.visit_all()
    received = cgv.get_call_graph()

    expected = {
        "softmax": ("np.exp", "np.max", "np.sum"),
        "scaled_dot_product_attention": (
            "query.reshape",
            "key.reshape",
            "value.reshape",
            "np.matmul",
            "np.sqrt",
            "softmax",
            "np.matmul",
            "context.reshape",
        ),
        "MultiHeadAttention.__init__": (),
        "MultiHeadAttention.split_heads": ("x.reshape", "x.transpose"),
        "MultiHeadAttention.combine_heads": ("x.transpose.reshape",),
        "MultiHeadAttention.forward": (
            "MultiHeadAttention.split_heads",
            "MultiHeadAttention.split_heads",
            "MultiHeadAttention.split_heads",
            "scaled_dot_product_attention",
            "MultiHeadAttention.combine_heads",
        ),
        "FeedForwardNetwork.__init__": ("np.random.randn", "np.random.randn"),
        "FeedForwardNetwork.forward": ("np.matmul", "np.matmul"),
        "TransformerLayer.__init__": (
            "MultiHeadAttention.__init__",
            "FeedForwardNetwork.__init__",
        ),
        "TransformerLayer.forward": (
            "MultiHeadAttention.forward",
            "FeedForwardNetwork.forward",
        ),
    }

    assert expected == received
