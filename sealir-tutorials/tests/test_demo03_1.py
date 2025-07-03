import demo03_1_tensor_optimization
from demo03_1_tensor_optimization import *

from .autotests import autotest_notebook


def test_demo03_1_autotest():
    autotest_notebook(demo03_1_tensor_optimization)


def test_extraction_result():
    np.random.seed(0)

    input_1 = np.random.randn(M, N)
    input_2 = np.random.randn(N, K)
    input_3 = np.random.randn(N, K)
    input_4 = np.random.randn(K, 2)

    argtypes = []
    arg_facts = []
    for i, arr_val in enumerate([input_1, input_2, input_3, input_4]):
        array_desc, array_infos = array_desc_rules(
            f"array_{i}", shape=arr_val.shape, dtype=TypeFloat64, layout="c"
        )
        argtypes.append(array_desc.toType())
        arg_facts.extend(array_infos)

    cres = compiler(
        fn=original_mma,
        argtypes=argtypes,
        extra_ruleset=extra_ruleset,
        arg_facts=ruleset(*arg_facts),
        global_ns={"np": np},
        cost_model=MyCostModel(),
        converter_class=MyEGraphToRVSDG,
        sourcemaker_class=SourceMaker,
    )
    extracted = cres.extracted

    original = original_mma(input_1, input_2, input_3, input_4)
    got = extracted(input_1, input_2, input_3, input_4)
    np.testing.assert_allclose(original, got)

    assert (
        cres.source
        == """
def func(arg1, arg2, arg3, arg4):
    v2 = np.hstack((arg1, arg1)); v5 = (arg2 @ arg4); v7 = (arg3 @ arg4); v8 = np.vstack((v5, v7)); v9 = (v2 @ v8)
    return v9
"""
    )
