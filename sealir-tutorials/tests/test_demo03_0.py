import demo03_0_matmul_assoc
from demo03_0_matmul_assoc import *

from .autotests import autotest_notebook


def test_demo03_0_autotest():
    autotest_notebook(demo03_0_matmul_assoc)


def test_extraction_result():
    arr0 = np.random.random((M, N))
    arr1 = np.random.random((N, K))
    arr2 = np.random.random((K, N))
    arr3 = np.random.random((N, 1))

    argtypes = []
    arg_facts = []
    for i, arr_val in enumerate([arr0, arr1, arr2, arr3]):
        array_desc, array_infos = array_desc_rules(
            f"array_{i}", shape=arr_val.shape, dtype=TypeFloat64, layout="c"
        )
        argtypes.append(array_desc.toType())
        arg_facts.extend(array_infos)

    cres = compiler(
        fn=code_input,
        argtypes=argtypes,
        arg_facts=ruleset(*arg_facts),
        extra_ruleset=extra_ruleset,
        cost_model=MyCostModel(),
        converter_class=MyEGraphToRVSDG,
        sourcemaker_class=SourceMaker,
    )
    source = cres.source
    extracted_pyfunc = cres.extracted
    print(source)
    assert (
        source.strip()
        == """
def func(arg1, arg2, arg3, arg4):
    v5 = (arg3 @ arg4); v6 = (arg2 @ v5); v7 = (arg3 @ v6); v8 = (arg2 @ v7); v9 = (arg1 @ v8)
    return v9
""".strip()
    )

    res1 = extracted_pyfunc(arr0, arr1, arr2, arr3)
    res0 = f1(arr0, arr1, arr2, arr3)
    np.testing.assert_allclose(res0, res1)
