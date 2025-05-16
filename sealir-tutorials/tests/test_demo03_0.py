from demo03_0_matmul_assoc import *


def test_extraction_result():

    arr0 = np.random.random((M, N))
    arr1 = np.random.random((N, K))
    arr2 = np.random.random((K, N))
    arr3 = np.random.random((N, 1))

    rvsdg_expr, _dbginfo = frontend(code_input)
    memo = egraph_conversion(rvsdg_expr)
    root = GraphRoot(memo[rvsdg_expr])
    eg = EGraph()
    eg.let("root", root)
    eg.run(
        (
            ruleset_array_facts
            | setup_argtypes(
                array_0_desc.toType(),
                array_1_desc.toType(),
                array_2_desc.toType(),
                array_3_desc.toType(),
            )
            | base_ruleset
            | ruleset_matmul_op
            | ruleset_optimize_matmul
        ).saturate()
    )
    cost, out_expr = egraph_extraction(
        egraph=eg,
        rvsdg_sexpr=rvsdg_expr,
        cost_model=MyCostModel(),
        converter_class=MyEGraphToRVSDG,
    )
    visitor = SourceMaker()

    outports = out_expr.body.ports
    [out_equ] = [p.value for p in outports if p.name == "!ret"]
    ase.apply_bottomup(out_equ, visitor)

    source = visitor.get_source()
    print(source)
    assert (
        source.strip()
        == """
def func(arg1, arg2, arg3, arg4):
    v4 = (arg2 @ arg3); v6 = (v4 @ arg4); v7 = (v4 @ v6); v8 = (arg1 @ v7)
    return v8
""".strip()
    )

    extracted = visitor.get_function()

    res1 = extracted(arr0, arr1, arr2, arr3)
    res0 = f1(arr0, arr1, arr2, arr3)
    np.testing.assert_allclose(res0, res1)
