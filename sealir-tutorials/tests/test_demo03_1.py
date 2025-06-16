from demo03_1_tensor_optimization import *


def test_extraction_result():
    np.random.seed(0)

    input_1 = np.random.randn(M, N)
    input_2 = np.random.randn(M, N)
    input_3 = np.random.randn(N, K)
    input_4 = np.random.randn(N, K)

    rvsdg_expr, _dbginfo = frontend(original_mma)
    print(format_rvsdg(rvsdg_expr))
    memo = egraph_conversion(rvsdg_expr)
    root = GraphRoot(memo[rvsdg_expr])
    eg = EGraph()
    eg.let("root", root)
    eg.run(
        (
            ruleset_array_facts
            | setup_argtypes(
                array_0_desc.toType(),
                array_0_desc.toType(),
                array_1_desc.toType(),
                array_1_desc.toType(),
            )
            | base_ruleset
            | ruleset_module
            | facts_numpy_module
            | ruleset_numpy_add
            | ruleset_broadcasting
            | ruleset_optimize_matmul
            | ruleset_tensat
            | ruleset_specialize_numpy
        ).saturate()
    )
    cost, out_expr = egraph_extraction(
        egraph=eg,
        rvsdg_sexpr=rvsdg_expr,
        cost_model=MyCostModel(),
        converter_class=MyEGraphToRVSDG,
    )
    print(format_rvsdg(out_expr))

    visitor = SourceMaker()
    outports = out_expr.body.ports
    [out_equ] = [p.value for p in outports if p.name == "!ret"]
    ase.apply_bottomup(out_equ, visitor)

    print(visitor.get_source())
    extracted = visitor.get_function(global_ns={"np": np})

    original = original_mma(input_1, input_2, input_3, input_4)
    got = extracted(input_1, input_2, input_3, input_4)
    np.testing.assert_allclose(original, got)
