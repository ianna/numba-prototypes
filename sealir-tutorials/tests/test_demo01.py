from demo01_gelu_tanh_approx import *


def test_demo01_baseline():
    llvm_module, func_egraph = compiler.lower_py_fn(
        gelu_tanh_forward,
        argtypes=(Float32,),
        ruleset=(
            base_ruleset | setup_argtypes(TypeFloat32) | additional_rules
        ),
    )
    compiler.run_backend_passes(llvm_module)
    jt = compiler.compile_module(llvm_module, func_egraph)

    assert "llvm.call @tanhf" in str(llvm_module)
    run_test(gelu_tanh_forward, jt, (0.234,))


def test_demo01_optimized():
    llvm_module, func_egraph = compiler.lower_py_fn(
        gelu_tanh_forward,
        argtypes=(Float32,),
        ruleset=(
            base_ruleset
            | setup_argtypes(TypeFloat32)
            | additional_rules
            | optimize_rules
        ),
    )
    compiler.run_backend_passes(llvm_module)
    jt = compiler.compile_module(llvm_module, func_egraph)

    # tanhf not used
    assert "llvm.call @tanhf" not in str(llvm_module)
    # powf not used
    assert "llvm.call @powf" not in str(llvm_module)
    # test correctness
    relclose = lambda x, y: np.allclose(x, y, rtol=1e-6)
    run_test(gelu_tanh_forward, jt, (0.234,), equal=relclose)
