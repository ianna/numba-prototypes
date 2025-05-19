import numpy as np
from egglog import (
    EGraph,
    String,
    Vec,
    function,
    i64,
    rewrite,
    rule,
    ruleset,
    set_,
    subsume,
    union,
)
from sealir import ase
from sealir.eqsat.py_eqsat import (
    Py_AddIO,
    Py_Call,
    Py_LoadGlobal,
)
from sealir.eqsat.rvsdg_convert import egraph_conversion
from sealir.eqsat.rvsdg_eqsat import GraphRoot, Term, TermList
from sealir.eqsat.rvsdg_extract import (
    egraph_extraction,
)
from sealir.rvsdg import format_rvsdg

from ch01_basic_compiler import frontend
from ch04_1_typeinfer_ifelse import TypeFloat64
from ch05_typeinfer_array import (
    ArrayDesc,
    Broadcast,
    Dim,
    ExtendEGraphToRVSDG,
    Grammar,
)
from ch05_typeinfer_array import MyCostModel as _ch05_CostModel
from ch05_typeinfer_array import (
    NbOp_Base,
    TypeVar,
    array_desc_rules,
    base_ruleset,
    ruleset_broadcasting,
    setup_argtypes,
)
from demo01_gelu_tanh_approx import (
    Module,
    ModuleGetAttr,
    ruleset_module,
)
from demo03_0_matmul_assoc import (
    ArrayDescOp,
)
from demo03_0_matmul_assoc import MatMul as Npy_MatMul
from demo03_0_matmul_assoc import (
    MatMul_KnownShape,
    NbOp_Base,
    NbOp_MatMulKnownShape,
    SExpr,
)
from demo03_0_matmul_assoc import SourceMaker as _demo03_0_SourceMaker
from demo03_0_matmul_assoc import (
    ruleset_optimize_matmul,
)

# --- Original version ---
# Two separate matmuls followed by elementwise add


def original_mma(input_1, input_2, input_3, input_4):
    out1 = np.matmul(input_1, input_3)
    out2 = np.matmul(input_2, input_4)
    return out1 + out2


# --- Optimized version ---
# Concatenate inputs along feature dimensions


def optimized_mma(input_1, input_2, input_3, input_4):
    concat_input = np.hstack([input_1, input_2])
    concat_weight = np.vstack([input_3, input_4])
    return np.matmul(concat_input, concat_weight)


M = 1024
N = 256
K = 256


np.random.seed(0)


input_1 = np.random.randn(M, N)
input_2 = np.random.randn(M, N)
input_3 = np.random.randn(N, K)
input_4 = np.random.randn(N, K)


if __name__ == "__main__":
    # Set seed for reproducibility

    original = original_mma(input_1, input_2, input_3, input_4)
    optimized = optimized_mma(input_1, input_2, input_3, input_4)
    # --- Comparison ---
    print("Max absolute difference:", np.max(np.abs(original - optimized)))
    print("Are they close?", np.allclose(original, optimized, atol=1e-6))


# ===============================


@ruleset
def facts_numpy_module(io: Term, name: String, op: Term, args: Vec[Term]):

    yield rule(
        op == Py_LoadGlobal(io, name),
        name == String("np"),
    ).then(set_(TypeVar(op).getType()).to(Module("numpy").toType()))

    # ------ attributes ------
    numpy_mod = Module("numpy")

    yield rule(
        op
        == (
            stmt := Py_Call(
                func=ModuleGetAttr(numpy_mod, "matmul"),
                io=io,
                args=TermList(args),
            )
        ),
        args.length() == i64(2),
    ).then(
        subsume(stmt),
        union(op.getPort(0)).with_(io),
        union(op.getPort(1)).with_(Npy_MatMul(args[0], args[1])),
    )


@ruleset
def ruleset_numpy_add(
    io: Term,
    ary0: Term,
    ary1: Term,
    aryres: Term,
    ad0: ArrayDesc,
    ad1: ArrayDesc,
    shapeM: i64,
    shapeN: i64,
    shapeK: i64,
):
    yield rule(
        aryres == Py_AddIO(io, ary0, ary1),
        ad0.toType() == TypeVar(ary0).getType(),
        ad1.toType() == TypeVar(ary1).getType(),
    ).then(
        union(aryres.getPort(0)).with_(io),
        union(aryres.getPort(1)).with_(Npy_Add(ary0, ary1)),
    )

    yield rule(
        aryres == Npy_Add(ary0, ary1),
        ad0.toType() == TypeVar(ary0).getType(),
        ad1.toType() == TypeVar(ary1).getType(),
    ).then(set_(TypeVar(aryres).getType()).to(Broadcast(ad0, ad1).toType()))


@ruleset
def ruleset_tensat(ary1: Term, ary2: Term, ary3: Term, ary4: Term):
    ewadd = Npy_Add
    matmul = Npy_MatMul
    hstack = Npy_HStack
    vstack = Npy_VStack
    yield rewrite(
        ewadd(matmul(ary1, ary3), matmul(ary2, ary4)),
    ).to(matmul(hstack(ary1, ary2), vstack(ary3, ary4)))


@ruleset
def ruleset_specialize_numpy(
    ary1: Term,
    ary2: Term,
    ary3: Term,
    ary4: Term,
    ad0: ArrayDesc,
    ad1: ArrayDesc,
    ad2: ArrayDesc,
    shapeM: i64,
    shapeN: i64,
    shapeK: i64,
):
    # EWADD
    yield rule(
        ary3 == Npy_Add(ary1, ary2),
        ad0.toType() == TypeVar(ary3).getType(),
        ad0.ndim == i64(2),
        Dim.fixed(shapeM) == ad0.dim(0),
        Dim.fixed(shapeN) == ad0.dim(1),
    ).then(union(ary3).with_(Npy_Add_KnownShape(ary1, ary2, shapeM * shapeN)))

    # HSTACK
    yield rule(
        # M x N, M x K -> M x (N+K)
        ary3 == Npy_HStack(ary1, ary2),
        ad0.toType() == TypeVar(ary1).getType(),
        ad1.toType() == TypeVar(ary2).getType(),
        ad0.ndim == i64(2),
        ad1.ndim == i64(2),
        Dim.fixed(shapeM) == ad0.dim(0),
        Dim.fixed(shapeN) == ad0.dim(1),
        Dim.fixed(shapeM) == ad1.dim(0),
        Dim.fixed(shapeK) == ad1.dim(1),
    ).then(
        union(ary3).with_(
            Npy_HStack_KnownShape(ary1, ary2, shapeM, shapeN + shapeK)
        ),
        ad2 := ArrayDescOp(ary3),
        set_(TypeVar(ary3).getType()).to(ad2.toType()),
        set_(ad2.ndim).to(2),
        set_(ad2.dim(0)).to(Dim.fixed(shapeM)),
        set_(ad2.dim(1)).to(Dim.fixed(shapeN + shapeK)),
        set_(ad2.dtype).to(ad0.dtype),
    )
    # VSTACK
    yield rule(
        # N x M, K x M -> (N+K) x M
        ary3 == Npy_VStack(ary1, ary2),
        ad0.toType() == TypeVar(ary1).getType(),
        ad1.toType() == TypeVar(ary2).getType(),
        ad0.ndim == i64(2),
        ad1.ndim == i64(2),
        Dim.fixed(shapeN) == ad0.dim(0),
        Dim.fixed(shapeM) == ad0.dim(1),
        Dim.fixed(shapeK) == ad1.dim(0),
        Dim.fixed(shapeM) == ad1.dim(1),
    ).then(
        union(ary3).with_(
            Npy_VStack_KnownShape(ary1, ary2, shapeN + shapeK, shapeM)
        ),
        ad2 := ArrayDescOp(ary3),
        set_(TypeVar(ary3).getType()).to(ad2.toType()),
        set_(ad2.ndim).to(2),
        set_(ad2.dim(0)).to(Dim.fixed(shapeN + shapeK)),
        set_(ad2.dim(1)).to(Dim.fixed(shapeM)),
        set_(ad2.dtype).to(ad0.dtype),
    )


@function
def Npy_Add(lhs: Term, rhs: Term) -> Term: ...


@function
def Npy_Add_KnownShape(lhs: Term, rhs: Term, size: i64) -> Term: ...


@function
def Npy_HStack(lhs: Term, rhs: Term) -> Term: ...


@function
def Npy_HStack_KnownShape(lhs: Term, rhs: Term, m: i64, n: i64) -> Term: ...


@function
def Npy_VStack(lhs: Term, rhs: Term) -> Term: ...


@function
def Npy_VStack_KnownShape(lhs: Term, rhs: Term, m: i64, n: i64) -> Term: ...


# ===============================


class MyCostModel(_ch05_CostModel):
    def get_cost_function(self, nodename, op, ty, cost, children):
        match op, tuple(children):
            case "Npy_Add_KnownShape", (_lhs, _rhs, m):
                return self.get_equation(
                    lambda lhs, rhs, *_, m: m, constants=dict(m=m)
                )
            case "Npy_HStack_KnownShape", (_lhs, _rhs, m, n):
                return self.get_equation(
                    lambda lhs, rhs, *_, m, n: m + n, constants=dict(m=m, n=n)
                )
            case "Npy_VStack_KnownShape", (_lhs, _rhs, m, n):
                return self.get_equation(
                    lambda lhs, rhs, *_, m, n: m + n, constants=dict(m=m, n=n)
                )
            case "MatMul_KnownShape", (_lhs, _rhs, m, n, k):
                return self.get_equation(
                    lambda lhs, rhs, *_, m, n, k: 2 * m * n * k,
                    constants=dict(m=m, n=n, k=k),
                )

            case "Npy_Add", _:
                return self.get_simple(1e99)
            case "Npy_HStack", _:
                return self.get_simple(1e99)
            case "Npy_VStack", _:
                return self.get_simple(1e99)
            case "MatMul", _:
                return self.get_simple(1e99)
        return super().get_cost_function(nodename, op, ty, cost, children)


# ===============================


class NbOp_Npy_MatMul(NbOp_Base):
    lhs: SExpr
    rhs: SExpr


class NbOp_Npy_Add_KnownShape(NbOp_Base):
    lhs: SExpr
    rhs: SExpr
    size: int


class NbOp_Npy_Add(NbOp_Base):
    lhs: SExpr
    rhs: SExpr


class NbOp_Npy_HStack(NbOp_Base):
    lhs: SExpr
    rhs: SExpr


class NbOp_Npy_HStack_KnownShape(NbOp_Base):
    lhs: SExpr
    rhs: SExpr
    m: int
    n: int


class NbOp_Npy_VStack(NbOp_Base):
    lhs: SExpr
    rhs: SExpr


class NbOp_Npy_VStack_KnownShape(NbOp_Base):
    lhs: SExpr
    rhs: SExpr
    m: int
    n: int


class MyEGraphToRVSDG(ExtendEGraphToRVSDG):

    def handle_Term(self, op: str, children: dict | list, grm: Grammar):
        match op, children:
            # case "Npy_Add", {"lhs": lhs, "rhs": rhs}:
            #     return grm.write(NbOp_Npy_Add(lhs=lhs, rhs=rhs))

            # case "Npy_HStack", {"lhs": lhs, "rhs": rhs}:
            #     return grm.write(NbOp_Npy_HStack(lhs=lhs, rhs=rhs))

            # case "Npy_VStack", {"lhs": lhs, "rhs": rhs}:
            #     return grm.write(NbOp_Npy_VStack(lhs=lhs, rhs=rhs))

            case "Npy_HStack_KnownShape", {
                "lhs": lhs,
                "rhs": rhs,
                "m": m,
                "n": n,
            }:
                return grm.write(
                    NbOp_Npy_HStack_KnownShape(lhs=lhs, rhs=rhs, m=m, n=n)
                )

            case "Npy_VStack_KnownShape", {
                "lhs": lhs,
                "rhs": rhs,
                "m": m,
                "n": n,
            }:
                return grm.write(
                    NbOp_Npy_VStack_KnownShape(lhs=lhs, rhs=rhs, m=m, n=n)
                )

            case "MatMul_KnownShape", {
                "lhs": lhs,
                "rhs": rhs,
                "m": m,
                "n": n,
                "k": k,
            }:
                return grm.write(
                    NbOp_MatMulKnownShape(lhs=lhs, rhs=rhs, m=m, n=n, k=k)
                )

            case "Npy_Add_KnownShape", {
                "lhs": lhs,
                "rhs": rhs,
                "size": size,
            }:
                return grm.write(
                    NbOp_Npy_Add_KnownShape(lhs=lhs, rhs=rhs, size=size)
                )

        return super().handle_Term(op, children, grm)


# ===============================

array_0_desc, array_0_infos = array_desc_rules(
    "array_0", shape=(M, N), dtype=TypeFloat64, layout="c"
)
array_1_desc, array_1_infos = array_desc_rules(
    "array_1", shape=(N, K), dtype=TypeFloat64, layout="c"
)
ruleset_array_facts = ruleset(*array_0_infos, *array_1_infos)


if __name__ == "__main__":
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


class SourceMaker(_demo03_0_SourceMaker):
    def visit(self, expr: SExpr):
        memo = self.memo
        buf = self.out
        match expr:
            case NbOp_Npy_HStack_KnownShape(lhs=lhs, rhs=rhs, m=m, n=n):
                lhs, rhs = memo[lhs], memo[rhs]
                memo[expr] = ref = f"v{len(memo)}"
                buf.append(f"{ref} = np.hstack(({lhs}, {rhs}))")
            case NbOp_Npy_VStack_KnownShape(lhs=lhs, rhs=rhs, m=m, n=n):
                lhs, rhs = memo[lhs], memo[rhs]
                memo[expr] = ref = f"v{len(memo)}"
                buf.append(f"{ref} = np.vstack(({lhs}, {rhs}))")
            case _:
                return super().visit(expr)


if __name__ == "__main__":
    visitor = SourceMaker()
    outports = out_expr.body.ports
    [out_equ] = [p.value for p in outports if p.name == "!ret"]
    ase.apply_bottomup(out_equ, visitor)

    print(visitor.get_source())
    extracted = visitor.get_function(global_ns={"np": np})

    got = extracted(input_1, input_2, input_3, input_4)
    np.testing.assert_allclose(original, got)

    # %timeit original_mma(input_1, input_2, input_3, input_4)
    # %timeit optimized_mma(input_1, input_2, input_3, input_4)
    # %timeit extracted(input_1, input_2, input_3, input_4)
