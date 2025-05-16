# # Performance is not associative: Matrix Multiplication
#
# Inspired by [this blog post](https://www.johndcook.com/blog/2017/12/12/efficiency-is-not-associative-for-matrix-multiplication/),
# this notebook demonstrate how we can use cost-based extraction to get the
# optimal ordering of matrix multiplication using its associativity property.
import numpy as np
from egglog import (
    EGraph,
    function,
    i64,
    i64Like,
    rewrite,
    rule,
    ruleset,
    set_,
    subsume,
    union,
)
from sealir import ase
from sealir.eqsat.py_eqsat import (
    Py_MatMultIO,
)
from sealir.eqsat.rvsdg_convert import egraph_conversion
from sealir.eqsat.rvsdg_eqsat import (
    GraphRoot,
    Term,
)
from sealir.eqsat.rvsdg_extract import (
    egraph_extraction,
)
from sealir.rvsdg import format_rvsdg
from sealir.rvsdg import grammar as rg

from ch01_basic_compiler import frontend
from ch04_1_typeinfer_ifelse import TypeFloat64
from ch05_typeinfer_array import (
    ArrayDesc,
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
    setup_argtypes,
)

SExpr = ase.SExpr

# ## NumPy examples:
#
# ### Basic: Three Matrix Multiplication

M = 200
N = 100
K = 200

# (M x N) (N x K) (K x N)

if __name__ == "__main__":
    arr0 = np.random.random((M, N))
    arr1 = np.random.random((N, K))
    arr2 = np.random.random((K, N))
    res0 = (arr0 @ arr1) @ arr2
    res1 = arr0 @ (arr1 @ arr2)
    print(res0.shape)
    print(res1.shape)
    np.testing.assert_allclose(res0, res1)
    # unoptimized
    # %timeit arr0 @ arr1 @ arr2
    # optimized
    # %timeit arr0 @ (arr1 @ arr2)

# ## Five matrix multiplications

f0 = lambda arr0, arr1, arr2, arr3: arr0 @ arr1 @ arr2 @ arr1 @ arr2 @ arr3


def f1(arr0, arr1, arr2, arr3):
    x = arr1 @ arr2
    return arr0 @ (x @ (x @ arr3))


if __name__ == "__main__":
    arr0 = np.random.random((M, N))
    arr1 = np.random.random((N, K))
    arr2 = np.random.random((K, N))
    arr3 = np.random.random((N, 1))
    res0 = f0(arr0, arr1, arr2, arr3)
    res1 = f1(arr0, arr1, arr2, arr3)
    np.testing.assert_allclose(res0, res1)

    # unoptimized
    # %timeit f0(arr0, arr1, arr2, arr3)
    # optimized
    # %timeit f1(arr0, arr1, arr2, arr3)

# ## Egraph


array_0_desc, array_0_infos = array_desc_rules(
    "array_0", shape=(M, N), dtype=TypeFloat64, layout="c"
)
array_1_desc, array_1_infos = array_desc_rules(
    "array_1", shape=(N, K), dtype=TypeFloat64, layout="c"
)
array_2_desc, array_2_infos = array_desc_rules(
    "array_2", shape=(K, N), dtype=TypeFloat64, layout="c"
)
array_3_desc, array_3_infos = array_desc_rules(
    "array_3", shape=(N, 1), dtype=TypeFloat64, layout="c"
)
ruleset_array_facts = ruleset(
    *array_0_infos, *array_1_infos, *array_2_infos, *array_3_infos
)


@function
def MatMul(lhs: Term, rhs: Term) -> Term: ...


@function
def MatMul_KnownShape(
    lhs: Term, rhs: Term, m: i64Like, n: i64Like, k: i64Like
) -> Term: ...


@function
def ArrayDescOp(term: Term) -> ArrayDesc: ...


@ruleset
def ruleset_optimize_matmul(
    ary0: Term,
    ary1: Term,
    ary2: Term,
    ad0: ArrayDesc,
    ad1: ArrayDesc,
    shapeM: i64,
    shapeN: i64,
    shapeK: i64,
):
    # associative
    yield rewrite(MatMul(MatMul(ary0, ary1), ary2)).to(
        MatMul(ary0, MatMul(ary1, ary2))
    )

    yield rule(
        # array desc propagation
        ary2 == MatMul(ary0, ary1),
        ad0.toType() == TypeVar(ary0).getType(),
        ad1.toType() == TypeVar(ary1).getType(),
        ad0.ndim == i64(2),
        ad1.ndim == i64(2),
        m := ad0.dim(0),
        q := ad1.dim(1),
        ad0.dim(1) == ad1.dim(0),
        ad0.dtype == ad1.dtype,
    ).then(
        # set output dims
        ad2 := ArrayDescOp(ary2),
        set_(TypeVar(ary2).getType()).to(ad2.toType()),
        set_(ad2.ndim).to(2),
        set_(ad2.dim(0)).to(m),
        set_(ad2.dim(1)).to(q),
        set_(ad2.dtype).to(ad0.dtype),
    )

    yield rule(
        ary2 == (stmt := MatMul(ary0, ary1)),
        ad0.toType() == TypeVar(ary0).getType(),
        ad1.toType() == TypeVar(ary1).getType(),
        ad0.ndim == i64(2),
        ad1.ndim == i64(2),
        Dim.fixed(shapeM) == ad0.dim(0),
        Dim.fixed(shapeN) == ad0.dim(1),
        Dim.fixed(shapeN) == ad1.dim(0),
        Dim.fixed(shapeK) == ad1.dim(1),
        ad0.dtype == ad1.dtype,
    ).then(
        # Convert to MatMul_KnownShape
        union(ary2).with_(
            MatMul_KnownShape(ary0, ary1, shapeM, shapeN, shapeK)
        ),
        subsume(stmt),
    )


@ruleset
def ruleset_matmul_op(io: Term, lhs: Term, rhs: Term, res: Term):
    yield rule(res == Py_MatMultIO(io, lhs, rhs)).then(
        union(res.getPort(1)).with_(MatMul(lhs, rhs)),
        union(res.getPort(0)).with_(io),
    )


# ## Compile


def code_input(ary0, ary1, ary2, ary3):
    return ary0 @ ary1 @ ary2 @ ary1 @ ary2 @ ary3


if __name__ == "__main__":
    rvsdg_expr, _dbginfo = frontend(code_input)
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
                array_1_desc.toType(),
                array_2_desc.toType(),
                array_3_desc.toType(),
            )
            | base_ruleset
            | ruleset_matmul_op
            | ruleset_optimize_matmul
        ).saturate()
    )


# ## Cost Model
#
# We estimate the cost of matrix-multiplication to be `2 * m * n * k`


class MyCostModel(_ch05_CostModel):
    def get_cost_function(self, nodename, op, ty, cost, children):
        match op, tuple(children):
            case "MatMul_KnownShape", (_lhs, _rhs, m, n, k):
                return self.get_equation(
                    lambda lhs, rhs, *_, m, n, k: 2 * m * n * k,
                    constants=dict(m=m, n=n, k=k),
                )
        return super().get_cost_function(nodename, op, ty, cost, children)


# +
class NbOp_MatMulKnownShape(NbOp_Base):
    lhs: SExpr
    rhs: SExpr

    m: int
    n: int
    k: int


class MyEGraphToRVSDG(ExtendEGraphToRVSDG):

    def handle_Term(self, op: str, children: dict | list, grm: Grammar):
        match op, children:
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
        return super().handle_Term(op, children, grm)


# -

if __name__ == "__main__":
    cost, out_expr = egraph_extraction(
        egraph=eg,
        rvsdg_sexpr=rvsdg_expr,
        cost_model=MyCostModel(),
        converter_class=MyEGraphToRVSDG,
    )
    print(cost)
    print(format_rvsdg(out_expr))


# ## Extract Python AST


class SourceMaker(ase.TreeVisitor):
    def __init__(self):
        super().__init__()
        self.memo = {}
        self.out = []

    def visit(self, expr: SExpr):
        memo = self.memo
        buf = self.out
        match expr:
            case rg.Attrs():
                return None
            case rg.RegionBegin(attrs=attrs, inports=inports):
                memo[expr] = [f"arg{i}" for i, v in enumerate(inports)]
                self.nargs = len(inports) - 1
            case rg.Unpack(val=val, idx=idx):
                val = memo[val]
                memo[expr] = val[idx]
            case NbOp_MatMulKnownShape(lhs=lhs, rhs=rhs, m=m, n=n, k=k):
                lhs, rhs = memo[lhs], memo[rhs]
                memo[expr] = ref = f"v{len(memo)}"
                buf.append(f"{ref} = ({lhs} @ {rhs})")
            case _:
                raise ValueError(expr)

    def get_source(self):
        source = "; ".join(self.out)
        args = [f"arg{i}" for i in range(1, self.nargs + 1)]
        last = f"v{len(self.memo) - 1}"
        template = f"""
def func({', '.join(args)}):
    {source}
    return {last}
"""
        return template

    def get_function(self):
        source_code = self.get_source()
        g = {}
        exec(source_code, g)
        return g["func"]


if __name__ == "__main__":
    visitor = SourceMaker()

    outports = out_expr.body.ports
    [out_equ] = [p.value for p in outports if p.name == "!ret"]
    ase.apply_bottomup(out_equ, visitor)
    extracted = visitor.get_function()

    res1 = extracted(arr0, arr1, arr2, arr3)
    res0 = f1(arr0, arr1, arr2, arr3)
    np.testing.assert_allclose(res0, res1)

    # %timeit f0(arr0, arr1, arr2, arr3)

    # %timeit f1(arr0, arr1, arr2, arr3)

    # %timeit extracted(arr0, arr1, arr2, arr3)
