using AutoHOOT
using Test

const ad = AutoHOOT.autodiff
const gops = AutoHOOT.graphops

@testset "test rewrite expression" begin
    a1 = ad.Variable(name = "a1", shape = [3, 2])
    a2 = ad.Variable(name = "a2", shape = [2, 3])
    x = ad.einsum("ik,kj->ij", a1, a2)
    y = ad.einsum("sm,ml->sl", a1, a2)
    gops.rewrite_einsum_expr(x)
    gops.rewrite_einsum_expr(y)
    @test x.einsum_subscripts == y.einsum_subscripts
end

@testset "test optimizer and simplify" begin
    A = ad.Variable(name = "A", shape = [2, 2])
    out = ad.einsum("ab,bc->ac", A, ad.einsum("ab,bc->ac", ad.identity(2), ad.identity(2)))
    newout_optimize = gops.optimize(out)
    newout_simplify = gops.simplify(out)
    @test newout_optimize == A
    @test newout_simplify == A
end
