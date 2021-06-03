using AutoHOOT
using Test

const ad = AutoHOOT.autodiff
const T = AutoHOOT.backend

@testset "AutoHOOT.jl" begin
    # Write your tests here.
    T.set_backend("numpy")

    x2 = ad.Variable(name = "x2", shape = [3, 2])
    x3 = ad.Variable(name = "x3", shape = [2, 3])
    matmul = ad.einsum("ik,kj->ij", x2, x3)
    y = ad.sum(matmul)

    grad_x2, grad_x3 = ad.gradients(y, [x2, x3])

    executor = ad.Executor([y, grad_x2, grad_x3])
    x2_val = T.tensor([[1, 2], [3, 4], [5, 6]])  # 3x2
    x3_val = T.tensor([[7, 8, 9], [10, 11, 12]])  # 2x3

    y_val, grad_x2_val, grad_x3_val =
        executor.run(feed_dict = Dict(x2 => x2_val, x3 => x3_val))

    expected_grad_sum = T.ones_like(T.dot(x2_val, x3_val))
    expected_yval = T.sum(T.dot(x2_val, x3_val))
    expected_grad_x2_val = T.dot(expected_grad_sum, T.transpose(x3_val))
    expected_grad_x3_val = T.dot(T.transpose(x2_val), expected_grad_sum)

    @test T.array_equal(y_val, expected_yval)
    @test T.array_equal(grad_x2_val, expected_grad_x2_val)
    @test T.array_equal(grad_x3_val, expected_grad_x3_val)

end
