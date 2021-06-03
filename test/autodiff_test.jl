using AutoHOOT
using Test

const ad = AutoHOOT.autodiff
const T = AutoHOOT.backend

T.set_backend("numpy")

@testset "test einsum" begin
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

@testset "test vjps" begin
    x = ad.Variable(name = "x", shape = [2])
    A = ad.Variable(name = "A", shape = [3, 2])
    v = ad.Variable(name = "v", shape = [3])
    y = ad.einsum("ab, b->a", A, x)

    transposed_vjp_x, = ad.transposed_vjps(y, [x], v)

    executor = ad.Executor([y, transposed_vjp_x])
    x_val = T.tensor([1.0, 2.0])  # 1x3
    A_val = T.tensor([[1.0, 2.0], [3.0, 4.0], [5, 6]])
    v_val = T.tensor([1.0, 2.0, 3.0])

    y_val, transposed_vjp_x_val =
        executor.run(feed_dict = Dict(x => x_val, A => A_val, v => v_val))

    expected_yval = T.einsum("ab, b->a", A_val, x_val)
    expected_transposed_vjp_x_val = T.einsum("b, ba->a", v_val, A_val)

    @test T.array_equal(y_val, expected_yval)
    @test T.array_equal(transposed_vjp_x_val, expected_transposed_vjp_x_val)
end

@testset "test inner product hvp" begin
    x = ad.Variable(name = "x", shape = [3, 1])
    v = ad.Variable(name = "v", shape = [3, 1])
    y = ad.sum(ad.einsum("ab,bc->ac", ad.transpose(x), x))

    grad_x, = ad.gradients(y, [x])
    Hv, = ad.hvp(output_node = y, node_list = [x], vector_list = [v])

    executor = ad.Executor([y, grad_x, Hv])
    x_val = T.tensor([[1.0], [2.0], [3]])  # 3x1
    v_val = T.tensor([[1.0], [2.0], [3]])  # 3x1
    y_val, grad_x_val, Hv_val = executor.run(feed_dict = Dict(x => x_val, v => v_val))

    expected_yval = T.sum(T.dot(T.transpose(x_val), x_val))
    expected_grad_x_val = 2 * x_val
    expected_hv_val = 2 * v_val

    @test T.array_equal(y_val, expected_yval)
    @test T.array_equal(grad_x_val, expected_grad_x_val)
    @test T.array_equal(Hv_val, expected_hv_val)
end
