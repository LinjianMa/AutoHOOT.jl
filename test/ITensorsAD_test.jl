using AutoHOOT
using ITensors
using Zygote

const go = AutoHOOT.graphops
const itensorad = AutoHOOT.ITensorsAD

@testset "test interface" begin
    i = Index(2, "i")
    j = Index(3, "j")
    k = Index(2, "k")
    l = Index(4, "l")

    A = randomITensor(i, j)
    B = randomITensor(j, k)
    C = randomITensor(k, l)

    out = A * B * C
    network = [A, C, B]

    nodes, dict = itensorad.generate_einsum_expr([network])
    network = itensorad.extract_network(nodes[1], dict)
    network2 = itensorad.generate_network(nodes[1], dict)
    out2 = network[1] * network[2] * network[3]
    out3 = network2[1] * network2[2] * network2[3]
    @test isapprox(storage(out), storage(out2))
    @test isapprox(storage(out), storage(out3))
end

@testset "test compute" begin
    i = Index(2, "i")
    j = Index(3, "j")
    k = Index(4, "k")
    l = Index(5, "l")
    m = Index(6, "m")

    A = randomITensor(i, j)
    B = randomITensor(j, k)
    C = randomITensor(k, l)
    D = randomITensor(l, m)
    E = randomITensor(m, i)


    out = A * B * C * D * E
    network = [A, B, C, D, E]

    nodes, dict = itensorad.generate_einsum_expr([network])
    node = go.generate_optimal_tree(nodes[1])
    out_list = itensorad.compute_graph([node], dict)
    out2 = out_list[1]

    @test isapprox(storage(out), storage(out2))
end

@testset "test optimal contraction path" begin
    i = Index(2, "i")
    j = Index(3, "j")
    k = Index(4, "k")
    l = Index(5, "l")
    m = Index(6, "m")

    A = randomITensor(i, j)
    B = randomITensor(j, k)
    C = randomITensor(k, l)
    D = randomITensor(l, m)
    E = randomITensor(m, i)

    out = A * B * C * D * E

    network = itensorad.generate_optimal_tree([A, B, C, D, E])
    out2 = contract(network)

    @test isapprox(storage(out), storage(out2))
end

@testset "test gradient" begin
    i = Index(2, "i")
    j = Index(3, "j")
    k = Index(4, "k")
    l = Index(5, "l")
    m = Index(6, "m")

    A = randomITensor(i, j)
    B = randomITensor(j, k)
    C = randomITensor(k, l)
    D = randomITensor(l, m)
    E = randomITensor(m, i)

    gradA_direct = B * C * D * E
    gradB_direct = A * C * D * E

    networks = itensorad.gradients([A, B, C, D, E], [A, B])
    gradA = contract(networks[1])
    gradB = contract(networks[2])

    @test isapprox(norm(gradA_direct), norm(gradA))
    @test isapprox(norm(gradB_direct), norm(gradB))
end

@testset "test zygote interface" begin
    i = Index(2, "i")
    j = Index(3, "j")
    k = Index(2, "k")
    A = randomITensor(i, j)
    B = randomITensor(j, k)
    C = randomITensor(k, i)

    function network(A)
        tensor_network = [A, B, C]
        out = itensorad.batch_tensor_contraction([tensor_network], A)
        return itensorad.scalar(sum(out))
    end
    grad_A = gradient(network, A)
    @test isapprox(norm(grad_A), norm(B * C))
end

@testset "test zygote interface for inner product" begin
    i = Index(2, "i")
    a = randomITensor(i)
    # build a symmetric H
    H = ITensor(i, i')
    H[i=>1, i'=>1] = 1.0
    H[i=>2, i'=>1] = 2.0
    H[i=>1, i'=>2] = 2.0
    H[i=>2, i'=>2] = 3.0

    function inner(a)
        b = prime(a)
        network = [a, H, b]
        inner = itensorad.batch_tensor_contraction([network], network...)
        return itensorad.scalar(sum(inner))
    end
    grad = gradient(inner, a)
    @test isapprox(norm(grad), norm(2 * H * a))
end

@testset "test zygote interface with sum" begin
    A = ITensor(3.0)
    B = ITensor(2.0)
    function add(A, B)
        return itensorad.scalar(sum([A, B]))
    end
    grad = gradient(add, A, B)
    @test isapprox(norm(grad[1]), norm(ITensor(1.0)))
end

@testset "test zygote interface with multiple networks" begin
    i = Index(2, "i")
    j = Index(2, "j")
    k = Index(2, "k")
    l = Index(2, "l")
    A = randomITensor(i, j)
    B = randomITensor(j, i)
    C = randomITensor(j, k)
    D = randomITensor(k, l)
    E = randomITensor(l, i)

    function inner(A)
        networks = [[A, B], [A, C, D, E]]
        contract = itensorad.batch_tensor_contraction(networks, A, B, C, D, E)
        return itensorad.scalar(sum(contract))
    end
    grad = gradient(inner, A)
    @test isapprox(norm(grad), norm(B + C * D * E))
end
