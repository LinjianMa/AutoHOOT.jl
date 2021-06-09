using AutoHOOT
using ITensors

@testset "test interface" begin
    i = Index(2, "i")
    j = Index(3, "j")
    k = Index(2, "k")
    l = Index(4, "l")

    A = randomITensor(i, j)
    B = randomITensor(j, k)
    C = randomITensor(k, l)

    out = A * B * C

    node, dict = AutoHOOT.ITensorsAD.generate_einsum_expr([A, C, B])
    network = AutoHOOT.ITensorsAD.generate_network(node, dict)
    out2 = network[1] * network[2] * network[3]
    if ITensors.version() < v"0.2"
      @test isapprox(store(out), store(out2))
    else
      @test isapprox(storage(out), storage(out2))
    end
end
