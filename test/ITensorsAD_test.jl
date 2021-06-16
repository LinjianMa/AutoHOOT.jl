using AutoHOOT
using ITensors

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

    node, dict = itensorad.generate_einsum_expr([A, C, B])
    network = itensorad.extract_network(node, dict)
    network2 = itensorad.generate_network(node, dict)
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

    node, dict = itensorad.generate_einsum_expr([A, B, C, D, E])
    node = go.generate_optimal_tree(node)
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
