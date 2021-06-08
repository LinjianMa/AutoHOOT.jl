using Test

@testset "AutoHOOT.jl" begin
    @testset "$filename" for filename in
                             ["autodiff_test.jl", "graphops_test.jl", "ITensorsAD_test.jl"]
        println("Running $filename")
        include(filename)
    end
end
