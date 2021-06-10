using ITensors
using ..autodiff
using ..graphops

const ad = autodiff
const go = graphops

include("utils.jl")

# Wrapper of the transposed_vjp function:
# https://github.com/LinjianMa/AutoHOOT/blob/master/autohoot/autodiff.py#L1462
"""Take vector-jacobian product of output_network with respect to each tensor in tensor_list.
Parameters
----------
output_network: output node that we are taking derivative of.
tensor_list: list of nodes that we are taking derivative wrt.
input_vector: input vector in the vjps.
Returns
-------
mathematically, it is calculating (v^T @ J)^T
A list of vjp values, one for each node in tensor_list respectively.
The returned list shapes are the same as the tensor_list shapes.
"""
function contraction_transposed_vjp(output_network::Array, tensor_list::Array, input_vector)
    #TODO
    return
end

function generate_optimal_tree(network::Array)
    node, dict = generate_einsum_expr(network)
    node = go.generate_optimal_tree(node)
    return extract_network(node, dict)
end
