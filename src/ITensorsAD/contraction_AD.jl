using ITensors
using ..autodiff

const ad = autodiff

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

"""Generate ITensor input network based on AutoHOOT einsum expression
Parameters
----------
outnode: AutoHOOT einsum node
node_dict: A dictionary mapping AutoHOOT node to ITensor tensor
Returns
-------
A list of ITensor tensors
"""
function generate_network(out_node, node_dict)
    einstr = out_node.einsum_subscripts
    input_nodes = out_node.inputs
    str_input, _ = split(einstr, "->")
    str_in_list = split(str_input, ",")

    tensor_list = []
    index_dict = Dict{Char,Index{Int64}}()
    for (i, str) in enumerate(str_in_list)
        node = input_nodes[i]
        index_list = []
        for (j, char) in enumerate(collect(str))
            if haskey(index_dict, char) == false
                index_dict[char] = Index(node.shape[j], string(char))
            end
            push!(index_list, index_dict[char])
        end
        tensor = replaceinds(node_dict[node], node_dict[node].inds => index_list)
        push!(tensor_list, tensor)
    end
    return tensor_list
end

"""Generate AutoHOOT einsum expression based on ITensor input network
Parameters
----------
network: An array of ITensor tensors
Returns
-------
An AutoHOOT einsum node;
A dictionary mapping AutoHOOT input node to ITensor tensor
"""
function generate_einsum_expr(network::Array)
    input_nodes, node_dict = input_nodes_generation(network)
    einstr = einstr_generation(network)
    return ad.einsum(einstr, input_nodes...), node_dict
end

"""Generate AutoHOOT nodes based on ITensor tensors
Parameters
----------
network: An array of ITensor tensors
Returns
-------
node_list: An array of AutoHOOT nodes
node_dict: A dictionary mapping AutoHOOT node to ITensor tensor
"""
function input_nodes_generation(network::Array)
    node_list = []
    node_dict = Dict()
    for (i, tensor) in enumerate(network)
        nodename = "tensor" * Char('0' + i)
        shape = [index.space for index in tensor.inds]
        node = ad.Variable(nodename, shape = shape)
        push!(node_list, node)
        node_dict[node] = tensor
    end
    return node_list, node_dict
end

function einstr_generation(network::Array)
    index_dict = Dict{Index{Int64},Char}()
    newchar = 'a'
    string_list = []
    # build input string
    for tensor in network
        str = ""
        for index in tensor.inds
            if haskey(index_dict, index) == false
                index_dict[index] = newchar
                newchar += 1
            end
            str = str * index_dict[index]
        end
        push!(string_list, str)
    end
    instr = join(string_list, ",")
    # build output string
    output_inds = noncommoninds(network...)
    outstr = join([index_dict[i] for i in output_inds])
    return instr * "->" * outstr
end
