using ITensors
using ..autodiff

const ad = autodiff

"""Compute the computational graph defined in AutoHOOT.
Parameters
----------
outnodes: A list of AutoHOOT einsum nodes
node_dict: A dictionary mapping AutoHOOT node to ITensor tensor
Returns
-------
A list of ITensor tensors.
"""
function compute_graph(out_nodes, node_dict)
    topo_order = ad.find_topo_sort(out_nodes)
    for node in topo_order
        if haskey(node_dict, node) == false
            node_dict[node] = contract([node_dict[n] for n in node.inputs])
        end
    end
    return [node_dict[node] for node in out_nodes]
end

"""Extract an ITensor network from an input network based on AutoHOOT einsum tree.
The ITensor input network is defined by the tensors in node_dict.
Note: this function ONLY extracts network based on the input nodes rather than einstr.
The output network can be hierarchical.
Parameters
----------
outnode: AutoHOOT einsum node
node_dict: A dictionary mapping AutoHOOT node to ITensor tensor
Returns
-------
A list representing the ITensor output network.
Example
-------
>>> extract_network(einsum("ab,bc->ac", A, einsum("bd,dc->bc", B, C)),
                    Dict(A => A_tensor, B => B_tensor, C => C_tensor))
>>> [A_tensor, [B_tensor, C_tensor]]
"""
function extract_network(out_node, node_dict)
    topo_order = ad.find_topo_sort([out_node])
    for node in topo_order
        if haskey(node_dict, node) == false
            node_dict[node] = [node_dict[in_node] for in_node in node.inputs]
        end
    end
    return node_dict[out_node]
end

"""Generate ITensor input network based on AutoHOOT einsum expression
Note: this function generates a network with NEW indices
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
        tensor = replaceinds(node_dict[node], inds(node_dict[node]) => index_list)
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
        shape = [space(index) for index in inds(tensor)]
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
        for index in inds(tensor)
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
