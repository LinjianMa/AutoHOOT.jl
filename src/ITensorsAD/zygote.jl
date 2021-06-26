using Zygote
using Zygote: @adjoint
using ITensors: setinds

const ad = autodiff
const go = graphops

@adjoint function prime(T::ITensor)
    indsT = inds(T)
    adjoint_pullback(dT::ITensor) = (setinds(dT, indsT),)
    return prime(T), adjoint_pullback
end

@adjoint function prime(T::ITensor, indices)
    indsT = inds(T)
    adjoint_pullback(dT::ITensor) = (setinds(dT, indsT), nothing)
    return prime(T, indices), adjoint_pullback
end

scalar(A::ITensor) = ITensors.scalar(A)

@adjoint function scalar(A::ITensor)
    adjoint_pullback(s) = (ITensor(s),)
    return scalar(A), adjoint_pullback
end

"""Perform a batch of tensor contractions, each one defined by a tensor network.
Parameters
----------
networks: An array of networks. Each network is represented by an array of ITensor tensors
variables: the tensors to take derivative of
Returns
-------
A list of tensors representing the contraction outputs of each network.
"""
function batch_tensor_contraction(networks, variables...)
    nodes, node_dict = generate_einsum_expr(networks)
    # TODO: add caching here
    for (i, n) in enumerate(nodes)
        nodes[i] = go.generate_optimal_tree(n)
    end
    return compute_graph(nodes, node_dict)
end

@adjoint function batch_tensor_contraction(networks, variables...)
    nodes, node_dict = generate_einsum_expr(networks)
    # TODO: add caching here
    for (i, n) in enumerate(nodes)
        nodes[i] = go.generate_optimal_tree(n)
    end
    # build jacobians graphs
    innodes_list = []
    for network in networks
        innodes = [retrieve_key(node_dict, t) for t in variables if t in network]
        push!(innodes_list, innodes)
    end
    jacobians_graph = []
    for (i, n) in enumerate(nodes)
        jac = ad.gradients(n, innodes_list[i])
        push!(jacobians_graph, jac)
    end
    # compute the graph
    forward_tensors = compute_graph(nodes, node_dict)
    jacobians = []
    for jac_graph in jacobians_graph
        jac = compute_graph(jac_graph, node_dict)
        push!(jacobians, jac)
    end
    # compute the vector-jacobian products
    function vjps(vector)
        vjp_dict = Dict()
        @assert(length(vector) == length(jacobians))
        for (i, jac) in enumerate(jacobians)
            innodes = innodes_list[i]
            for (j, t) in enumerate(jac)
                innode = innodes[j]
                if haskey(vjp_dict, innode)
                    vjp_dict[innode] = vjp_dict[innode] + vector[i] * t
                else
                    vjp_dict[innode] = vector[i] * t
                end
            end
        end
        innodes = [retrieve_key(node_dict, t) for t in variables]
        vjp_output = [vjp_dict[n] for n in innodes]
        return (nothing, Tuple(vjp_output)...)
    end
    return forward_tensors, vjps
end
