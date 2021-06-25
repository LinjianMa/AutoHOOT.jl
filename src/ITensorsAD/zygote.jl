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

scalar(A::ITensor) = ITensors.scalar(A)

@adjoint function scalar(A::ITensor)
    adjoint_pullback(s) = (ITensor(s),)
    return scalar(A), adjoint_pullback
end

"""Extract the scalar from an array with only 1 element.
"""
function extract_scalar(v::Array)
    @assert length(v) == 1
    return scalar(v[1])
end

@adjoint extract_scalar(v::Array) = extract_scalar(v), g -> ([ITensor(g)],)

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
    innodes = [retrieve_key(node_dict, t) for t in variables]
    jacobians_graph = []
    for n in nodes
        jac = ad.gradients(n, innodes)
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
    function vjps(vector::Array)
        vjp_output = []
        out_size = length(jacobians[1])
        in_size = length(jacobians)
        @assert(length(vector) == in_size)
        for u_index = 1:out_size
            out = jacobians[1][u_index] * vector[1]
            for i = 2:in_size
                out += jacobians[i][u_index] * vector[i]
            end
            push!(vjp_output, out)
        end
        return (nothing, Tuple(vjp_output)...)
    end
    return forward_tensors, vjps
end
