module graphops

using PyCall
include("utils.jl")

export
    # graph_transformer
    optimize,
    simplify,
    rewrite_einsum_expr,
    # graph_optimizer
    fuse_einsums,
    # graph_inv_optimizer
    optimize_inverse,
    prune_inv_node,
    # graph_generator
    generate_optimal_tree,
    split_einsum,
    generate_optimal_tree_w_constraint,
    # graph_als_optimizer
    generate_sequential_optimal_tree

const gops = PyNULL()

function __init__()
    copy!(gops, pyimport_conda("autohoot.graph_ops", "autohoot"))
end

@func_loader gops.graph_transformer optimize simplify rewrite_einsum_expr
@func_loader gops.graph_optimizer fuse_einsums
@func_loader gops.graph_inv_optimizer optimize_inverse prune_inv_node
@func_loader gops.graph_generator generate_optimal_tree split_einsum generate_optimal_tree_w_constraint
@func_loader gops.graph_als_optimizer generate_sequential_optimal_tree

end
