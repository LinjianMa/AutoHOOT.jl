module graphops

using PyCall
include("utils.jl")

export
    # graph_transformer
    optimize,
    simplify,
    fuse_einsums,
    # expr_generator
    rewrite_einsum_expr,
    # graph_inv_optimizer
    optimize_inverse,
    prune_inv_node,
    # optimal_tree
    generate_optimal_tree,
    split_einsum,
    generate_optimal_tree_w_constraint,
    # graph_als_optimizer
    generate_sequential_optimal_tree

const eingraph = PyNULL()
const gops = PyNULL()

function __init__()
    copy!(eingraph, pyimport_conda("autohoot.einsum_graph", "autohoot"))
    copy!(gops, pyimport_conda("autohoot.graph_ops", "autohoot"))
end

@func_loader eingraph.expr_generator rewrite_einsum_expr
@func_loader gops.graph_transformer optimize simplify fuse_einsums
@func_loader gops.graph_inv_optimizer optimize_inverse prune_inv_node
@func_loader gops.optimal_tree generate_optimal_tree split_einsum generate_optimal_tree_w_constraint
@func_loader gops.graph_als_optimizer generate_sequential_optimal_tree

end
