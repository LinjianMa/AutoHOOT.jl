module backend

using PyCall
include("utils.jl")

export set_backend,
    context,
    tensor,
    is_tensor,
    shape,
    ndim,
    to_numpy,
    copy,
    transpose,
    ones,
    ones_like,
    zeros,
    zeros_like,
    sum,
    norm,
    dot,
    power,
    array_equal,
    einsum,
    random,
    seed,
    tensorinv

const T = PyNULL()

function __init__()
    copy!(T, pyimport_conda("autohoot.backend", "autohoot"))
end

@func_loader T set_backend context tensor is_tensor shape ndim to_numpy copy transpose ones ones_like zeros zeros_like
@func_loader T sum norm dot power array_equal einsum random seed tensorinv

end
