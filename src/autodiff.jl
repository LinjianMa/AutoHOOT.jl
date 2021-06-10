module autodiff

using PyCall
include("utils.jl")

export Variable,
    Matrix,
    Constant,
    Empty,
    add,
    mul,
    sub,
    add_byconst,
    mul_byconst,
    sub_byconst,
    ones,
    oneslike,
    zeroslike,
    negative,
    power,
    einsum,
    norm,
    identity,
    tensorinv,
    scalar,
    transpose,
    sum,
    tensordot,
    Executor,
    jacobians,
    jvps,
    jtjvps,
    transposed_vjps,
    gradients,
    hvp,
    hessian,
    # util function
    find_topo_sort

const ah = PyNULL()

function __init__()
    copy!(ah, pyimport_conda("autohoot", "autohoot"))
end

@func_loader ah.autodiff Variable Matrix Constant Empty add mul sub add_byconst mul_byconst sub_byconst ones oneslike zeroslike negative power einsum norm identity tensorinv scalar transpose sum tensordot
@func_loader ah.autodiff Executor jacobians jvps jtjvps transposed_vjps gradients hvp hessian
@func_loader ah.utils find_topo_sort

end
