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
    gradients,
    hvp,
    hessian

const ad = PyNULL()

function __init__()
    copy!(ad, pyimport_conda("autohoot.autodiff", "autohoot"))
end

@func_loader ad Variable Matrix Constant Empty add mul sub add_byconst mul_byconst sub_byconst ones oneslike zeroslike negative power einsum norm identity tensorinv scalar transpose sum tensordot
@func_loader ad Executor jacobians jvps jtjvps gradients hvp hessian

end
