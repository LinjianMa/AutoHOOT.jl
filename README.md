# AutoHOOT

This package is a Julia wrapper around [AutoHOOT](https://github.com/LinjianMa/AutoHOOT), a Python-based automatic differentiation framework targeting high-order optimization for large scale tensor computations.

# Installation

1. Clone AutoHOOT locally, for example:
```bash
git clone https://github.com/LinjianMa/AutoHOOT.git ~/software/AutoHOOT
```
2. Install the local AutoHOOT Python package you just cloned (you will need to have Python installed on your system):
```bash
pip install -e ~/software/AutoHOOT
```
3. Open up Julia (first [download it](https://julialang.org/downloads/) if you don't already have it), install PyCall, tell it to use the desired version of Python in your path, and then build PyCall to use that version of Python:
```julia
julia> ENV["PYTHON"] = "python" # Or specify a path to your local version of Python

julia> using Pkg

julia> Pkg.build("PyCall")
```
4. Install AutoHOOT.jl:
```julia
julia> Pkg.add("https://github.com/LinjianMa/AutoHOOT.jl.git")
```

Now you should be able to start using AutoHOOT:
```julia
julia> using AutoHOOT

julia> const ad = AutoHOOT.autodiff
AutoHOOT.autodiff

julia> const gops = AutoHOOT.graphops
AutoHOOT.graphops

julia> A = ad.Variable(name = "A", shape = [2, 2])
PyObject A

julia> out = ad.einsum("ab,bc->ac", A, ad.einsum("ab,bc->ac", ad.identity(2), ad.identity(2)))
PyObject T.einsum('ab,bc->ac',A,T.einsum('ab,bc->ac',T.identity(2),T.identity(2)))

julia> out_simplified = gops.simplify(out)
[2021-06-04 10:08:34,420 graph_optimizer.py:138] Start fusing einsum
[2021-06-04 10:08:34,420 graph_optimizer.py:190] Generated new subscript: ab,bd,dc->ac
[2021-06-04 10:08:34,421 graph_transformer.py:292] Rewrite to new subscript: ab->ab
[2021-06-04 10:08:34,421 graph_optimizer.py:138] Start fusing einsum
[2021-06-04 10:08:34,421 graph_optimizer.py:190] Generated new subscript: ab->ab
PyObject A
```

To disable AutoHOOT loggings, run 
```
set_logger(disabled=true)
```

# Examples

Here is an example of generating optimal sequences of einsum expressions for the gradients of a network contraction with respect to each tensor in the network:
<details>
 <summaryClick me! ></summary>
<p>
 
```julia
using AutoHOOT

set_logger(disabled=true)

const ad = AutoHOOT.autodiff
const go = AutoHOOT.graphops

x1 = ad.Variable(name = "x1", shape = [2, 3])
x2 = ad.Variable(name = "x2", shape = [3, 4])
x3 = ad.Variable(name = "x3", shape = [4, 5])
x4 = ad.Variable(name = "x4", shape = [5, 6])
x5 = ad.Variable(name = "x5", shape = [6, 2])

println("\nTensor shapes we want to contract:")
@show [x1.shape, x2.shape, x3.shape, x4.shape, x5.shape]

ein = ad.einsum("ij,jk,kl,lm,mi->", x1, x2, x3, x4, x5)

println("\nOriginal einsum expression for the contraction we want to take the gradient of:")
@show ein

ein_opt = go.optimize(ein)

println("\nOptimized contraction sequence:")
@show ein_opt

ein_grads = ad.gradients(ein_opt, [x1, x2, x3, x4, x5])

println("\nEinsum expressions for computing the gradients:")
display(ein_grads)

ein_grads_cache = ad.find_topo_sort(ein_grads)

println("\nEinsum expressions for computing the gradients with caching:")
display(ein_grads_cache)
```
 </p>
</details>

which outputs:
<details>
 <summaryClick me! ></summary>
<p>
 
```julia
Tensor shapes we want to contract:
[x1.shape, x2.shape, x3.shape, x4.shape, x5.shape] = [[2, 3], [3, 4], [4, 5], [5, 6], [6, 2]]

Original einsum expression for the contraction we want to take the gradient of:
ein = PyObject T.einsum('ij,jk,kl,lm,mi->',x1,x2,x3,x4,x5)

Optimized contraction sequence:
ein_opt = PyObject T.einsum('ab,ab->',T.einsum('ab,cb->ac',T.einsum('ab,cb->ac',T.einsum('bc,ca->ab',x4,x5),x3),x2),x1)

Einsum expressions for computing the gradients:
5-element Vector{PyCall.PyObject}:
 PyObject T.einsum('ab,->ab',T.einsum('ab,cb->ac',T.einsum('ab,cb->ac',T.einsum('bc,ca->ab',x4,x5),x3),x2),1.0)
 PyObject T.einsum('ac,ab->bc',T.einsum('ab,cb->ac',T.einsum('bc,ca->ab',x4,x5),x3),T.einsum('ab,->ab',x1,1.0))
 PyObject T.einsum('ac,ab->bc',T.einsum('bc,ca->ab',x4,x5),T.einsum('bc,ab->ac',x2,T.einsum('ab,->ab',x1,1.0)))
 PyObject T.einsum('ca,ab->bc',x5,T.einsum('bc,ab->ac',x3,T.einsum('bc,ab->ac',x2,T.einsum('ab,->ab',x1,1.0))))
 PyObject T.einsum('bc,ab->ca',x4,T.einsum('bc,ab->ac',x3,T.einsum('bc,ab->ac',x2,T.einsum('ab,->ab',x1,1.0))))

Einsum expressions for computing the gradients with caching:
17-element Vector{PyCall.PyObject}:
 PyObject x4
 PyObject x5
 PyObject T.einsum('bc,ca->ab',x4,x5)
 PyObject x3
 PyObject T.einsum('ab,cb->ac',T.einsum('bc,ca->ab',x4,x5),x3)
 PyObject x2
 PyObject T.einsum('ab,cb->ac',T.einsum('ab,cb->ac',T.einsum('bc,ca->ab',x4,x5),x3),x2)
 PyObject 1.0
 PyObject T.einsum('ab,->ab',T.einsum('ab,cb->ac',T.einsum('ab,cb->ac',T.einsum('bc,ca->ab',x4,x5),x3),x2),1.0)
 PyObject x1
 PyObject T.einsum('ab,->ab',x1,1.0)
 PyObject T.einsum('ac,ab->bc',T.einsum('ab,cb->ac',T.einsum('bc,ca->ab',x4,x5),x3),T.einsum('ab,->ab',x1,1.0))
 PyObject T.einsum('bc,ab->ac',x2,T.einsum('ab,->ab',x1,1.0))
 PyObject T.einsum('ac,ab->bc',T.einsum('bc,ca->ab',x4,x5),T.einsum('bc,ab->ac',x2,T.einsum('ab,->ab',x1,1.0)))
 PyObject T.einsum('bc,ab->ac',x3,T.einsum('bc,ab->ac',x2,T.einsum('ab,->ab',x1,1.0)))
 PyObject T.einsum('ca,ab->bc',x5,T.einsum('bc,ab->ac',x3,T.einsum('bc,ab->ac',x2,T.einsum('ab,->ab',x1,1.0))))
 PyObject T.einsum('bc,ab->ca',x4,T.einsum('bc,ab->ac',x3,T.einsum('bc,ab->ac',x2,T.einsum('ab,->ab',x1,1.0))))
```
 </p>
</details>
