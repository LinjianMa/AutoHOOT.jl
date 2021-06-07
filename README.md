# AutoHOOT

This package is a Julia wrapper around [AutoHOOT](https://github.com/LinjianMa/AutoHOOT), a Python-based automatic differentiation framework targeting high-order optimization for large scale tensor computations.

# Installatoin

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
