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

