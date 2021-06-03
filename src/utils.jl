macro func_loader(class, methods...)
    blocks = Expr(:block)
    for method in methods
        block = quote
            function $(esc(method))(args...; kwargs...)
                $(class).$(method)(args...; kwargs...)
            end
        end
        push!(blocks.args, block)
    end
    blocks
end
