module AutoHOOT

using PyCall

export set_logger

# disable loggings from the python library
function set_logger(; disabled::Bool = true)
    logging = pyimport("logging")
    logger = logging.getLogger("optimizer")
    logger.disabled = disabled
end

include("backend.jl")
include("autodiff.jl")
include("graph_ops.jl")

end
