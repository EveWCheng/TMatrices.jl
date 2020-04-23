module TMat

export LS

include("LS.jl")


SquareWell(R ; width=1., height=0.1) = (R < width ? height : 0.)
Gaussian(R ; width=1., height=0.1) = height*exp(-1/2 * (R/width)^2)

end # module
