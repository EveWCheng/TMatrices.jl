module TMatrices

export LS,
    ODE

include("quadrature.jl")
include("LS.jl")
include("ODE.jl")


"""
SquareWell(R ; width=1., height=0.1)

Sample potential, given by V = `height` for R < `width` and V = 0 otherwise.
"""
SquareWell(R ; width=1., height=0.1) = (R < width ? height : 0.)

"""
Gaussian(R ; width=1., height=0.1)

Sample potential, given by a Gaussian centred at R=0 with a width of `width` and
scaled by `height`.
"""
Gaussian(R ; width=1., height=0.1) = height*exp(-1/2 * (R/width)^2)

"""
TestCase(R)

A sample potential which includes a repulsive barrier and an attractive part, to
mimic an atomic system.
"""
TestCase(R) = SquareWell(R, width=0.5, height=10.0) + Gaussian(R, width=2., height=-0.5)

end # module
