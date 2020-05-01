using DanUtils
using Constants
using BesselFunctions
using DifferentialEquations
using LegendrePolynomials
using QuadGK
using Dierckx

kFromE(E,m=1mₑ) = uconvert(u"a0^-1", sqrt(2*m*E) / ħ)
EFromk(k,m=1mₑ) = uconvert(Eh, ħ^2*k^2/2m )

"""
    DerivFunc(u,u′,r, E,k,potential,l,m=1mₑ)
    
This is the differential equation derivative function. This function should work
in Unitful quantities.
"""
function DerivFunc(u,u′,r, E,k,potential,l,m=1mₑ)
    # Expressing this as the plane wave expansion with the Y_l0 at the end,
    # gives rise to the sqrt() prefactors.
    jl = sqrt(4pi)*sqrt(2l+1) * 1/k * ricj(l, k*r)

    prefac = 2m/ħ^2
    U = -prefac*potential(r)

    u′′ = @. (l*(l+1)/r^2)*u - ((prefac*E + U)*u + U*jl)

    return u′,u′′
end


"""
    SolveOutwards(rspan, E,k,potential,l ; init_derivs=[0.0,1.0])
    
Calculates the wavefunction for the off-shell T-matrix problem. Integrates 
outwards using `DifferentialEquations` with `rspan`. The ODE derivative is
created from the `potential` given (which should handle `Unitful` quantities).
    
The incoming wave has a wavenumber of `k` and the energy that the problem is
calculated on is `E`.

TODO: Put the ODE here.

Note: the inhomogeneous part is not included in the solution. i.e. this is
solving for \tilde{f}. Also not that there are two solutions calculated, which
need to be matched to the appropriate boundary conditions using e.g. `ElminateIngoing`.
    
Returns (r,u,u′)
"""
function SolveOutwards(rspan, E,k,potential,l ; init_derivs=[0.0,1.0])
    # Going to deal with units in the deriv funcs
    # Note: wavefunctions are unitless in the picture with \phi_k = (2\pi)^-3/2 exp(ikr)
    # But u = r*R so there is units for u.

    prob_func = function (f,p,r)
        r = r*a0
        u = f[1,:] .* a0
        u′ = f[2,:]
        du,du′ = DerivFunc(u, u′,r, E, k, potential,l)
        [ustrip.(NoUnits, du)' ; ustrip.(1/a0, du′)']
    end

    # Not sure if I will need to make this a more carefully chosen derivative.
    # At least the value being zero will be necessary.
    u0 = ComplexF64[[0. 0.] ; permutedims(init_derivs)]

    # Needs to include the jl part that is in f, as the BC is for f not tildef.
    jl = sqrt(4pi)*sqrt(2l+1) * 1/k * ricj(l, k*rspan[1])
    u0 .-= [ustrip(a0, jl) ; 0.]

    prob = ODEProblem(prob_func, u0, ustrip.(a0, rspan))
    sol = solve(prob, Tsit5(), abstol=1e-10, reltol=1e-12)

    this_soln = reduce((x,y) -> cat(x,y,dims=3), sol.u)
    r = sol.t .* a0

    u = this_soln[1,:,:] .* a0
    u′ = this_soln[2,:,:]
    return r, u, u′
end

"""
Old function, hasn't been tested since included in the module.
"""
function SolveInwards(r_list, E,k,potential,l, A)

    prob_func = function (f,p,r)
        r = r*a0
        u = f[1] .* a0
        u′ = f[2]
        du,du′ = DerivFunc(u, u′,r, E, k, potential,l)
        [ustrip.(NoUnits, du) ; ustrip.(1/a0, du′)]
    end

    rspan = extrema(r_list)
    rmax = rspan[2]

    κ = kFromE(E)
    u0 = A*[rich₊(l, κ*rmax) ; ustrip(1/a0, drich₊dr(l, κ, rmax))]

    prob = ODEProblem(prob_func, u0, ustrip.(a0, reverse(rspan)))
    sol = solve(prob, Tsit5(), abstol=1e-10, reltol=1e-12)#, callback=DiscreteCallback(RenormCond, RenormAffect!))

    # this_r = filter(x -> rspan[1] <= x <= sol.t[end]*a0, r_list)
    this_r = r_list
    this_soln = sol.(ustrip.(a0,this_r))
    this_soln = reduce(hcat, this_soln)

    r = this_r
    u = this_soln[1,:] .* a0
    u′ = this_soln[2,:]
    # coeffs = MatchSolutionUnitless(u, u′, sol.t[end], ustrip(1/a0,k), l)

    # conv = TransformToPureIngoingOutgoing(coeffs)

    return r, u, u′
end

function MatchSolution(u, u′, r, k, l)
    u = ustrip.(a0,u)
    u′ = u′
    r = ustrip(a0,r)
    k = ustrip(1/a0,k)
    MatchSolutionUnitless(u,u′,r,k,l)
end
function MatchSolution(f, r, k, l)
    r = ustrip(a0,r)
    k = ustrip(1/a0,k)
    MatchSolutionUnitless(f, r, k, l)
end
function MatchSolutionUnitless(u, u′, r, k, l)
    f = [u' ; u′']
    MatchSolutionUnitless(f, r, k, l)
end
function MatchSolutionUnitless(f, r, κ, l)
    M = [rich₊(l,κ*r)    rich₋(l,κ*r) ;
         drich₊dr(l,κ,r) drich₋dr(l,κ,r)]
    coeffs = M\f
end

"""
TransformToPureIngoingOutgoing(C)

Given a coeff matrix `C`, relating two solutions obtained from `SolveOutwards`
to the ingoing/outgoing states, combine the two solutions to satisfy the
boundary conditions.

Note: because of the inhomogeneity, this must satisfy u = A·u1 + (1-A)·u2
instead of the usual freedom of u = A·u1 + B·u2.
"""
function TransformToPureIngoingOutgoing(C)
    x1 = -C[2,2] / (C[2,1] - C[2,2])
    x2 = -C[1,2] / (C[1,1] - C[1,2])

    M = [[x1;(1-x1)] [x2;(1-x2)]]
    return M
end

"""
EliminateIngoing(u, u′, r, k, l)

Matches to the boundary conditions for \tilde{u} using the output from `SolveOutwards`.
"""
function EliminateIngoing(u, u′, r, k, l)
    coeffs = MatchSolution(u[:,end], u′[:,end], r[end], k, l)
    M = TransformToPureIngoingOutgoing(coeffs)
    M = M[:,1:1]
    M = permutedims(M)

    return dropdims(M*u, dims=1),
           dropdims(M*u′, dims=1)
end


"""
    AddBesselPart(r, tildeu, tildeu′, k, l)

Includes the inhomogeneity that was missing from the ODE solution. i.e. converts
from \tilde{u} to u.
"""
function AddBesselPart(r, tildeu, tildeu′, k, l)
    u = sqrt(4pi)*sqrt(2l+1) * 1/k * ricj.(l,k*r) + tildeu
    u′ = sqrt(4pi)*sqrt(2l+1) * 1/k * dricjdr.(l,k,r) + tildeu′

    return u,u′
end

"""
Tmat_l(r,u,potential,p,costh,l)

Once a complete solution for u has been obtained, this function calculates the
`T` matrix element. The `u` going into here must be the full matched solution
and not \tilde{u}.

Note: this was before I considered T_l. This term is instead one of the Legendre
contributions of T(theta) for a specific `costh` and `l`.
"""
function Tmat_l(r,u,potential,p,costh,l)
    # One l component only. costh = p⋅z (or p⋅k more generally)
    V = r -> ustrip(Eh,potential(r*a0))
    p2 = ustrip(1/a0,p)
    r2 = ustrip.(a0,r)

    interp_r = Spline1D(r2,real.(ustrip.(a0,u)),k=1)
    interp_i = Spline1D(r2,imag.(ustrip.(a0,u)),k=1)
    interp = r -> interp_r(r) + im*interp_i(r)
    out, = quadgk(r -> interp(r)*V(r)*ricj(l,p2*r), extrema(r2)...)
    out *= a0*Eh*a0
    # These terms come from the plane wave prefactors for p.
    out *= sqrt(4pi)*sqrt(2l+1) * 1/p * Pl(costh,l)
    # Adjusting for my lack of (2π)^-3/2 in the wavefunctions
    out /= (2π)^3
end

fFromT(T,m=mₑ) = uconvert(a0, -(2pi)^2*m/ħ^2 * T)

"""
TCSandOptTheorem(faa, k)

Compare the total cross section with the optical theorem result, assuming that
`faa` is the forward scattering part (i.e. `faa` = f(θ=0)) and assuming a low
enough energy that the scattering is s-wave.

`faa` can be obtained from `fFromT`.
"""
function TCSandOptTheorem(faa, k)
    DCS = abs2(faa)
    TCS = 4pi * DCS

    opt_theorem = 4pi/k * imag(faa)

    a = sqrt(TCS/4pi)
    return (DCS=DCS, TCS=TCS, opt=opt_theorem, a=a)
end

# |faa|^2 =a^2
# -(2pi)^2*m/\hbar^2 * T = a
# hbr^2 a / (2pi)^2 / m
