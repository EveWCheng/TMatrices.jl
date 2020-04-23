module ODE

using DanUtils
using Constants
#using Scattering
using BesselFunctions
# using Plots, UnitfulRecipes
# pyplot()

kFromE(E,m=1mₑ) = uconvert(u"a0^-1", sqrt(2*m*E) / ħ)
EFromk(k,m=1mₑ) = uconvert(Eh, ħ^2*k^2/2m )

# This is solving for tildeu = r*tildeR
# function DerivFunc(u,u′,r, E,k,potential,l,m=1mₑ)
#     prefac = 2m/ħ^2

#     # Expressing this as the plane wave expansion with the Y_l0 at the end,
#     # gives rise to the sqrt() prefactors.
#     jl = sqrt(4pi)*sqrt(2l+1) * 1/k * ricj(l, k*r)
#     #jl = sqrt(2l+1) * 1/k * ricj(l, k*r)
#     U = -prefac*potential(r)

#     u′′ = @. (l*(l+1)/r^2)*u - ((prefac*E + U)*u + U*jl)

#     du = u′
#     du′ = u′′

#     return du,du′
# end
function DerivFunc(u,u′,r, E,k,potential,l,m=1mₑ)
    prefac = 2m/ħ^2

    # Expressing this as the plane wave expansion with the Y_l0 at the end,
    # gives rise to the sqrt() prefactors.
    jl = sqrt(4pi)*sqrt(2l+1) * 1/k * ricj(l, k*r)
    #jl = sqrt(2l+1) * 1/k * ricj(l, k*r)
    U = -prefac*potential(r)


    # ΔE = ħ^2*k^2/2m - E
    u′′ = @. (l*(l+1)/r^2)*u - ((prefac*E + U)*u + U*jl)
    # u′′ = @. (l*(l+1)/r^2)*u - ((prefac*E + U)*u + prefac*ΔE*jl)

    du = u′
    du′ = u′′

    return du,du′
end

using DifferentialEquations
# RenormCond(u,t,int) = any(abs2.(u) .> 1e3)
function RenormCond(u,t,int)
    # Dodgy for now
    # t > int.sol.t[1]+0.1
    false
end
function RenormAffect!(int)
    # maxval = sqrt.(maximum(abs2.(int.u), dims=1))

    # int.u ./= maxval
    # for u_i = int.sol.u
    #     u_i ./= maxval
    # end

    # if cond(int.u) > 1e3
        @warn "Going to terminate because cond" int.t cond(int.u)
        terminate!(int)
    # end
end
function SolveOutwards(r_list, E,k,potential,l)

    # Going to deal with unitless in the deriv funcs
    # Note: wavefunctions are unitless in the picture with \phi_k = (2\pi)^-3/2 exp(ikr)
    # But u = r*R so there is units for u.

    # prob_func = (u,p,t) -> DerivFunc(u,t, E,k,potential,l)  

    # unitless_potential = r -> ustrip(u"Eₕ", potential(r*u"a₀"))
    prob_func = function (f,p,r)
        r = r*a0
        u = f[1,:] .* a0
        u′ = f[2,:]
        du,du′ = DerivFunc(u, u′,r, E, k, potential,l)
        #@show du du′
        [ustrip.(NoUnits, du)' ; ustrip.(1/a0, du′)']
    end

    # TODO: Callback for exp growth. Also do orthogonalising here.

    # Not sure if I will need to make this a more carefully chosen derivative.
    # At least the value being zero will be necessary.
    u0 = [0. -1.0 ; 0. 1.0]
    rspan = extrema(r_list)
    local sol
    prev_solns = zeros(ComplexF64,2,2,0)
    prev_r = typeof(1.0a0)[]
    while true
        prob = ODEProblem(prob_func, u0, ustrip.(a0, rspan))
        sol = solve(prob, Tsit5(), abstol=1e-10, reltol=1e-12, callback=DiscreteCallback(RenormCond, RenormAffect!))

        this_r = filter(x -> rspan[1] <= x <= sol.t[end]*a0, r_list)
        this_soln = sol.(ustrip.(a0,this_r))
        this_soln = reduce((x,y) -> cat(x,y,dims=3), this_soln)

        u = sol.u[end][1,:]
        u′ = sol.u[end][2,:]
        coeffs = MatchSolutionUnitless(u, u′, sol.t[end], ustrip(1/a0,k), l)

        conv = TransformToPureIngoingOutgoing(coeffs)

        prev_r = vcat(prev_r, this_r)
        prev_solns = cat(prev_solns, this_soln, dims=3)
        # prev_solns = mapslices(mat -> mat*conv, prev_solns, dims=[1,2])
        
        # new_start = (sol.t[end]*a0 + rspan[1])/2.
        # rspan = (new_start, rspan[2])
        rspan = (this_r[end], rspan[2])
        u0 = prev_solns[:,:,end]

        if rspan[1] == rspan[2]
            break
        end


    end

    # out_r_list = filter(x -> rspan[1] <= x <= rspan[2], r_list)

    #return sol
    # out = sol.(ustrip.(a0,out_r_list))
    # out = reduce((x,y) -> cat(x,y;dims=3), out)

    # out ./= out[1:1,:,end:end]

    # return out_r_list, out

    u = prev_solns[1,:,:] .* a0
    u′ = prev_solns[2,:,:]
    return prev_r, u, u′
end

function SolveOutwards2(rspan, E,k,potential,l ; init_derivs=[0.0,1.0])

    # Going to deal with unitless in the deriv funcs
    # Note: wavefunctions are unitless in the picture with \phi_k = (2\pi)^-3/2 exp(ikr)
    # But u = r*R so there is units for u.

    # prob_func = (u,p,t) -> DerivFunc(u,t, E,k,potential,l)  

    # unitless_potential = r -> ustrip(u"Eₕ", potential(r*u"a₀"))
    prob_func = function (f,p,r)
        r = r*a0
        u = f[1,:] .* a0
        u′ = f[2,:]
        du,du′ = DerivFunc(u, u′,r, E, k, potential,l)
        #@show du du′
        [ustrip.(NoUnits, du)' ; ustrip.(1/a0, du′)']
    end

    # TODO: Callback for exp growth. Also do orthogonalising here.

    # Not sure if I will need to make this a more carefully chosen derivative.
    # At least the value being zero will be necessary.
    u0 = ComplexF64[[0. 0.] ; permutedims(init_derivs)]

    # Needs to include the jl part that is in f, as the BC is for f not tildef.
    jl = sqrt(4pi)*sqrt(2l+1) * 1/k * ricj(l, k*rspan[1])
    # jl′ = sqrt(4pi)*sqrt(2l+1) * 1/k * dricjdr(l, k, rspan[1])
    # u0 .+= [jl ; jl′]
    u0 .-= [ustrip(a0, jl) ; 0.]
    # @show u0

    prob = ODEProblem(prob_func, u0, ustrip.(a0, rspan))
    sol = solve(prob, Tsit5(), abstol=1e-10, reltol=1e-12)

    this_soln = reduce((x,y) -> cat(x,y,dims=3), sol.u)
    r = sol.t .* a0

    u = this_soln[1,:,:] .* a0
    u′ = this_soln[2,:,:]
    return r, u, u′
    # return r, this_soln
end

function MatchBackwards(sets, coeffs)

end

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

MatchSolution(sol, k, l) = MatchSolution(UnitfulSolution(sol[end],sol.t[end])..., k, l)
function MatchSolution(u, u′, r, k, l)
    u = ustrip.(a0,u)
    u′ = u′
    r = ustrip(a0,r)
    k = ustrip(1/a0,k)
    # E = ustrip(Eh,E)
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
function TildeF(f, r, k, l)
    jl = @. sqrt(4pi)*sqrt(2l+1) * 1/k * ricj(l, k*r)
    jl′ = @. sqrt(4pi)*sqrt(2l+1) * 1/k * dricjdr(l, k, r)

    jl = ustrip.(a0,jl)
    if ndims(f) == 2
        tildef = f .- vcat(jl,jl′)
    elseif ndims(f) == 3
        tildef = f .- reshape(vcat(jl,jl), 2, 1, :)
    end
end
function MatchSolutionUnitless(f, r, κ, l)
    M = [rich₊(l,κ*r)    rich₋(l,κ*r) ;
         drich₊dr(l,κ,r) drich₋dr(l,κ,r)]
    coeffs = M\f
end
UnitfulSolution(f,r) = f[1,:].*a0, f[2,:], r*a0

function TransformToPureIngoingOutgoing(C)
    x1 = -C[2,2] / (C[2,1] - C[2,2])
    x2 = -C[1,2] / (C[1,1] - C[1,2])

    M = [[x1;(1-x1)] [x2;(1-x2)]]
    # display(C*M)
    # display(C)
    # display(M)
    # error()
    return M
end

function ConvertToIngoingOutgoing(f, r, k, l)
    coeffs = MatchSolution(f[:,:,end], r[end], k, l)
    # display(coeffs)
    M = TransformToPureIngoingOutgoing(coeffs)
    # M = permutedims(M)
    display(M)

    return mapslices(x->x*M, f, dims=(1,2))
end

function EliminateIngoing(u, u′, r, k, l)
    coeffs = MatchSolution(u[:,end], u′[:,end], r[end], k, l)
    display(coeffs)
    M = TransformToPureIngoingOutgoing(coeffs)
    M = M[:,1:1]
    M = permutedims(M)

    return dropdims(M*u, dims=1),
           dropdims(M*u′, dims=1)
end

using EllipsisNotation
function AddBesselPart(r, tildeu, tildeu′, k, l)
    # tildeu = tildef[1,..]
    # tildeu′ = tildef[2,..]
    u = sqrt(4pi)*sqrt(2l+1) * 1/k * ricj.(l,k*r) + tildeu
    u′ = sqrt(4pi)*sqrt(2l+1) * 1/k * dricjdr.(l,k,r) + tildeu′

    # f = vcat(ustrip(a0,u), u′)
    return u,u′
end

using LegendrePolynomials
using QuadGK
using Dierckx
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
    #out *= sqrt(2l+1) * 1/p * Pl(costh,l)
    # Adjusting for my lack of (2π)^-3/2 in the wavefunctions
    out /= (2π)^3
end

fFromT(T,m=mₑ) = uconvert(a0, -(2pi)^2*m/ħ^2 * T)

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
end
