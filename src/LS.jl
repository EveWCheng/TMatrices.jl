
module LS

using BesselFunctions
using LegendrePolynomials
using QuadGK
using Dierckx


Δk_func(k1,k2,cosθ) = sqrt(k2^2 + k1^2 - 2*k1*k2*cosθ)

function VkkFromVΔk(k1, k2, l, VΔk_func)
    (Vl_fromspl.(VΔk_spl), k1,k2, l)
    val,err = quadgk(cosθ -> VΔk_func(Δk_func(k1,k2,cosθ)) * Pl(cosθ,l),
                     -1, 1,
                     rtol=1e-4, atol=0.)

    return 2π*val
end

# Assumes isotropic V(r)
function VΔkFromVr(Δk, Vr_func ; rmin=0, rmax=Inf)
    # Special case for Δk == 0 needed to prevent nans.
    if Δk == 0
        sphj_func = r -> 1.
    else
        sphj_func = r -> sphj(0,Δk*r)
    end
    val,err = quadgk(r -> r^2 * Vr_func(r) * sphj_func(r), rmin, rmax, rtol=1e-6, atol=1e-10)

    return 1/(2*pi^2) * val
end

function PrepareInterpedVkk(Vr_func, Δk_list ; kwds...)
    VΔk_list = @msgwrap "Calculating VΔk_spl" VΔkFromVr.(Δk_list, Ref(Vr_func) ; kwds...)
    let VΔk_spl = Spline1D(Δk_list, VΔk_list, k=1)
        (k1,k2,l) -> VkkFromVΔk(k1,k2,l, VΔk_spl)
    end
end


function SetupMatrix(k_list,
                     k_in,
                     pot_r,
                     en,
                     l,
                     u ;
                     rmin,rmax,
                     )

    @assert k_in ∈ k_list

    # I need all Vs connecting all ks to other ks
    k1 = k_list[:,newaxis]
    k2 = k_list[newaxis,:]

    Δk_list = linspace(0., 2*maximum(k_list), 1001)
    
    pot_k = PrepareInterpedVkk(pot_r, Δk_list, rmin=rmin, rmax=rmax)
    Vmat = @msgwrap "Calc Vmat" pot_k.(k1,k2, l)

    f = k_list'.^2 .* Vmat
    M = u' .* f

    ind = findfirst(==(k_in), k_list)
    RHS = Vmat[:,ind]

    return M,RHS,Vmat
end
    
include("quadrature.jl")



function Test(target_k=1.0, l=0 ; kwds...)
    # k_list = linspace(0,10,1001) .+ 0.001
    # k_in = k_list[101]
    # k_list = linspace(0,10,11) .+ 0.001
    # k_list = linspace(0,10,101) .+ 0.001
    # k_list = linspace(0,10,201) .+ 0.001
    # k_list = linspace(0,100,101) .+ 0.001
    # k_list = linspace(0,100,201) .+ 0.001
    # k_list = linspace(0,100,501) .+ 0.001
    # k_list = linspace(0,20,201) .+ 0.001
    # k_list = linspace(0,spread*target_k,N+1) |> RunningAvg
    # k_in = k_list[length(k_list)÷2]

    mass = 1.
    
    # en = 1.0 - im*1e-3
    # en = 1.0
    en = target_k^2/2mass
    # rmin,rmax = 1e-4,20.
    rmin,rmax = 1e-8,10.
    
    # l,m = 0,0
    # l′,m′ = l,m

    k_list,u = Quadrature(en ; kwds...)
    # k_list,u = KMatQuadrature(en ; kwds...)


    k_in = target_k
    @show k_in
    # k_in = target_k

    
    # potfunc = R -> Gaussian(R, height=1e-1)
    potfunc = R -> Gaussian(R, height=1.)
    # potfunc = R -> Gaussian(R, height=10.)
    # potfunc = R -> Gaussian(R, height=1e-3)

    # potfunc = R -> SquareWell(R, height=1e-2)
    # potfunc = R -> SquareWell(R, height=1e-3)
    # potfunc = R -> SquareWell(R, height=1)
    # potfunc = R -> SquareWell(R, height=1e2)
    # potfunc = R -> SquareWell(R, height=1e3)

    # potfunc = R -> SquareWell(R, height=-1)
    # potfunc = R -> SquareWell(R, height=-1e-2)
    
    M,RHS,Vmat = SetupMatrix(k_list, k_in, potfunc, rmin, rmax, en, l, u)

    T = (I - M)\RHS
    # T = (I + M)\RHS
    @AutoNT (k_list, k_in, M, RHS, en, Vmat, T, potfunc, u)
end

end
