using LinearAlgebra
using BesselFunctions
using LegendrePolynomials
using QuadGK
using Dierckx
using MsgWrap

####################################
# * Potential funcs
#----------------------------------

Δk_func(k1,k2,cosθ) = sqrt(k2^2 + k1^2 - 2*k1*k2*cosθ)

function VkkFromVΔk(k1, k2, l, VΔk_func)
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

"""
    PrepareInterpedVkk(Vr_func, Δk_list)

Returns a function which will evaluate V_l(k,k′) from an isotropic V(r) given by
`Vr_func`, by first building up a set of values of V(Δk) on a grid given in
`Δk_list` and interpolating between these.
"""
function PrepareInterpedVkk(Vr_func, Δk_list ; kwds...)
    VΔk_list = @msgwrap Debug "Calculating VΔk_spl" VΔkFromVr.(Δk_list, Ref(Vr_func) ; kwds...)
    let VΔk_spl = Spline1D(Δk_list, VΔk_list, k=1)
        (k1,k2,l) -> VkkFromVΔk(k1,k2,l, VΔk_spl)
    end
end

################################
# * The Equations
#------------------------------

"""
SetupMatrix(k_list, pot_r, l, u ; rmin,rmax, pot_k=:auto)

Sets up the matrix `M` which represents the integration in the
Lippmann-Schwinger equation for the T-matrix of partial wave `l`. Returns both `M` and the potential
matrix `Vmat`, which is the potential evaluated at all (k,k′) pairs of the grid
`k_list`.

The majority of the work of the integral is actually done before this function is called, as `u`
is the integrating vector. Obtain this through `Quadrature`.

Either pass the potential in as V(r) with `pot_r` or V(k,k′) with `pot_k`. If
`pot_r` is used, then `PrepareInterpedVkk` will be called, using the values
`rmin` and `rmax`. This may be inefficient if this function is used multiple
times, and providing a `pot_k` will be faster.
"""
function SetupMatrix(k_list,
                     pot_r,
                     l,
                     u ;
                     rmin=nothing,rmax=nothing,
                     pot_k=:auto
                     )

    # I need all Vs connecting all ks to other ks
    k1 = k_list
    k2 = k_list'

    if pot_k == :auto
        Δk_list = LinRange(0., 2*maximum(k_list), 10001)
        pot_k = PrepareInterpedVkk(pot_r, Δk_list, rmin=rmin, rmax=rmax)
    end

    Vmat = @msgwrap Debug "Calc Vmat" pot_k.(k1,k2, l)

    f = k_list'.^2 .* Vmat
    M = u' .* f

    return M,Vmat
end

"""
OffShellScattering(en, l, potfunc ; rmin=1e-4, rmax=10., pot_k=:auto, kwds...)

Do a calculation of all T_l(k,k′) values for `l` and the V(r) of `potfunc`, from the k grid created by `Quadrature`
with the corresponding `kwds`.

This should work for any energy, even on-shell, but in the future it may be
optimised to not assume that the divergence at k=κ is missed.
"""
function OffShellScattering(en, l, potfunc ; rmin=1e-4, rmax=10., pot_k=:auto, kwds...)
    k_list,u = Quadrature(en ; kwds...)

    κ = sqrt(2*en)

    M,Vmat = SetupMatrix(k_list, potfunc, l, u ; rmin=rmin, rmax=rmax, pot_k=pot_k )

    T = (I - M)\Vmat

    return k_list,T
end

"""
OnShellScattering(en, l, potfunc ; rmin=1e-4, rmax=10.)

Like `OffShellScattering` but only solves for Tₗ(k,k) where ℏ²k²/2m = `en`.
Returns a single value, in contrast to `OffShellScattering`.
"""
function OnShellScattering(en, l, potfunc ; rmin=1e-4, rmax=10., kwds...)
    target_k = sqrt(2*en)

    @assert isreal(target_k)
    k_list,u = Quadrature(en ; kwds...)

    M,Vmat = SetupMatrix(k_list, potfunc, l, u ; rmin=rmin, rmax=rmax )

    ind = findfirst(==(target_k), k_list)
    RHS = Vmat[:,ind]

    T = (I - M)\RHS

    T[ind]
end

# For testing
function OnShellFromOffShell(en, l, potfunc, k_target=Float64(sqrt(2*en))  ; kwds...)
    # k_target = Float64(sqrt(2*en))

    k_list,T_mat = OffShellScattering(en, l, potfunc ; kwds...)
    Tr = Spline1D(k_list, real.(diag(T_mat)))(k_target)
    Ti = Spline1D(k_list, imag.(diag(T_mat)))(k_target)
    T = Tr + im*Ti
end


    ##############################################################
# * Applied to the Lax formalism
#------------------------------------------------------------

"""
SearchLax(k_target, potfunc, c; N=10, max_iters=101, zero_limit=false, tol=1e-5, guess=:auto, α=1.)

Finds a self-consistent solution for the equation from Lax (1952), ε = (ℏk)²/2m + cT.

This iterates on the equation until the energy converges to within `tol`. If
set, `guess` provides the starting energy, otherwise it is set to the kinetic
energy. An error is raised if `max_iters` is reached. The parameter `N` is
passed through to `Quadrature`. A dampening term can be achieved with `α` < 1.

`zero_limit` provides a way to force the imaginary-part to zero. For some
reason, the self-consistent loop seems to diverge if this isn't included.

Currently assumes s-wave scattering is sufficient to determine T.

Note: it seems to be that a unique solution is not gaurnateed. And this may find
solutions with positive imaginary parts which are unphysical.
"""
function SearchLax(k_target, potfunc, c; max_iters=101, zero_limit=false, tol=1e-5, guess=:auto, α=1., rmin, rmax, k_max=:auto, kwds...)
    K = k_target^2/2
    # This comes from a factor of 4pi from my choice of Legendre expansion (T_0 -> T(a<-a))
    # and a factor of 1/(2pi)^3 from the choice of plane wave in the # solution of T.
    TEnergy(T) = 2*pi^2 * T
    ESelfCon(T) = K + c*TEnergy(T)

    if k_max == :auto
        k_list,u = Quadrature(K ; kwds...)
        # Note: got to go extra, because the energy will change.
        k_max = maximum(k_list) * 4
        @debug "Automatically chose k_max" k_max
    end
    Δk_list = LinRange(0., k_max, 10001)
    pot_k = PrepareInterpedVkk(potfunc, Δk_list, rmin=rmin, rmax=rmax)

    if guess == :auto
        en = ESelfCon(0.)
    else
        en = guess
    end
    
    l = 0

    for i = 1:max_iters
        i == max_iters && error("Max iterations ($max_iters) reached!")

        k_list,T_mat = OffShellScattering(en, l, potfunc, pot_k=pot_k ; kwds...)
        maximum(k_list) > k_max && @warn "Energy creates a k_list with max > k_max" en k_max maximum(k_list)

        Tr = Spline1D(k_list, real.(diag(T_mat)))(k_target)
        Ti = Spline1D(k_list, imag.(diag(T_mat)))(k_target)
        T = Tr + im*Ti

        new_en = ESelfCon(T)
        @debug "Iteration" en new_en T K TEnergy(T)
        if zero_limit
            # Still need the imaginary part here in case en goes negative (for sqrt).
            new_en = real(new_en) + 0im
        end
        
        if abs(new_en - en) < tol
            break
        end
        en = (1-α)*en + α*new_en
    end

    return en
end


using ProgressMeter
function LaxScan(k_list, potfunc, c ; adjust_guess=true, kwds...)
    E_list = []
    guess = :auto
    @showprogress for k in k_list
        E = try
            SearchLax(k, potfunc, c ; guess=guess, kwds...)
        catch exc
            @error "Got exception for k=$k" k exc
            NaN
        end
        push!(E_list, E)
        if adjust_guess && !isnan(E)
            guess = E
        end
    end
    return E_list
end
    
