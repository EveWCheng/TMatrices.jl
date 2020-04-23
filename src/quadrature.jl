
using FastGaussQuadrature
function ScaleGaussLegendre(N, a, b)
    x,w = gausslegendre(N)
    mid = (b+a)/2
    L = (b-a)/2

    x = L*x .+ mid
    w = L * w

    return x,w
end
function ScaleGaussLaguerre(N, a, scale=1.)
    x,w = gausslaguerre(N)
    w = scale * exp.(log.(w) .+ x)

    # This is for the generalised Gauss Laguerre - but it should really be
    # done in the original func... but I don't see how that can be done
    # cleanly in matrix form.
    # x,w = gausslaguerre(N,2.0)
    # w = @. scale * exp(log(w) + x - 2*log(x))


    x = x*scale .+ a

    return x,w
end



"""
    Quadrature(en_target, ...)
    
This is the workhorse of the code. It figures out how to approximate the integral in the Lippmann-Schwinger equations and evaluates the Greens function too.
"""
function Quadrature(en_target ; N=10,
                    N_inner=N,
                    N_mid=N,
                    N_outer=N,
                    sym_size=0.1, #sym_inner_frac=0.01)
                    sym_inner_frac=nothing,
                    outer_style=(:laguerre, 1.0),
                    mass = 1.,
                    ħ = 1.
                    )
    k_target = sqrt(2mass * en_target) / ħ

    if k_target < sym_size
        sym_size = k_target
    end

    Gfunc(x) = 1/(en_target - ħ^2*x^2/2mass)

    regions = []

    if sym_inner_frac == nothing
        isreal(en_target) && @assert iseven(N_mid)

        # a) create one region spanning (k_target - sym_size) : (k_target + sym_szie)
        x,w = ScaleGaussLegendre(N_mid, k_target - sym_size, k_target + sym_size)

        w .*= Gfunc.(x)
        push!(regions, (x,w))
    else
        # b) Create two regions from (k_target - sym_size) : (k_target - sym_inner_size) and
        # similar above. Then apply the special rule above.
        error("Not implemented")
        sym_inner_size = sym_size * sym_inner_frac
    end

    if isreal(en_target)
        # Got to have the divergence always
        push!(regions, ([k_target], [DivergenceQuad(k_target)]))
    end

    # Then create a Gauss-Laguerre region above and a Gauss-Legendre region below (if sym_size != k_target).
    if sym_size != k_target
        a = 0
        b = k_target - sym_size
        x,w = ScaleGaussLegendre(N_inner, a, b)
        w .*= Gfunc.(x)
        lower = (x,w)

        pushfirst!(regions, lower)
    end

    if outer_style[1] == :laguerre
        laguerre_scale = outer_style[2]
        x,w = ScaleGaussLaguerre(N_outer, k_target + sym_size, laguerre_scale)
        w .*= Gfunc.(x)
        outer = (x,w)
        push!(regions, outer)
    elseif outer_style[1] == :legendre
        legendre_max = outer_style[2]
        x,w = ScaleGaussLegendre(N_outer, k_target + sym_size, k_target + sym_size + legendre_max)
        w .*= Gfunc.(x)
        outer = (x,w)
        push!(regions, outer)
    else
        error("Unknown outer_style $(outer_style[1])")
    end
    

    # x = vcat(first.(regions)...)
    # u = vcat(last.(regions)...)
    x,u = reduce(regions) do r1,r2
        x1,u1 = r1
        x2,u2 = r2
        x = union(x1,x2)
        u = zeros(ComplexF64, size(x))

        u[indexin(x1,x)] .+= u1
        u[indexin(x2,x)] .+= u2

        (x,u)
    end

    inds = sortperm(x)
    x = x[inds]
    u = u[inds]
    return x,u,regions
end

function DivergenceQuad(qi)
    mass = ħ = 1.
    u = im*pi * 1/(ħ^2/mass * qi)
end


"""
    TaylorApprox(k)
    
This evaluates the divergent part of the integral, using a Taylor series expansion around that point.
"""
function TaylorApprox(k3)
    @assert length(k3) == 3
    @assert diff(k3) |> x -> (x[1] == x[2])
    qi = k3[2]
    dk = diff(k3)[1]

    @info "Including approximation for the divergent part"
    mass = ħ = 1.
    prefac = 2*mass/ħ^2 * dk/qi
    # f = k_list'.^2 .* Vmat

    u = zeros(3)
    u[:,3] = -prefac/2dk
    u[:,1] = prefac/2dk
    u[:,2] = prefac/2qi

    return u
end



# This was for a test calculating the K matrix instead of the T matrix.
function KMatQuadrature(en_target ; N=10,
                        N_inner=N,
                        N_mid=N,
                        N_outer=N,
                        sym_size=0.1, #sym_inner_frac=0.01)
                        sym_inner_frac=nothing,
                        laguerre_scale=1.0)
    if isreal(en_target)
        # Need to make sure there is a symmetric region in here.

        mass = ħ = 1.
        k_target = sqrt(2mass * en_target) / ħ

        if k_target < sym_size
            sym_size = k_target
        end

        function Gfunc(x)
            denom = en_target - ħ^2*x/2mass
            denom == 0. ? 0. : 1/denom
        end

        regions = []

        # Allow for two options
        if sym_inner_frac == nothing
            @assert isodd(N_mid)
            # a) create one region spanning (k_target - sym_size) : (k_target + sym_szie)
            x,w = ScaleGaussLegendre(N_mid, k_target - sym_size, k_target + sym_size)

            w .*= Gfunc.(x)
            push!(regions, (x,w))
        else
            # b) Create two regions from (k_target - sym_size) : (k_target - sym_inner_size) and
            # similar above. Then apply the special rule above.
            error("Not implemented")
            sym_inner_size = sym_size * sym_inner_frac
        end

        # Then create a Gauss-Laguerre region above and a Gauss-Legendre region below (if sym_size != k_target).
        if sym_size != k_target
            a = 0
            b = k_target - sym_size
            x,w = ScaleGaussLegendre(N_inner, a, b)
            w .*= Gfunc.(x)
            lower = (x,w)

            pushfirst!(regions, lower)
        end

        x,w = ScaleGaussLaguerre(N_outer, k_target + sym_size, laguerre_scale)
        w .*= Gfunc.(x)
        outer = (x,w)
        push!(regions, outer)

        # x = vcat(first.(regions)...)
        # u = vcat(last.(regions)...)
        x,u = reduce(regions) do r1,r2
            x1,u1 = r1
            x2,u2 = r2
            x = union(x1,x2)
            u = zeros(size(x))

            u[indexin(x1,x)] .+= u1
            u[indexin(x2,x)] .+= u2

            (x,u)
        end

        inds = sortperm(x)
        x = x[inds]
        u = u[inds]
        return x,u,regions
    else
        error("Not implemented")

        # Maybe do a GaussLegendre region around the real value of k_target, but
        # down to zero. Then throw in a Laguerre region for the outer.
        
    end
end
