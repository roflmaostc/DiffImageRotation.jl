module DiffImageRotation

using KernelAbstractions, ChainRulesCore, Atomix

export imrotate

imrotate(arr::AbstractArray{T, 2}, θ; mid=size(arr) .÷ 2 .+ 1) where T = 
    view(imrotate(reshape(arr, (size(arr,1), size(arr, 2), 1)), θ; mid), :, :, 1)

"""
    imrotate(arr::AbstractArray, θ)

Rotates a matrix around the center pixel `size(arr) ÷ 2 .+ 1` which means there is a real center pixel 
The angle `θ` is interpreted in radians.

The adjoint is defined with ChainRulesCore. This method also runs with CUDA.
If `arr` is a `AbstractArray{T, 3}`, the third dimension is interpreted as batch dimension.


# Examples
```julia-repl
julia> arr = zeros((6, 6)); arr[3:4, 4] .= 1;

julia> arr
6×6 Matrix{Float64}:
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  1.0  0.0  0.0
 0.0  0.0  0.0  1.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0

julia> imrotate(arr, deg2rad(45))
6×6 view(::Array{Float64, 3}, :, :, 1) with eltype Float64:
 0.0  0.0  0.0        0.0        0.0       0.0
 0.0  0.0  0.0        0.0        0.0       0.0
 0.0  0.0  0.0        0.292893   0.585786  0.0
 0.0  0.0  0.0857864  1.0        0.292893  0.0
 0.0  0.0  0.0        0.0857864  0.0       0.0
 0.0  0.0  0.0        0.0        0.0       0.0

julia> imrotate(arr, deg2rad(90))
6×6 view(::Array{Float64, 3}, :, :, 1) with eltype Float64:
 0.0  0.0  0.0  0.0  0.0          0.0
 0.0  0.0  0.0  0.0  0.0          0.0
 0.0  0.0  0.0  0.0  1.11022e-16  0.0
 0.0  0.0  0.0  1.0  1.0          0.0
 0.0  0.0  0.0  0.0  0.0          0.0
 0.0  0.0  0.0  0.0  0.0          0.0

julia> using CUDA

julia> Array(imrotate(CuArray(arr), deg2rad(90))) ≈ imrotate(arr, deg2rad(90))
true

julia> using Zygote

julia> f(x) = sum(abs2.(imrotate(x, π/4)))
f (generic function with 1 method)

julia> Zygote.gradient(f, arr)
([0.0 0.0 … 0.0 0.0; 0.0 0.0 … 5.387705551013979e-17 0.0; … ; 0.0 0.0 … 0.08578643762690495 0.0; 0.0 0.0 … 0.0 0.0],)

```
"""
function imrotate(arr::AbstractArray{T, 3}, θ; mid=size(arr) .÷ 2 .+ 1) where T
    # needed for rotation matrix
    sinθ, cosθ = sincos(T(θ))

    # important variables
    mid = mid

    # out array
    out = similar(arr)
    fill!(out, 0)

    # KernelAbstractions specific
    backend = get_backend(arr)
    kernel! = imrotate_kernel!(backend)
    # launch kernel
    kernel!(out, arr, sinθ, cosθ, mid, size(arr, 1), size(arr, 2),
            ndrange=(size(arr, 1), size(arr, 2), size(arr, 3)))

	return out
end

# KernelAbstractions specific
@kernel function imrotate_kernel!(out, arr, sinθ, cosθ, mid, imax, jmax)
    i, j, k = @index(Global, NTuple)
    y = i - mid[1]
    x = j - mid[2]
    yrot = cosθ * y - sinθ * x
    xrot = sinθ * y + cosθ * x
    yrotf = floor(yrot)
    xrotf = floor(xrot)
    inew = floor(Int, yrot) + mid[1]
    jnew = floor(Int, xrot) + mid[2]
    
    if 1 ≤ inew ≤ imax && 1 ≤ jnew ≤ jmax 
        xdiff = (xrot - xrotf)
        xdiff_diff = 1 - xdiff
        ydiff = (yrot - yrotf)
        ydiff_diff = 1 - ydiff
        # in case we hit the boundary stripe, then we need to avoid out of bounds access
        Δi = inew != imax
        Δj = jnew != jmax

        @inbounds out[i, j, k] = 
             ( xdiff_diff * ydiff_diff * arr[inew,      jnew,      k]
             + xdiff_diff * ydiff      * arr[inew + Δi, jnew,      k]
             + xdiff      * ydiff_diff * arr[inew,      jnew + Δj, k] 
             + xdiff      * ydiff      * arr[inew + Δi, jnew + Δj, k])
    end
end


imrotate_adj(arr::AbstractArray{T, 2}, θ; mid=size(arr) .÷ 2 .+ 1) where T = 
    view(imrotate_adj(reshape(arr, (size(arr,1), size(arr, 2), 1)), θ; mid), :, :, 1)

function imrotate_adj(arr::AbstractArray{T, 3}, θ; mid=size(arr) .÷ 2 .+ 1) where T
    # needed for rotation matrix
    sinθ, cosθ = sincos(T(θ))
    
    # important variables
    mid = mid 
    # out array
    out = similar(arr)
    fill!(out, 0)
    
    # KernelAbstractions specific
    backend = get_backend(arr)
    kernel! = imrotate_kernel_adj!(backend)
    # launch kernel
    kernel!(out, arr, sinθ, cosθ, mid, size(arr, 1), size(arr, 2),
        ndrange=(size(arr, 1), size(arr, 2), size(arr, 3)))
    
    return out
end



# KernelAbstractions specific
@kernel function imrotate_kernel_adj!(out, arr, sinθ, cosθ, mid, imax, jmax)
    i, j, k = @index(Global, NTuple)
    y = i - mid[1]
    x = j - mid[2]
    yrot = cosθ * y - sinθ * x
    xrot = sinθ * y + cosθ * x
    yrotf = floor(yrot)
    xrotf = floor(xrot)
    inew = floor(Int, yrot) + mid[1]
    jnew = floor(Int, xrot) + mid[2]
    if 1 ≤ inew ≤ imax && 1 ≤ jnew ≤ jmax
        o = arr[i, j, k]
        xdiff = (xrot - xrotf)
        ydiff = (yrot - yrotf)
        # in case we hit the boundary stripe, then we need to avoid out of bounds access
        Δi = inew != imax
        Δj = jnew != jmax
        Atomix.@atomic out[inew,      jnew,      k] += (1 - xdiff) * (1 - ydiff) * o
        Atomix.@atomic out[inew + Δi, jnew,      k] += (1 - xdiff) * ydiff       * o
        Atomix.@atomic out[inew,      jnew + Δj, k] += xdiff       * (1 - ydiff) * o
        Atomix.@atomic out[inew + Δi, jnew + Δj, k] += xdiff       * ydiff       * o
    end
end

function ChainRulesCore.rrule(imrotate, array, θ)
    res = imrotate(array, θ)
    function pb_rotate(ȳ)
        f̄ = NoTangent()
        ad = imrotate_adj(ȳ, θ)
        return NoTangent(), ad, NoTangent()
    end    
	return res, pb_rotate
end

end # module DiffImageRotation
