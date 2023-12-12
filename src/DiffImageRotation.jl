module DiffImageRotation

using KernelAbstractions, ChainRulesCore, Atomix

export imrotate

imrotate(arr::AbstractArray{T, 2}, θ) where T = 
    view(imrotate(reshape(arr, (size(arr,1), size(arr, 2), 1)), θ), :, :, 1)

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
function imrotate(arr::AbstractArray{T, 3}, θ) where T
    @assert size(arr, 1) == size(arr, 2) "only quadratic arrays in dimension 1 and 2"
    # needed for rotation matrix
    sinθ, cosθ = sincos(T(θ))

    # important variables
    mid = size(arr, 1) .÷ 2 + 1

    # out array
    out = similar(arr)
    fill!(out, 0)

    # KernelAbstractions specific
    backend = get_backend(arr)
    kernel! = imrotate_kernel!(backend)
    # launch kernel
    kernel!(out, arr, sinθ, cosθ, mid,
            ndrange=(size(arr, 1), size(arr, 2), size(arr, 3)))

	return out
end

# KernelAbstractions specific
@kernel function imrotate_kernel!(out, arr, sinθ, cosθ, mid)
    i, j, k = @index(Global, NTuple)
    y = i - mid
    x = j - mid
    yrot = cosθ * y - sinθ * x
    xrot = sinθ * y + cosθ * x
    yrotf = floor(yrot)
    xrotf = floor(xrot)
    inew = floor(Int, yrot) + mid
    jnew = floor(Int, xrot) + mid
    
    if 1 ≤ inew < size(out, 1) && 1 ≤ jnew < size(out, 2)
        xdiff = (xrot - xrotf)
        ydiff = (yrot - yrotf)
        @inbounds out[i, j, k] = 
            ((1 - xdiff) * (1 - ydiff) * arr[inew, jnew, k]
            + (1 - xdiff) * ydiff * arr[inew + 1, jnew, k]
            + xdiff * (1 - ydiff) * arr[inew, jnew + 1, k] 
            + xdiff * ydiff * arr[inew + 1, jnew + 1, k])
    end
end


imrotate_adj(arr::AbstractArray{T, 2}, θ) where T = 
    view(imrotate_adj(reshape(arr, (size(arr,1), size(arr, 2), 1)), θ), :, :, 1)

function imrotate_adj(arr::AbstractArray{T, 3}, θ) where T
    @assert size(arr, 1) == size(arr, 2) "only quadratic arrays in dimension 1 and 2"
    # needed for rotation matrix
    sinθ, cosθ = sincos(T(θ))
    
    # important variables
    mid = size(arr, 1) .÷ 2 + 1
    # out array
    out = similar(arr)
    fill!(out, 0)
    
    # KernelAbstractions specific
    backend = get_backend(arr)
    kernel! = imrotate_kernel_adj!(backend)
    # launch kernel
    kernel!(out, arr, sinθ, cosθ, mid,
        ndrange=(size(arr, 1), size(arr, 2), size(arr, 3)))
    
    return out
end



# KernelAbstractions specific
@kernel function imrotate_kernel_adj!(out, arr, sinθ, cosθ, mid)
    i, j, k = @index(Global, NTuple)
    y = i - mid
    x = j - mid
    yrot = cosθ * y - sinθ * x
    xrot = sinθ * y + cosθ * x
    yrotf = floor(yrot)
    xrotf = floor(xrot)
    inew = floor(Int, yrot) + mid
    jnew = floor(Int, xrot) + mid
    if 1 ≤ inew < size(out, 1) && 1 ≤ jnew < size(out, 2)
        o = arr[i, j, k]
        xdiff = (xrot - xrotf)
        ydiff = (yrot - yrotf)
        Atomix.@atomic out[inew, jnew, k] += (1 - xdiff) * (1 - ydiff) * o
        Atomix.@atomic out[inew + 1, jnew, k] += (1 - xdiff) * ydiff * o
        Atomix.@atomic out[inew, jnew + 1, k] += xdiff * (1 - ydiff) * o
        Atomix.@atomic out[inew + 1, jnew + 1, k] += xdiff * ydiff * o
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
