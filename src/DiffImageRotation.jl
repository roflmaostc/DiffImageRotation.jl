module DiffImageRotation

using KernelAbstractions, ChainRulesCore, Atomix

export imrotate

# handle types correctly
# and also rounding sometimes does weird things
# so we reduce θ ∈ [0, π)
function get_sin_cos(::Type{T}, θ) where T
    K = promote_type(Float32, T)
    θ2 = K(θ)
    θ2 = mod(θ2, typeof(θ2)(2π))
    sinθ, cosθ = sincos(θ2)

    if θ2 ≥ π
        sinθ, cosθ = -sin(θ2 -K(π)), cos(K(2π) - θ2)
    end

    return sinθ, cosθ
end


imrotate(arr::AbstractArray{T, 2}, θ; mid=size(arr) .÷ 2 .+ 1, method=:bilinear) where T = 
    view(imrotate(reshape(arr, (size(arr,1), size(arr, 2), 1)), θ; mid, method), :, :, 1)

"""
    imrotate(arr::AbstractArray, θ; method=:bilinear)

Rotates a matrix around the center pixel `size(arr) ÷ 2 .+ 1` which means there is a real center pixel 
The angle `θ` is interpreted in radians.

The adjoint is defined with ChainRulesCore. This method also runs with CUDA.
If `arr` is a `AbstractArray{T, 3}`, the third dimension is interpreted as batch dimension.

# Keywords
* `method=:bilinear` for bilinear interpolation or `method=:nearest` for nearest neighbour

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

julia> imrotate(arr, deg2rad(90), method=:nearest)
6×6 view(::Array{Float64, 3}, :, :, 1) with eltype Float64:
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  1.0  1.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0

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
function imrotate(arr::AbstractArray{T, 3}, θ; method=:bilinear, mid=size(arr) .÷ 2 .+ 1) where T
    @assert (T <: Integer && method==:nearest || T <: AbstractFloat) "If the array has an Int eltype, only method=:nearest is supported"
    # out array
    out = similar(arr)
    fill!(out, 0)
    # needed for rotation matrix
    
    #sinθ, cosθ = get_sin_cos(T, θ)
    sinθ, cosθ = sincos(T(θ)) 
    # KernelAbstractions specific
    backend = get_backend(arr)
    if method == :bilinear
        kernel! = imrotate_kernel!(backend)
        # launch kernel
        kernel!(out, arr, sinθ, cosθ, mid, size(arr, 1), size(arr, 2),
                ndrange=(size(arr, 1), size(arr, 2), size(arr, 3)))
    elseif method == :nearest
        kernel! = imrotate_kernel_nearest!(backend)
        # launch kernel
        kernel!(out, arr, sinθ, cosθ, mid, size(arr, 1), size(arr, 2),
                ndrange=(size(arr, 1), size(arr, 2), size(arr, 3)))
    else 
        throw(ArgumentError("No interpolation method such as $method"))
    end
	return out
end

@kernel function imrotate_kernel_nearest!(out, arr, sinθ, cosθ, mid, imax, jmax)
    i, j, k = @index(Global, NTuple)
    y = i - mid[1]
    x = j - mid[2]
    yrot = cosθ * y - sinθ * x + mid[1]
    xrot = sinθ * y + cosθ * x + mid[2]
    inew = round(Int, yrot)
    jnew = round(Int, xrot)

    if 1 ≤ inew ≤ imax && 1 ≤ jnew ≤ jmax 
        @inbounds out[i, j, k] = arr[inew, jnew, k]
    end
end


# KernelAbstractions specific
@kernel function imrotate_kernel!(out, arr, sinθ, cosθ, mid, imax, jmax)
    i, j, k = @index(Global, NTuple)
    y = i - mid[1]
    x = j - mid[2]
    yrot = cosθ * y - sinθ * x + mid[1]
    xrot = sinθ * y + cosθ * x + mid[2]
    yrotf = floor(yrot)
    xrotf = floor(xrot)
    inew = floor(Int, yrot)
    jnew = floor(Int, xrot)
    
    if 0 ≤ inew ≤ imax && 0 ≤ jnew ≤ jmax 
        xdiff = (xrot - xrotf)
        xdiff_diff = 1 - xdiff
        ydiff = (yrot - yrotf)
        ydiff_diff = 1 - ydiff
        # in case we hit the boundary stripe, then we need to avoid out of bounds access
        Δi = inew != imax
        Δj = jnew != jmax
      
        # we need to avoid that we access arr[0]
        # in rare cases the rounding clips off values on the left and top border
        # we still try to access them with this extra comparison
        Δi_min = inew == 0
        Δj_min = jnew == 0

        @inbounds out[i, j, k] = 
             ( xdiff_diff * ydiff_diff * arr[inew + Δi_min,      jnew + Δj_min,      k]
             + xdiff_diff * ydiff      * arr[inew + Δi, jnew + Δj_min,      k]
             + xdiff      * ydiff_diff * arr[inew + Δi_min,      jnew + Δj, k] 
             + xdiff      * ydiff      * arr[inew + Δi, jnew + Δj, k])
    end
end


imrotate_adj(arr::AbstractArray{T, 2}, θ; method=:bilinear,  mid=size(arr) .÷ 2 .+ 1) where T = 
    view(imrotate_adj(reshape(arr, (size(arr,1), size(arr, 2), 1)), θ; method, mid), :, :, 1)

function imrotate_adj(arr::AbstractArray{T, 3}, θ; method=:bilinear, mid=size(arr) .÷ 2 .+ 1) where T
    # needed for rotation matrix
    sinθ, cosθ = get_sin_cos(T, θ)
    
    # out array
    out = similar(arr)
    fill!(out, 0)
    
    # KernelAbstractions specific
    backend = get_backend(arr)

    if method == :bilinear
        kernel! = imrotate_kernel_adj!(backend)
        # launch kernel
        kernel!(out, arr, sinθ, cosθ, mid, size(arr, 1), size(arr, 2),
                ndrange=(size(arr, 1), size(arr, 2), size(arr, 3)))
    elseif method == :nearest
        kernel! = imrotate_kernel_nearest_adj!(backend)
        kernel!(out, arr, sinθ, cosθ, mid, size(arr, 1), size(arr, 2),
                ndrange=(size(arr, 1), size(arr, 2), size(arr, 3)))
    else 
        throw(ArgumentError("No interpolation method such as $method"))
    end
	return out
end



# KernelAbstractions specific
@kernel function imrotate_kernel_adj!(out, arr, sinθ, cosθ, mid, imax, jmax)
    i, j, k = @index(Global, NTuple)
    y = i - mid[1]
    x = j - mid[2]
    yrot = cosθ * y - sinθ * x + mid[1] 
    xrot = sinθ * y + cosθ * x + mid[2]
    yrotf = floor(yrot)
    xrotf = floor(xrot)
    inew = floor(Int, yrot)
    jnew = floor(Int, xrot)
    if 0 ≤ inew ≤ imax && 0 ≤ jnew ≤ jmax
        o = arr[i, j, k]
        xdiff = (xrot - xrotf)
        ydiff = (yrot - yrotf)
        # in case we hit the boundary stripe, then we need to avoid out of bounds access
        Δi = inew != imax
        Δj = jnew != jmax
        # we need to avoid that we access arr[0]
        # in rare cases the rounding clips off values on the left and top border
        # we still try to access them with this extra comparison
        Δi_min = inew == 0
        Δj_min = jnew == 0
        Atomix.@atomic out[inew + Δi_min,   jnew + Δj_min, k]   += (1 - xdiff) * (1 - ydiff) * o
        Atomix.@atomic out[inew + Δi,       jnew + Δj_min, k]   += (1 - xdiff) * ydiff       * o
        Atomix.@atomic out[inew + Δi_min,   jnew + Δj, k]       += xdiff       * (1 - ydiff) * o
        Atomix.@atomic out[inew + Δi,       jnew + Δj, k]       += xdiff       * ydiff       * o
    end
end


@kernel function imrotate_kernel_nearest_adj!(out, arr, sinθ, cosθ, mid, imax, jmax)
    i, j, k = @index(Global, NTuple)
    y = i - mid[1]
    x = j - mid[2]
    yrot = cosθ * y - sinθ * x + mid[1]
    xrot = sinθ * y + cosθ * x + mid[2]
    inew = round(Int, yrot)
    jnew = round(Int, xrot)

    if 1 ≤ inew ≤ imax && 1 ≤ jnew ≤ jmax 
        Atomix.@atomic out[inew, jnew, k] += arr[i, j, k]
    end
end




# is this rrule good? 
# no @thunk and @unthunk
function ChainRulesCore.rrule(::typeof(imrotate), array, θ; method=:bilinear, mid=size(array) .÷ 2 .+ 1)
    res = imrotate(array, θ; method, mid)
    function pb_rotate(ȳ)
        f̄ = NoTangent()
        ad = imrotate_adj(ȳ, θ; method, mid)
        return NoTangent(), ad, NoTangent()
    end    
	return res, pb_rotate
end

end # module DiffImageRotation
