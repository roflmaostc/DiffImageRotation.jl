module DiffImageRotation

using KernelAbstractions, ChainRulesCore, Atomix

export imrotate, imrotate!

# this rotates the coordinates and either applies round(nearest neighbour)
# or floor (:bilinear interpolation)
@inline function rotate_coordinates(sinθ, cosθ, i, j, midpoint, round_or_floor)
    y = i - midpoint[1]
    x = j - midpoint[2]
    yrot = cosθ * y - sinθ * x + midpoint[1]
    xrot = sinθ * y + cosθ * x + midpoint[2]
    yrot_f = round_or_floor(yrot)
    xrot_f = round_or_floor(xrot)
    yrot_int = round_or_floor(Int, yrot)
    xrot_int = round_or_floor(Int, xrot)
    return yrot, xrot, yrot_f, xrot_f, yrot_int, xrot_int
end

# helper function for bilinear
@inline function bilinear_helper(yrot, xrot, yrot_f, xrot_f, yrot_int, xrot_int, imax, jmax)
        xdiff = (xrot - xrot_f)
        xdiff_diff = 1 - xdiff
        ydiff = (yrot - yrot_f)
        ydiff_diff = 1 - ydiff
        # in case we hit the boundary stripe, then we need to avoid out of bounds access
        Δi = 1#yrot_int != imax
        Δj = 1#xrot_int != jmax
      
        # we need to avoid that we access arr[0]
        # in rare cases the rounding clips off values on the left and top border
        # we still try to access them with this extra comparison
        Δi_min = 0#yrot_int == 0
        Δj_min = 0#xrot_int == 0
        return Δi, Δj, Δi_min, Δj_min, ydiff, ydiff_diff, xdiff, xdiff_diff
end


imrotate(arr::AbstractArray{T, 2}, θ; midpoint=size(arr) .÷ 2 .+ 1, method=:bilinear, adjoint=false, fillvalue=zero(T)) where T = 
    reshape(imrotate(reshape(arr, (size(arr,1), size(arr, 2), 1)), θ; midpoint, method, adjoint, fillvalue), (size(arr, 1), size(arr, 2)))

"""
    imrotate(arr::AbstractArray, θ; method=:bilinear, midpoint=size(arr) .÷ 2 .+ 1)

Rotates a matrix around the center pixel `midpoint`.
The angle `θ` is interpreted in radians.

The adjoint is defined with ChainRulesCore.jl. This method also runs with CUDA (and in principle all KernelAbstractions.jl supported backends).
If `arr` is a `AbstractArray{T, 3}`, the third dimension is interpreted as batch dimension.

# Keywords
* `method=:bilinear` for bilinear interpolation or `method=:nearest` for nearest neighbour
* `midpoint=size(arr) .÷ 2 .+ 1` means there is always a real center pixel around it is rotated.

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

julia> imrotate(arr, deg2rad(90), midpoint=(3,3))
6×6 view(::Array{Float64, 3}, :, :, 1) with eltype Float64:
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  1.0  1.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0

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
function imrotate(arr::AbstractArray{T, 3}, θ; method=:bilinear, midpoint=size(arr) .÷ 2 .+ 1,
                  adjoint=false, fillvalue=zero(T)) where T
    @assert (T <: Integer && method==:nearest || !(T <: Integer)) "If the array has an Int eltype, only method=:nearest is supported"
    @assert typeof(midpoint) <: Tuple "midpoint keyword has to be a tuple"
    # out array
    out = similar(arr)
    fill!(out, fillvalue)
    imrotate!(out, arr, θ; method, midpoint, adjoint)
    return out
end


function imrotate!(out::AbstractArray{T, 2}, arr::AbstractArray{T, 2}, θ; method=:bilinear, midpoint=size(arr) .÷ 2 .+ 1,
                  adjoint=false) where T
    imrotate!(reshape(out, (size(out, 1), size(out, 2), 1)), 
              reshape(arr, (size(arr, 1), size(arr, 2), 1)), 
              θ; method, midpoint, adjoint)
    return out 
end
"""
    imrotate!(out, arr::AbstractArray, θ; method=:bilinear, midpoint=size(arr) .÷ 2 .+ 1)

In-place version of [`imrotate`](@ref).

!!! warning 
    The values in `out` are not overwritten but added to the values of the rotation operation!
    So `out .+ arr == imrotate!(out, arr, θ)`
"""
function imrotate!(out::AbstractArray{T, 3}, arr::AbstractArray{T, 3}, θ; method=:bilinear, midpoint=size(arr) .÷ 2 .+ 1,
                  adjoint=false) where T
    # needed for rotation matrix
    θ = mod(real(T)(θ), real(T)(2π))

    if iszero(θ)
        out .+= arr
        return out
    end

    # check for special cases where rotations are trivial
    if midpoint[1] ≈ size(arr, 1) ÷ 2 + 0.5 && midpoint[2] ≈ size(arr, 2) ÷ 2 + 0.5
        if θ ≈ π / 2 
            out .+= reverse(PermutedDimsArray(arr, (2,1,3)), dims=(2,))
            return out
        elseif θ ≈ π
            out .+= reverse(arr, dims=(1,2))
            return out
        elseif θ ≈ 3 / 2 * π
            out .+= reverse(PermutedDimsArray(arr, (2,1,3)), dims=(1,))
            return out
        end
    end
    midpoint = real(T).(midpoint)
    sinθ, cosθ = sincos(real(T)(θ)) 
    # KernelAbstractions specific
    backend = get_backend(arr)
    if adjoint
        if method == :bilinear
            kernel! = imrotate_kernel_adj!(backend)
        elseif method == :nearest
            kernel! = imrotate_kernel_nearest_adj!(backend)
        else 
            throw(ArgumentError("No interpolation method such as $method"))
        end
    else
        if method == :bilinear
            kernel! = imrotate_kernel!(backend)
        elseif method == :nearest
            kernel! = imrotate_kernel_nearest!(backend)
        else 
            throw(ArgumentError("No interpolation method such as $method"))
        end
    end
    kernel!(out, arr, sinθ, cosθ, midpoint, size(arr, 1), size(arr, 2),
            ndrange=(size(arr, 1), size(arr, 2), size(arr, 3)))
	return out
end

@kernel function imrotate_kernel_nearest!(out, arr, sinθ, cosθ, midpoint, imax, jmax)
    i, j, k = @index(Global, NTuple)

    _, _, _, _, yrot_int, xrot_int = rotate_coordinates(sinθ, cosθ, i, j, midpoint, round) 
    if 1 ≤ yrot_int ≤ imax && 1 ≤ xrot_int ≤ jmax
        @inbounds out[i, j, k] += arr[yrot_int, xrot_int, k]
    end
end


@kernel function imrotate_kernel!(out, arr, sinθ, cosθ, midpoint, imax, jmax)
    i, j, k = @index(Global, NTuple)
    
    yrot, xrot, yrot_f, xrot_f, yrot_int, xrot_int = rotate_coordinates(sinθ, cosθ, i, j, midpoint, floor) 
    if 1 ≤ yrot_int ≤ imax - 1&& 1 ≤ xrot_int ≤ jmax - 1 

        Δi, Δj, Δi_min, Δj_min, ydiff, ydiff_diff, xdiff, xdiff_diff = 
            bilinear_helper(yrot, xrot, yrot_f, xrot_f, yrot_int, xrot_int, imax, jmax)
        @inbounds out[i, j, k] += 
            (   xdiff_diff  * ydiff_diff    * arr[yrot_int + Δi_min, xrot_int + Δj_min, k]
             +  xdiff_diff  * ydiff         * arr[yrot_int + Δi,     xrot_int + Δj_min, k]
             +  xdiff       * ydiff_diff    * arr[yrot_int + Δi_min, xrot_int + Δj,     k] 
             +  xdiff       * ydiff         * arr[yrot_int + Δi,     xrot_int + Δj,     k])
    end
end



@kernel function imrotate_kernel_nearest_adj!(out, arr, sinθ, cosθ, midpoint, imax, jmax)
    i, j, k = @index(Global, NTuple)

    _, _, _, _, yrot_int, xrot_int = rotate_coordinates(sinθ, cosθ, i, j, midpoint, round) 
    if 1 ≤ yrot_int ≤ imax && 1 ≤ xrot_int ≤ jmax 
        Atomix.@atomic out[yrot_int, xrot_int, k] += arr[i, j, k]
    end
end


@kernel function imrotate_kernel_adj!(out, arr, sinθ, cosθ, midpoint, imax, jmax)
    i, j, k = @index(Global, NTuple)

    yrot, xrot, yrot_f, xrot_f, yrot_int, xrot_int = rotate_coordinates(sinθ, cosθ, i, j, midpoint, floor) 
    if 1 ≤ yrot_int ≤ imax - 1 && 1 ≤ xrot_int ≤ jmax - 1
        o = arr[i, j, k]
        Δi, Δj, Δi_min, Δj_min, ydiff, ydiff_diff, xdiff, xdiff_diff = 
            bilinear_helper(yrot, xrot, yrot_f, xrot_f, yrot_int, xrot_int, imax, jmax)
        Atomix.@atomic out[yrot_int + Δi_min,   xrot_int + Δj_min,  k]  += (1 - xdiff)  * (1 - ydiff) * o
        Atomix.@atomic out[yrot_int + Δi,       xrot_int + Δj_min,  k]  += (1 - xdiff)  * ydiff       * o
        Atomix.@atomic out[yrot_int + Δi_min,   xrot_int + Δj,      k]  += xdiff        * (1 - ydiff) * o
        Atomix.@atomic out[yrot_int + Δi,       xrot_int + Δj,      k]  += xdiff        * ydiff       * o
    end
end



# is this rrule good? 
# no @thunk and @unthunk
function ChainRulesCore.rrule(::typeof(imrotate), array::AbstractArray{T}, θ; method=:bilinear, midpoint=size(array) .÷ 2 .+ 1,
                              fillvalue=zero(T)) where T
    res = imrotate(array, θ; method, midpoint, fillvalue)
    function pb_rotate(ȳ)
        f̄ = NoTangent()
        ad = imrotate(ȳ, θ; method, midpoint, adjoint=true, fillvalue)
        return NoTangent(), ad, NoTangent()
    end    
	return res, pb_rotate
end

end # module DiffImageRotation
