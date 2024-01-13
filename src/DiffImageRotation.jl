module DiffImageRotation

using KernelAbstractions, ChainRulesCore, Atomix
export imrotate, imrotate!


include("utils.jl")


imrotate(arr::AbstractArray{T, 2}, θ; midpoint=size(arr) .÷ 2 .+ 1, method=:bilinear, fillvalue=zero(T)) where T =
    reshape(imrotate(reshape(arr, (size(arr,1), size(arr, 2), 1)), θ; midpoint, method, fillvalue), (size(arr, 1), size(arr, 2)))


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
                  fillvalue=zero(T)) where T
    @assert (T <: Integer && method==:nearest || !(T <: Integer)) "If the array has an Int eltype, only method=:nearest is supported"
    @assert typeof(midpoint) <: Tuple "midpoint keyword has to be a tuple"
    # out array
    out = similar(arr)
    fill!(out, fillvalue)
    imrotate!(out, arr, θ; method, midpoint)
    return out
end


function imrotate!(out::AbstractArray{T, 2}, arr::AbstractArray{T, 2}, θ; method=:bilinear, midpoint=size(arr) .÷ 2 .+ 1) where T
    imrotate!(reshape(out, (size(out, 1), size(out, 2), 1)),
              reshape(arr, (size(arr, 1), size(arr, 2), 1)),
              θ; method, midpoint)
    return out
end


"""
    imrotate!(out, arr::AbstractArray, θ; method=:bilinear, midpoint=size(arr) .÷ 2 .+ 1)

In-place version of [`imrotate`](@ref).

!!! warning
    The values in `out` are not overwritten but added to the values of the rotation operation!
    So `out .+ arr == imrotate!(out, arr, θ)`
"""
function imrotate!(out::AbstractArray{T, 3}, arr::AbstractArray{T, 3}, θ; method=:bilinear, midpoint=size(arr) .÷ 2 .+ 1) where T
    @assert (T <: Integer && method==:nearest || !(T <: Integer)) "If the array has an Int eltype, only method=:nearest is supported"
    @assert typeof(midpoint) <: Tuple "midpoint keyword has to be a tuple"
    
    # prepare out, the sin and cos and type of midpoint
    sinθ, cosθ, midpoint = _prepare_imrotate(arr, θ, midpoint) 
    # such as 0°, 90°, 180°, 270° and only if the midpoint is suitable
    _check_trivial_rotations!(out, arr, θ, midpoint) && return out

    # KernelAbstractions specific
    backend = KernelAbstractions.get_backend(arr)
    if method == :bilinear
        kernel! = imrotate_kernel_bilinear!(backend)
    elseif method == :nearest
        kernel! = imrotate_kernel_nearest!(backend)
    else 
        throw(ArgumentError("No interpolation method such as $method"))
    end
    kernel!(out, arr, sinθ, cosθ, midpoint, size(arr, 1), size(arr, 2),
            ndrange=(size(arr, 1), size(arr, 2), size(arr, 3)))
	return out
end




∇imrotate(dy, arr::AbstractArray{T, 2}, θ; midpoint=size(arr) .÷ 2 .+ 1, method=:bilinear, fillvalue=zero(T)) where T =
    reshape(∇imrotate(dy, reshape(arr, (size(arr,1), size(arr, 2), 1)), θ; midpoint, method, fillvalue), (size(arr, 1), size(arr, 2)))

"""
    ∇imrotate(dy, arr::AbstractArray{T, 3}, θ; method=:bilinear,
                                               midpoint=size(arr) .÷ 2 .+ 1)

Adjoint for `imrotate`. Gradient only with respect to `arr` and not `θ`.

# Arguments
* `dy`: input gradient 
* `arr`: Input from primal computation
* `θ`: rotation angle in radians
* `method=:bilinear` or `method=:nearest`
* `midpoint=size(arr) .÷ 2 .+ 1` rotates around a real center pixel for even and odd sized arrays
"""
function ∇imrotate(dy, arr::AbstractArray{T, 3}, θ; method=:bilinear, 
                   midpoint=size(arr) .÷ 2 .+ 1, fillvalue=zero(T)) where T
    
    out = similar(arr)
    fill!(out, fillvalue)
    
    sinθ, cosθ, midpoint = _prepare_imrotate(arr, θ, midpoint) 
    # for the adjoint, the trivial rotations go in the other direction!
    # pass dy and not arr
    _check_trivial_rotations!(out, dy, θ, midpoint, adjoint=true) && return out

    backend = KernelAbstractions.get_backend(arr)
    if method == :bilinear
        kernel! = ∇imrotate_kernel_bilinear!(backend)
    elseif method == :nearest
        kernel! = ∇imrotate_kernel_nearest!(backend)
    else 
        throw(ArgumentError("No interpolation method such as $method"))
    end
    # don't pass arr but dy! 
    kernel!(out, dy, sinθ, cosθ, midpoint, size(arr, 1), size(arr, 2),
            ndrange=(size(arr, 1), size(arr, 2), size(arr, 3)))
    return out
end


@kernel function imrotate_kernel_nearest!(out, arr, sinθ, cosθ, midpoint, imax, jmax)
    i, j, c = @index(Global, NTuple)

    _, _, _, _, yrot_int, xrot_int = rotate_coordinates(sinθ, cosθ, i, j, midpoint, round) 
    if 1 ≤ yrot_int ≤ imax && 1 ≤ xrot_int ≤ jmax
        @inbounds out[i, j, c] = arr[yrot_int, xrot_int, c]
    end
end


@kernel function imrotate_kernel_bilinear!(out, arr, sinθ, cosθ, midpoint, imax, jmax)
    i, j, c = @index(Global, NTuple)
    
    yrot, xrot, yrot_f, xrot_f, yrot_int, xrot_int = rotate_coordinates(sinθ, cosθ, i, j, midpoint, floor) 
    if 1 ≤ yrot_int ≤ imax - 1 && 1 ≤ xrot_int ≤ jmax - 1 

        ydiff, ydiff_1minus, xdiff, xdiff_1minus = 
            bilinear_helper(yrot, xrot, yrot_f, xrot_f)
        @inbounds out[i, j, c] = 
            (   xdiff_1minus    * ydiff_1minus  * arr[yrot_int      , xrot_int      , c]
             +  xdiff_1minus    * ydiff         * arr[yrot_int + 1  , xrot_int      , c]
             +  xdiff           * ydiff_1minus  * arr[yrot_int      , xrot_int + 1  , c] 
             +  xdiff           * ydiff         * arr[yrot_int + 1  , xrot_int + 1  , c])
    end
end


@kernel function ∇imrotate_kernel_nearest!(out, arr, sinθ, cosθ, midpoint, imax, jmax)
    i, j, c = @index(Global, NTuple)

    _, _, _, _, yrot_int, xrot_int = rotate_coordinates(sinθ, cosθ, i, j, midpoint, round) 
    if 1 ≤ yrot_int ≤ imax && 1 ≤ xrot_int ≤ jmax 
        Atomix.@atomic out[yrot_int, xrot_int, c] += arr[i, j, c]
    end
end


@kernel function ∇imrotate_kernel_bilinear!(out, arr, sinθ, cosθ, midpoint, imax, jmax)
    i, j, c = @index(Global, NTuple)

    yrot, xrot, yrot_f, xrot_f, yrot_int, xrot_int = rotate_coordinates(sinθ, cosθ, i, j, midpoint, floor) 
    if 1 ≤ yrot_int ≤ imax - 1 && 1 ≤ xrot_int ≤ jmax - 1
        o = arr[i, j, c]
        ydiff, ydiff_1minus, xdiff, xdiff_1minus = 
            bilinear_helper(yrot, xrot, yrot_f, xrot_f)
        Atomix.@atomic out[yrot_int     ,   xrot_int    , c]  += xdiff_1minus    * ydiff_1minus * o
        Atomix.@atomic out[yrot_int + 1 ,   xrot_int    , c]  += xdiff_1minus    * ydiff      * o
        Atomix.@atomic out[yrot_int     ,   xrot_int + 1, c]  += xdiff           * ydiff_1minus * o
        Atomix.@atomic out[yrot_int + 1 ,   xrot_int + 1, c]  += xdiff           * ydiff      * o
    end
end


# is this rrule good? 
# no @thunk and @unthunk
function ChainRulesCore.rrule(::typeof(imrotate), arr::AbstractArray{T}, θ; 
                              method=:bilinear, midpoint=size(arr) .÷ 2 .+ 1) where T
    res = imrotate(arr, θ; method, midpoint)
    function pb_rotate(dy)
        ad = ∇imrotate(unthunk(dy), arr, θ; method, midpoint)
        return NoTangent(), ad, NoTangent()
    end    

	return res, pb_rotate
end


end # module
