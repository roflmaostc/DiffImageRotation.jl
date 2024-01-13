"""
    rotate_coordinates(sinθ, cosθ, i, j, midpoint, round_or_floor)

this rotates the coordinates and either applies round(nearest neighbour)
or floor for :bilinear interpolation)
"""
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


"""
   bilinear_helper(yrot, xrot, yrot_f, xrot_f, yrot_int, xrot_int) 

Some helper variables
"""
@inline function bilinear_helper(yrot, xrot, yrot_f, xrot_f)
    xdiff = (xrot - xrot_f)
    xdiff_1minus = 1 - xdiff
    ydiff = (yrot - yrot_f)
    ydiff_1minus = 1 - ydiff
    
    return ydiff, ydiff_1minus, xdiff, xdiff_1minus
end


"""
    _prepare_imrotate(arr, θ, midpoint)

Prepate `sin` and `cos`, creates the output array and converts type
of `midpoint` if required.
"""
function _prepare_imrotate(arr::AbstractArray{T}, θ, midpoint) where T
    # needed for rotation matrix
    θ = mod(real(T)(θ), real(T)(2π))
    midpoint = real(T).(midpoint)
    sinθ, cosθ = sincos(real(T)(θ)) 
    return sinθ, cosθ, midpoint
end


"""
    _check_trivial_rotations!(out, arr, θ, midpoint) 

When `θ = 0 || π /2 || π || 3/2 || π` and if `midpoint` 
is in the middle of the array.
For an even array of size 4, the midpoint would need to be 2.5.
For an odd array of size 5, the midpoint would need to be 3.

In those cases, rotations are trivial just by reversing or swapping some axes.
"""
function _check_trivial_rotations!(out, arr, θ, midpoint; adjoint=false)
    if iszero(θ)
        out .= arr
        return true 
    end
    # check for special cases where rotations are trivial
    if (iseven(size(arr, 1)) && iseven(size(arr, 2)) && 
        midpoint[1] ≈ size(arr, 1) ÷ 2 + 0.5 && midpoint[2] ≈ size(arr, 2) ÷ 2 + 0.5) ||
        (isodd(size(arr, 1)) && isodd(size(arr, 2)) && 
        (midpoint[1] == size(arr, 1) ÷ 2 + 1 && midpoint[1] == size(arr, 2) ÷ 2 + 1))
        if θ ≈ π / 2 
            if adjoint == false
                out .= reverse(PermutedDimsArray(arr, (2, 1, 3)), dims=(2,))
            else
                out .= reverse(PermutedDimsArray(arr, (2, 1, 3)), dims=(1,))
            end
            return true
        elseif θ ≈ π
            out .= reverse(arr, dims=(1,2))
            return true
        elseif θ ≈ 3 / 2 * π
            if adjoint == false
                out .= reverse(PermutedDimsArray(arr, (2, 1, 3)), dims=(1,))
            else
                out .= reverse(PermutedDimsArray(arr, (2, 1, 3)), dims=(2,))
            end
            return true
        end
    end

    return false
end

