
function rotate_coordinates(sinθ, cosθ, i, j, midpoint, round_or_floor)
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
function bilinear_helper(yrot, xrot, yrot_f, xrot_f, yrot_int, xrot_int, imax, jmax)
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

function imrotate2(arr::AbstractArray{T, 3}, θ; method=:bilinear, midpoint=size(arr) .÷ 2 .+ 1,
                  adjoint=false) where T
    # needed for rotation matrix
    out = similar(arr)
    fill!(out, 0)
    θ = mod(real(T)(θ), real(T)(2π))

    backend = get_backend(arr)
    midpoint = real(T).(midpoint)
    sinθ, cosθ = sincos(real(T)(θ)) 
            kernel! = imrotate_kernel2!(backend)
            # launch kernel
            kernel!(out, arr, sinθ, cosθ, midpoint, size(arr, 1), size(arr, 2),
                    ndrange=(size(arr, 1), size(arr, 2), size(arr, 3)))
	return out
end


@kernel function imrotate_kernel2!(out, arr, sinθ, cosθ, midpoint, imax, jmax)
    i, j, k = @index(Global, NTuple)
    
    @inline yrot, xrot, yrot_f, xrot_f, yrot_int, xrot_int = rotate_coordinates(sinθ, cosθ, i, j, midpoint, floor) 
    if 1 ≤ yrot_int ≤ imax - 1&& 1 ≤ xrot_int ≤ jmax - 1 

        @inline Δi, Δj, Δi_min, Δj_min, ydiff, ydiff_diff, xdiff, xdiff_diff = 
            bilinear_helper(yrot, xrot, yrot_f, xrot_f, yrot_int, xrot_int, imax, jmax)
        @inbounds out[i, j, k] += 
            (   xdiff_diff  * ydiff_diff    * arr[yrot_int + Δi_min, xrot_int + Δj_min, k]
             +  xdiff_diff  * ydiff         * arr[yrot_int + Δi,     xrot_int + Δj_min, k]
             +  xdiff       * ydiff_diff    * arr[yrot_int + Δi_min, xrot_int + Δj,     k] 
             +  xdiff       * ydiff         * arr[yrot_int + Δi,     xrot_int + Δj,     k])
    end
end


function imrotate3(arr::AbstractArray{T, 3}, θ; method=:bilinear, midpoint=size(arr) .÷ 2 .+ 1,
                  adjoint=false) where T
    # needed for rotation matrix
    out = similar(arr)
    fill!(out, 0)
    θ = mod(real(T)(θ), real(T)(2π))

    backend = get_backend(arr)
    midpoint = real(T).(midpoint)
    sinθ, cosθ = sincos(real(T)(θ)) 
            kernel! = imrotate_kernel3!(backend)
            # launch kernel
            kernel!(out, arr, sinθ, cosθ, midpoint, size(arr, 1), size(arr, 2),
                    ndrange=(size(arr, 1)))
	return out
end


@kernel function imrotate_kernel3!(out, arr, sinθ, cosθ, midpoint, imax, jmax)
    i::UInt32  = @index(Global)
   

    @uniform l, channels, batch = size(out)

    for k in 1:batch
        for j in 1:channels
    @inline yrot, xrot, yrot_f, xrot_f, yrot_int, xrot_int = rotate_coordinates(sinθ, cosθ, i, j, midpoint, floor) 
    if 1 ≤ yrot_int ≤ imax - 1&& 1 ≤ xrot_int ≤ jmax - 1 

        @inline Δi, Δj, Δi_min, Δj_min, ydiff, ydiff_diff, xdiff, xdiff_diff = 
            bilinear_helper(yrot, xrot, yrot_f, xrot_f, yrot_int, xrot_int, imax, jmax)
        @inbounds out[i, j, k] += 
            (   xdiff_diff  * ydiff_diff    * arr[yrot_int + Δi_min, xrot_int + Δj_min, k]
             +  xdiff_diff  * ydiff         * arr[yrot_int + Δi,     xrot_int + Δj_min, k]
             +  xdiff       * ydiff_diff    * arr[yrot_int + Δi_min, xrot_int + Δj,     k] 
             +  xdiff       * ydiff         * arr[yrot_int + Δi,     xrot_int + Δj,     k])
    end
end
end
end

