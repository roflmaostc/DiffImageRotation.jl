module DiffImageRotation

using KernelAbstractions, ChainRulesCore, Atomix

export imrotate

imrotate(arr::AbstractArray{T, 2}, θ) where T = 
    view(imrotate(reshape(arr, (size(arr,1), size(arr, 2), 1)), θ), :, :, 1)

"""
    imrotate(arr::AbstractArray, θ)

Rotates a matrix around the center pixel `size(arr) ÷ 2 .+ 1` which is the real center.
The angle `θ` is interpreted in radians.

The adjoint is defined with ChainRulesCore. This method also runs with CUDA.
If `arr` is a `AbstractArray{T, 3}`, the third dimension is interpreted as batch dimension.
"""
function imrotate(arr::AbstractArray{T, 3}, θ) where T
	@assert size(arr, 1) == size(arr, 2) "only quadratic arrays in dimension 1 and 2"
	# needed for rotation matrix
	sinθ, cosθ = sincos(θ)

	# important variables
	mid = size(arr, 1) .÷ 2 + 1
    # we only rotate pixels inside this inner circle
    # other regions are cut off
	radius = mid - 2

	# out array
	out = similar(arr)
	fill!(out, 0)

    # KernelAbstractions specific
	backend = get_backend(arr)
	kernel! = imrotate_kernel!(backend)
    # launch kernel
	kernel!(out, arr, sinθ, cosθ, mid, radius,
            ndrange=(size(arr, 3), size(arr, 2), size(arr, 1)))

	return out
end

# KernelAbstractions specific
@kernel function imrotate_kernel!(out, arr, sinθ, cosθ, mid, radius)
	k, j, i = @index(Global, NTuple)
	y = i - mid
	x = j - mid
	if y^2 + x^2 < radius ^2
		yrot = cosθ * y - sinθ * x
		xrot = sinθ * y + cosθ * x
		yrotf = floor(yrot)
		xrotf = floor(xrot)
		inew = floor(Int, yrot) + mid
		jnew = floor(Int, xrot) + mid
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
	sinθ, cosθ = sincos(θ)

	# important variables
	mid = size(arr, 1) .÷ 2 + 1
    # we only rotate pixels inside this inner circle
    # other regions are cut off
	radius = mid - 2

	# out array
	out = similar(arr)
	fill!(out, 0)

    # KernelAbstractions specific
	backend = get_backend(arr)
	kernel! = imrotate_kernel_adj!(backend)
    # launch kernel
	kernel!(out, arr, sinθ, cosθ, mid, radius,
            ndrange=(size(arr, 3), size(arr, 2), size(arr, 1)))

	return out
end



# KernelAbstractions specific
@kernel function imrotate_kernel_adj!(out, arr, sinθ, cosθ, mid, radius)
	k, j, i = @index(Global, NTuple)
	y = i - mid
	x = j - mid
	if y^2 + x^2 < radius ^2
		yrot = cosθ * y - sinθ * x
		xrot = sinθ * y + cosθ * x
		yrotf = floor(yrot)
		xrotf = floor(xrot)
		inew = floor(Int, yrot) + mid
		jnew = floor(Int, xrot) + mid

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
