### A Pluto.jl notebook ###
# v0.19.35

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 36306bf6-98db-11ee-1eaa-4fd912050916
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	using Revise
end

# ╔═╡ 591cb769-5c19-45c5-b191-30c60828e468
using KernelAbstractions, CUDA, TestImages, FileIO, ImageShow, ImageTransformations, PlutoUI, PlutoTest, FiniteDifferences, Zygote

# ╔═╡ fbf6e63b-505d-4442-8420-a9edf87111ab
using BenchmarkTools

# ╔═╡ 5848314f-f97b-45bb-9dd3-228900c9c36a
using ChainRulesCore

# ╔═╡ 4714a5d1-3e5a-4861-84fd-1afecc7a7510
function rotate(arr, θ)
	@assert size(arr, 1) == size(arr, 2)
	# needed for rotation matrix
	sinθ, cosθ = sincos(θ)

	# important variables
	mid = size(arr, 1) .÷ 2 + 1
	radius = mid - 2

	# out array
	out = similar(arr)
	fill!(out, 0)
	Threads.@threads for j in 1:size(arr, 2)
		for i in 1:size(arr, 1)
			y = i - mid
			x = j - mid
			if y^2 + x^2 < radius ^2
				yrot = cosθ * y - sinθ * x
				xrot = sinθ * y + cosθ * x
				yrotf = floor(yrot)
				xrotf = floor(xrot)
				inew = floor(Int, yrot) + mid
				jnew = floor(Int, xrot) + mid

				@inbounds out[i, j] = ((1 - (xrot - xrotf)) * (1 - (yrot - yrotf)) * arr[inew, jnew]
							 + (1 - (xrot - xrotf)) * (yrot - yrotf) * arr[inew + 1, jnew]
							 + (xrot - xrotf) * (1 - (yrot - yrotf)) * arr[inew, jnew + 1] 
							 + (xrot - xrotf) * (yrot - yrotf) * arr[inew + 1, jnew + 1])
				
			end
		end
	end
	return out
end

# ╔═╡ 4d4cb538-0610-4f77-9005-3467aea3b9d3
function rotate_adj(arr, θ)
	@assert size(arr, 1) == size(arr, 2)
	# needed for rotation matrix
	sinθ, cosθ = sincos(θ)

	# important variables
	mid = size(arr, 1) .÷ 2 + 1
	radius = mid - 2

	# out array
	out = similar(arr)
	fill!(out, 0)
	Threads.@threads for j in 1:size(arr, 2)
		for i in 1:size(arr, 1)
			y = i - mid
			x = j - mid
			if y^2 + x^2 < radius ^2
				yrot = cosθ * y - sinθ * x
				xrot = sinθ * y + cosθ * x
				yrotf = floor(yrot)
				xrotf = floor(xrot)
				inew = floor(Int, yrot) + mid
				jnew = floor(Int, xrot) + mid

				o = arr[i, j]
				out[inew, jnew] += (1 - (xrot - xrotf)) * (1 - (yrot - yrotf)) * o
				out[inew + 1, jnew] += (1 - (xrot - xrotf)) * (yrot - yrotf) * o
				out[inew, jnew + 1] += (xrot - xrotf) * (1 - (yrot - yrotf)) * o
				out[inew + 1, jnew + 1] += (xrot - xrotf) * (yrot - yrotf) * o
			end
		end
	end
	return out
end

# ╔═╡ 2d60e260-92a6-4914-9bb5-8fb8e5803dec
function ChainRulesCore.rrule(rotate, array, angle)
	function pb_rotate(ȳ)
        f̄ = NoTangent()

		ad = rotate_adj(ȳ, angle)
        return NoTangent(), ad, NoTangent()
    end    
	return rotate(array, angle), pb_rotate
end

# ╔═╡ 64f971e7-acbf-43d2-852c-50db92e9d1c5
function imrotate(arr, θ)
	backend = get_backend(arr)
	@assert size(arr, 1) == size(arr, 2)
	# needed for rotation matrix
	sinθ, cosθ = sincos(θ)

	# important variables
	mid = size(arr, 1) .÷ 2 + 1
	radius = mid - 2

	# out array
	out = similar(arr)
	fill!(out, 0)

	kernel! = imrotate_kernel!(backend)
	kernel!(out, arr, sinθ, cosθ, mid, radius,
		    ndrange=(size(arr, 2), size(arr, 1)))

	return out
end

# ╔═╡ 31455f34-c7d0-4237-b31d-8ea05ebf55f5
@kernel function imrotate_kernel!(out, arr, sinθ, cosθ, mid, radius)
	j, i = @index(Global, NTuple)
	y = i - mid
	x = j - mid
	if y^2 + x^2 < radius ^2
		yrot = cosθ * y - sinθ * x
		xrot = sinθ * y + cosθ * x
		yrotf = floor(yrot)
		xrotf = floor(xrot)
		inew = floor(Int, yrot) + mid
		jnew = floor(Int, xrot) + mid

		@inbounds out[i, j] = ((1 - (xrot - xrotf)) * (1 - (yrot - yrotf)) * arr[inew, jnew]
					 + (1 - (xrot - xrotf)) * (yrot - yrotf) * arr[inew + 1, jnew]
					 + (xrot - xrotf) * (1 - (yrot - yrotf)) * arr[inew, jnew + 1] 
					 + (xrot - xrotf) * (yrot - yrotf) * arr[inew + 1, jnew + 1])
	end
end

# ╔═╡ c7c0b3aa-c05e-4c98-8a1d-14bfa4732765
function imrotate3(arr, θ)
	backend = get_backend(arr)
	@assert size(arr, 1) == size(arr, 2)
	# needed for rotation matrix
	sinθ, cosθ = sincos(θ)

	# important variables
	mid = size(arr, 1) .÷ 2 + 1
	radius = mid - 2

	# out array
	out = similar(arr)
	fill!(out, 0)

	kernel! = imrotate_kernel3!(backend)
	kernel!(out, arr, sinθ, cosθ, mid, radius,
		    ndrange=(size(arr, 3), size(arr, 2), size(arr, 1)))

	return out
end

# ╔═╡ 0804fa88-4805-4628-b75f-cd4faec7256a
@kernel function imrotate_kernel3!(out, arr, sinθ, cosθ, mid, radius)
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

		@inbounds out[i, j, k] = ((1 - (xrot - xrotf)) * (1 - (yrot - yrotf)) * arr[inew, jnew, k]
					 + (1 - (xrot - xrotf)) * (yrot - yrotf) * arr[inew + 1, jnew, k]
					 + (xrot - xrotf) * (1 - (yrot - yrotf)) * arr[inew, jnew + 1, k] 
					 + (xrot - xrotf) * (yrot - yrotf) * arr[inew + 1, jnew + 1, k])
	end
end

# ╔═╡ b71867f2-577b-4a31-9e5a-d99d8fdd8eac
# ╠═╡ disabled = true
#=╠═╡
CUDA.@time CUDA.@sync imrotate(big_c, deg2rad(45));
  ╠═╡ =#

# ╔═╡ 4e5f007b-1925-484a-9e05-63259f26c955
# ╠═╡ disabled = true
#=╠═╡
@benchmark CUDA.@sync imrotate3($big_c2, deg2rad(45))
  ╠═╡ =#

# ╔═╡ 1833e31a-f6dc-4ce1-b159-fad099b03b4b
@benchmark CUDA.@sync imrotate3($big_c2, deg2rad(45))

# ╔═╡ bf95687c-c8eb-4c2e-b021-0387181b42ed
CUDA.@time CUDA.@sync imrotate3(big_c2, deg2rad(45));

# ╔═╡ 928fc487-a15a-4c01-b44b-a0119c9aa3c4
@time imrotate(big, deg2rad(45));

# ╔═╡ 3fbdd801-4601-4386-a72e-c3f464dd566f


# ╔═╡ 85e86c16-75f8-417a-b90e-6d1bf6fc04ac
@benchmark CUDA.@sync imrotate($big_c, deg2rad(45))

# ╔═╡ 45039708-41fb-4ddc-8e80-62f5aca5900a
@bind angle3 Slider(0:360, show_value=true)

# ╔═╡ f2593988-32ee-49b3-ba75-6b11ca1d568b
f(x) = sum(abs2.(rotate(x, deg2rad(angle3))))

# ╔═╡ 4df0f225-d596-4b38-b3a1-2ed7349e009b
big = randn((8192, 8192));

# ╔═╡ 681b8003-afd4-4a6c-a1b1-75a29a6e4401
big_c = CuArray(big);

# ╔═╡ 54a45e1c-f13c-4f0c-a450-02cb9f7fd391
big_c2 = reshape(big_c, (8192, 8192, 1));

# ╔═╡ 7f27ed82-1bb1-4bf0-9f0f-e538c6264cc1
imrotate(big, deg2rad(45)) ≈ rotate(big, deg2rad(45))

# ╔═╡ 100649a5-affe-40bd-832d-e4bce5d4d758
Array(imrotate(big_c, deg2rad(45))) ≈ rotate(big, deg2rad(45))

# ╔═╡ 375412dd-8556-44a2-9ba3-1a5651639dd6
@benchmark rotate($big, deg2rad(45))

# ╔═╡ 5bc3371f-7c5e-4ece-bd7d-76c9d0827af3
@benchmark rotate_adj($big, deg2rad(45))

# ╔═╡ 7e8c8b20-0967-4485-a074-29481c784622
@time rotate(big, deg2rad(45));

# ╔═╡ d5f2a860-7307-4f69-801e-b07c695360f1
begin
	array_odd = zeros(15, 15)
	array_odd[9, 9] = 1
end

# ╔═╡ aa16fd1f-4c34-4a66-8e30-96b2678a32cb
grad = FiniteDifferences.grad(central_fdm(5, 1), f, array_odd)[1]

# ╔═╡ 44d19334-dbea-4d42-bf35-aeb733a7ecbc
simshow(abs.(grad))

# ╔═╡ 8bc5e9c7-a2c3-4375-98a1-49ccdd3036fa
grad2 = Zygote.gradient(f, array_odd)[1]

# ╔═╡ a8efda67-69ef-4ffc-9b44-78451a220dcc
simshow(grad2)

# ╔═╡ 4395d0ec-fa69-4342-81cc-899fddf7e017
simshow(array_odd)

# ╔═╡ 9f00ba8b-fb8d-4ee8-8c98-8bdfdbf73e7b
@bind angle Slider(0:360, show_value=true)

# ╔═╡ 05c5ac73-7932-456a-8fa1-7cdee950bbf7
simshow(rotate_adj(2 .* array_odd, deg2rad(angle)))

# ╔═╡ a78561d6-0721-4a4c-be42-383c5f639e83
array_odd_rotated = rotate(array_odd, deg2rad(angle));

# ╔═╡ 22777377-eabe-424a-afb5-6babc9fe074c
array_odd_rotated2 = imrotate(array_odd, deg2rad(angle), axes(array_odd), fillvalue=0);

# ╔═╡ 4a5982d4-f6ef-4bce-8daa-a6b6ac32302d
PlutoTest.@test array_odd_rotated2 ≈ array_odd_rotated

# ╔═╡ 7abce71b-9964-4e2e-a24c-231b992b6116
[simshow(array_odd_rotated, γ=0.01) simshow(array_odd_rotated2)]

# ╔═╡ 084700cf-8ebf-4493-a6e6-4dff924afb54


# ╔═╡ cb524d46-eea7-43de-9c10-9264e7fc5d18
img = Float32.(testimage("resolution_test_512"));

# ╔═╡ 2c32b1cc-d138-4e40-927e-aa903a722e90
simshow(img)

# ╔═╡ d4396166-378e-4885-8ed4-0d0a7b85c3b7
@bind angle2 Slider(0:360, show_value=true)

# ╔═╡ 03c1d9b4-fc5e-4a22-85ed-b98211eee91a
img_rot = rotate(img, deg2rad(angle2));

# ╔═╡ a8b1948e-dc63-4712-aafc-b45793ae4c3b
img_rot2 = imrotate(img, deg2rad(angle2), axes(img), fillvalue=0);

# ╔═╡ da9bbe60-de6d-41b8-9bc2-3dd50bdba735
 simshow(img_rot)

# ╔═╡ 256d2d90-8968-4aca-b20f-fdd0277ac4e9
simshow(img_rot2)

# ╔═╡ Cell order:
# ╠═36306bf6-98db-11ee-1eaa-4fd912050916
# ╠═591cb769-5c19-45c5-b191-30c60828e468
# ╠═fbf6e63b-505d-4442-8420-a9edf87111ab
# ╠═5848314f-f97b-45bb-9dd3-228900c9c36a
# ╠═4714a5d1-3e5a-4861-84fd-1afecc7a7510
# ╠═4d4cb538-0610-4f77-9005-3467aea3b9d3
# ╠═2d60e260-92a6-4914-9bb5-8fb8e5803dec
# ╠═64f971e7-acbf-43d2-852c-50db92e9d1c5
# ╠═31455f34-c7d0-4237-b31d-8ea05ebf55f5
# ╠═c7c0b3aa-c05e-4c98-8a1d-14bfa4732765
# ╠═0804fa88-4805-4628-b75f-cd4faec7256a
# ╠═b71867f2-577b-4a31-9e5a-d99d8fdd8eac
# ╠═4e5f007b-1925-484a-9e05-63259f26c955
# ╠═1833e31a-f6dc-4ce1-b159-fad099b03b4b
# ╠═bf95687c-c8eb-4c2e-b021-0387181b42ed
# ╠═928fc487-a15a-4c01-b44b-a0119c9aa3c4
# ╠═3fbdd801-4601-4386-a72e-c3f464dd566f
# ╠═681b8003-afd4-4a6c-a1b1-75a29a6e4401
# ╠═54a45e1c-f13c-4f0c-a450-02cb9f7fd391
# ╠═85e86c16-75f8-417a-b90e-6d1bf6fc04ac
# ╠═7f27ed82-1bb1-4bf0-9f0f-e538c6264cc1
# ╠═100649a5-affe-40bd-832d-e4bce5d4d758
# ╠═f2593988-32ee-49b3-ba75-6b11ca1d568b
# ╠═aa16fd1f-4c34-4a66-8e30-96b2678a32cb
# ╠═8bc5e9c7-a2c3-4375-98a1-49ccdd3036fa
# ╠═45039708-41fb-4ddc-8e80-62f5aca5900a
# ╠═a8efda67-69ef-4ffc-9b44-78451a220dcc
# ╠═44d19334-dbea-4d42-bf35-aeb733a7ecbc
# ╠═05c5ac73-7932-456a-8fa1-7cdee950bbf7
# ╠═4df0f225-d596-4b38-b3a1-2ed7349e009b
# ╠═375412dd-8556-44a2-9ba3-1a5651639dd6
# ╠═5bc3371f-7c5e-4ece-bd7d-76c9d0827af3
# ╠═7e8c8b20-0967-4485-a074-29481c784622
# ╠═d5f2a860-7307-4f69-801e-b07c695360f1
# ╠═4395d0ec-fa69-4342-81cc-899fddf7e017
# ╠═9f00ba8b-fb8d-4ee8-8c98-8bdfdbf73e7b
# ╠═a78561d6-0721-4a4c-be42-383c5f639e83
# ╠═22777377-eabe-424a-afb5-6babc9fe074c
# ╠═4a5982d4-f6ef-4bce-8daa-a6b6ac32302d
# ╠═7abce71b-9964-4e2e-a24c-231b992b6116
# ╠═084700cf-8ebf-4493-a6e6-4dff924afb54
# ╠═cb524d46-eea7-43de-9c10-9264e7fc5d18
# ╠═2c32b1cc-d138-4e40-927e-aa903a722e90
# ╠═03c1d9b4-fc5e-4a22-85ed-b98211eee91a
# ╠═a8b1948e-dc63-4712-aafc-b45793ae4c3b
# ╠═d4396166-378e-4885-8ed4-0d0a7b85c3b7
# ╠═da9bbe60-de6d-41b8-9bc2-3dd50bdba735
# ╠═256d2d90-8968-4aca-b20f-fdd0277ac4e9
