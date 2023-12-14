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

# ╔═╡ 39c0575c-98f6-11ee-3560-6f0da1c69455
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	using Revise
end

# ╔═╡ 89b77660-a50d-42bd-a448-952a1bc8a3e1
using KernelAbstractions, CUDA, TestImages, FileIO, ImageShow, ImageTransformations, PlutoUI, PlutoTest, FiniteDifferences, Zygote, NDTools, BenchmarkTools

# ╔═╡ ef423eca-7dc7-478a-876f-25c1f0f3473a
using FourierTools

# ╔═╡ 6c441f76-3e60-4330-9419-21105fa8c34c
begin
	using Random
	Random.seed!(42)
end

# ╔═╡ 99b4a70f-89da-469e-a3b3-23dced140c5c
using DiffImageRotation

# ╔═╡ 34e7e177-9411-4cc6-9881-6826083b439f
# ╠═╡ disabled = true
#=╠═╡
using DiffImageRotation
  ╠═╡ =#

# ╔═╡ c7e32e2d-d860-48a7-ab06-489569222128
img = Float32.(testimage("resolution_test_1920"));

# ╔═╡ a125a3b8-9c3f-4624-a2cc-55dfca896111
img_c = CuArray(img);

# ╔═╡ 38fa2c45-773b-4ccd-a5f7-d12d4e071480
md"# Speed"

# ╔═╡ 8f50e933-ce21-4e39-a59c-ecd41e23f5bb
# ╠═╡ disabled = true
#=╠═╡
@benchmark DiffImageRotation.imrotate($img, deg2rad(45))
  ╠═╡ =#

# ╔═╡ 54608e3b-8811-446d-bf47-11609c67c6f6
# ╠═╡ disabled = true
#=╠═╡
@benchmark ImageTransformations.imrotate($img, deg2rad(45), axes($img)) evals=10 samples=10
  ╠═╡ =#

# ╔═╡ b2f1e432-dbb3-463b-90ed-3e44ecf2b9bd
# ╠═╡ disabled = true
#=╠═╡
@benchmark FourierTools.rotate($img, deg2rad(45)) evals=10 samples=10
  ╠═╡ =#

# ╔═╡ 33dcd8ea-6fe9-42a9-841e-f67b530d4624
# ╠═╡ disabled = true
#=╠═╡
@benchmark (CUDA.@sync DiffImageRotation.imrotate($img_c, deg2rad(45))) evals=3 samples=10
  ╠═╡ =#

# ╔═╡ 66dfe9be-aa25-475f-9606-c52c289ec41a
# ╠═╡ disabled = true
#=╠═╡
@benchmark (CUDA.@sync DiffImageRotation.imrotate_adj($img_c, deg2rad(45))) evals=10 samples=10
  ╠═╡ =#

# ╔═╡ 010d579b-b73f-4a1a-90f2-945a2a9af613


# ╔═╡ 9814c45f-b9d6-4d9a-9253-dda9a3410531
function multirot(img, N)
	for i in 1:N
		#img = DiffImageRotation.imrotate(img, deg2rad(360/N))
		img = ImageTransformations.imrotate(img, deg2rad(360/N), axes(img), fillvalue=0)
	end
	return img
end

# ╔═╡ 6d5326de-9b90-41a9-b66b-8a141e6ee999
img3 = Float32.(testimage("resolution_test_512"));

# ╔═╡ c9485ccd-674c-4f12-aed9-b7e688524f93
simshow(multirot(img3, 20))

# ╔═╡ 14b0cc66-734f-409d-8317-5b02d5dda183
md"# Adjoint"

# ╔═╡ a497c753-e2ef-4a63-bad2-aa4e44b1d0f6
img2 = randn((32, 32));

# ╔═╡ c4d322f0-6427-4059-ba96-87dbbd62b56f
Zygote.refresh()

# ╔═╡ 50be2f63-ffbc-4012-833b-6f092175d133
f(x) = sum(abs2.(DiffImageRotation.imrotate(x, deg2rad(137), method=:nearest)))

# ╔═╡ 5aa6af39-6504-4928-8a36-a4de0281db5a
grad = FiniteDifferences.grad(central_fdm(5, 1), f, img2)[1]

# ╔═╡ 13c8ba14-e94b-46b6-a540-5c40818ceba9
grad2 = Zygote.gradient(f, img2)[1];

# ╔═╡ 0a8842d4-423e-4028-a83f-a4803f68c4a4
simshow(grad)

# ╔═╡ b88a894e-690d-41af-8fc2-c25f95b5e015
sum(grad)

# ╔═╡ c814a821-3b73-499c-b4a2-d47d8fa7b00c
all(.≈(1 .+ grad, 1 .+ grad2, rtol=1f-4))

# ╔═╡ fb17faa0-3e35-42a6-9615-cc1b18829965
simshow(.≈(1 .+ grad, 1 .+ grad2, rtol=1f-4))

# ╔═╡ 0628346f-3d56-49a4-9470-079fbab67376
# ╠═╡ disabled = true
#=╠═╡
@benchmark f($img_c) evals=10 samples=10
  ╠═╡ =#

# ╔═╡ cdf1d103-e72d-4159-ae5f-bebeba51faeb
# ╠═╡ disabled = true
#=╠═╡
@benchmark (CUDA.@sync Zygote.gradient($f, $img_c)) evals=10 samples=10
  ╠═╡ =#

# ╔═╡ 7e00fb11-c11d-4491-b173-07421aa6160f
# ╠═╡ disabled = true
#=╠═╡
@benchmark (CUDA.@sync DiffImageRotation.imrotate_adj($img_c, deg2rad(45)))  evals=10 samples=10
  ╠═╡ =#

# ╔═╡ 95a1da23-c124-4dae-b3dc-4ace5e1d8727
arr3 = zeros((32, 32)); arr3[16:18, 17-16:17+15] .= 1

# ╔═╡ 47568e40-9165-4d87-88e3-73a26bb261dc
DiffImageRotation.imrotate(arr3, rad2deg(45))

# ╔═╡ da9b976e-938f-45a2-83f0-bb37a51bc8a9
[simshow(arr3) simshow(ones(32,1)) simshow(DiffImageRotation.imrotate(arr3, rad2deg(45))) simshow(ones(32,1)) simshow(DiffImageRotation.imrotate(arr3, deg2rad(90)))]

# ╔═╡ ce6077ae-5d6a-4a0f-af85-791e6d46e027
@bind angle4 Slider(0:360, show_value=true)

# ╔═╡ 44ab1aac-b977-42b4-982e-74d8911d1f33


# ╔═╡ 0cc4866e-30e2-4260-bd7d-c41afdd68111
simshow(DiffImageRotation.imrotate(arr3, deg2rad(angle4), method=:nearest))

# ╔═╡ cd22b8d7-7cf5-40b3-a68d-3b8bae7ec6cf
# ╠═╡ disabled = true
#=╠═╡
using DiffImageRotation
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═39c0575c-98f6-11ee-3560-6f0da1c69455
# ╠═89b77660-a50d-42bd-a448-952a1bc8a3e1
# ╠═ef423eca-7dc7-478a-876f-25c1f0f3473a
# ╠═34e7e177-9411-4cc6-9881-6826083b439f
# ╠═c7e32e2d-d860-48a7-ab06-489569222128
# ╠═a125a3b8-9c3f-4624-a2cc-55dfca896111
# ╟─38fa2c45-773b-4ccd-a5f7-d12d4e071480
# ╠═8f50e933-ce21-4e39-a59c-ecd41e23f5bb
# ╠═54608e3b-8811-446d-bf47-11609c67c6f6
# ╠═b2f1e432-dbb3-463b-90ed-3e44ecf2b9bd
# ╠═33dcd8ea-6fe9-42a9-841e-f67b530d4624
# ╠═66dfe9be-aa25-475f-9606-c52c289ec41a
# ╠═010d579b-b73f-4a1a-90f2-945a2a9af613
# ╠═9814c45f-b9d6-4d9a-9253-dda9a3410531
# ╠═6d5326de-9b90-41a9-b66b-8a141e6ee999
# ╠═c9485ccd-674c-4f12-aed9-b7e688524f93
# ╟─14b0cc66-734f-409d-8317-5b02d5dda183
# ╠═6c441f76-3e60-4330-9419-21105fa8c34c
# ╠═a497c753-e2ef-4a63-bad2-aa4e44b1d0f6
# ╠═c4d322f0-6427-4059-ba96-87dbbd62b56f
# ╠═50be2f63-ffbc-4012-833b-6f092175d133
# ╠═5aa6af39-6504-4928-8a36-a4de0281db5a
# ╠═99b4a70f-89da-469e-a3b3-23dced140c5c
# ╠═13c8ba14-e94b-46b6-a540-5c40818ceba9
# ╠═0a8842d4-423e-4028-a83f-a4803f68c4a4
# ╠═b88a894e-690d-41af-8fc2-c25f95b5e015
# ╠═c814a821-3b73-499c-b4a2-d47d8fa7b00c
# ╠═fb17faa0-3e35-42a6-9615-cc1b18829965
# ╠═0628346f-3d56-49a4-9470-079fbab67376
# ╠═cdf1d103-e72d-4159-ae5f-bebeba51faeb
# ╠═7e00fb11-c11d-4491-b173-07421aa6160f
# ╠═95a1da23-c124-4dae-b3dc-4ace5e1d8727
# ╠═47568e40-9165-4d87-88e3-73a26bb261dc
# ╠═da9b976e-938f-45a2-83f0-bb37a51bc8a9
# ╠═ce6077ae-5d6a-4a0f-af85-791e6d46e027
# ╠═44ab1aac-b977-42b4-982e-74d8911d1f33
# ╠═0cc4866e-30e2-4260-bd7d-c41afdd68111
# ╠═cd22b8d7-7cf5-40b3-a68d-3b8bae7ec6cf
