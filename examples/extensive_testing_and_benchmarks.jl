### A Pluto.jl notebook ###
# v0.19.35

using Markdown
using InteractiveUtils

# ╔═╡ 39c0575c-98f6-11ee-3560-6f0da1c69455
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	using Revise
end

# ╔═╡ 89b77660-a50d-42bd-a448-952a1bc8a3e1
using KernelAbstractions, CUDA, TestImages, FileIO, ImageShow, ImageTransformations, PlutoUI, PlutoTest, FiniteDifferences, Zygote, NDTools, BenchmarkTools

# ╔═╡ 34e7e177-9411-4cc6-9881-6826083b439f
using DiffImageRotation

# ╔═╡ 6c441f76-3e60-4330-9419-21105fa8c34c
begin
	using Random
	Random.seed!(42)
end

# ╔═╡ c7e32e2d-d860-48a7-ab06-489569222128
img = Float32.(testimage("resolution_test_1920"));

# ╔═╡ a125a3b8-9c3f-4624-a2cc-55dfca896111
img_c = CuArray(img);

# ╔═╡ 38fa2c45-773b-4ccd-a5f7-d12d4e071480
md"# Speed"

# ╔═╡ 8f50e933-ce21-4e39-a59c-ecd41e23f5bb
@benchmark DiffImageRotation.imrotate($img, deg2rad(45)) evals=10 samples=10

# ╔═╡ 54608e3b-8811-446d-bf47-11609c67c6f6
@benchmark ImageTransformations.imrotate($img, deg2rad(45), axes($img)) evals=10 samples=10

# ╔═╡ 33dcd8ea-6fe9-42a9-841e-f67b530d4624
@benchmark (CUDA.@sync DiffImageRotation.imrotate($img_c, deg2rad(45))) evals=3 samples=10

# ╔═╡ 66dfe9be-aa25-475f-9606-c52c289ec41a
@benchmark (CUDA.@sync DiffImageRotation.imrotate_adj($img_c, deg2rad(45))) evals=10 samples=10

# ╔═╡ 010d579b-b73f-4a1a-90f2-945a2a9af613


# ╔═╡ 9814c45f-b9d6-4d9a-9253-dda9a3410531


# ╔═╡ 14b0cc66-734f-409d-8317-5b02d5dda183
md"# Adjoint"

# ╔═╡ a497c753-e2ef-4a63-bad2-aa4e44b1d0f6
img2 = randn((32, 32));

# ╔═╡ 50be2f63-ffbc-4012-833b-6f092175d133
f(x) = sum(abs2.(DiffImageRotation.imrotate(x, deg2rad(137 ))))

# ╔═╡ 5aa6af39-6504-4928-8a36-a4de0281db5a
grad = FiniteDifferences.grad(central_fdm(5, 1), f, img2)[1]

# ╔═╡ 13c8ba14-e94b-46b6-a540-5c40818ceba9
grad2 = Zygote.gradient(f, img2)[1];

# ╔═╡ 0a8842d4-423e-4028-a83f-a4803f68c4a4
simshow(grad2)

# ╔═╡ c814a821-3b73-499c-b4a2-d47d8fa7b00c
all(.≈(1 .+ grad, 1 .+ grad2, rtol=1f-5))

# ╔═╡ 0628346f-3d56-49a4-9470-079fbab67376
@benchmark f($img_c) evals=10 samples=10

# ╔═╡ cdf1d103-e72d-4159-ae5f-bebeba51faeb
@benchmark (CUDA.@sync Zygote.gradient($f, $img_c)) evals=10 samples=10

# ╔═╡ 7e00fb11-c11d-4491-b173-07421aa6160f
@benchmark (CUDA.@sync DiffImageRotation.imrotate_adj($img_c, deg2rad(45)))  evals=10 samples=10

# ╔═╡ Cell order:
# ╠═39c0575c-98f6-11ee-3560-6f0da1c69455
# ╠═89b77660-a50d-42bd-a448-952a1bc8a3e1
# ╠═34e7e177-9411-4cc6-9881-6826083b439f
# ╠═c7e32e2d-d860-48a7-ab06-489569222128
# ╠═a125a3b8-9c3f-4624-a2cc-55dfca896111
# ╟─38fa2c45-773b-4ccd-a5f7-d12d4e071480
# ╠═8f50e933-ce21-4e39-a59c-ecd41e23f5bb
# ╠═54608e3b-8811-446d-bf47-11609c67c6f6
# ╠═33dcd8ea-6fe9-42a9-841e-f67b530d4624
# ╠═66dfe9be-aa25-475f-9606-c52c289ec41a
# ╠═010d579b-b73f-4a1a-90f2-945a2a9af613
# ╠═9814c45f-b9d6-4d9a-9253-dda9a3410531
# ╟─14b0cc66-734f-409d-8317-5b02d5dda183
# ╠═6c441f76-3e60-4330-9419-21105fa8c34c
# ╠═a497c753-e2ef-4a63-bad2-aa4e44b1d0f6
# ╠═50be2f63-ffbc-4012-833b-6f092175d133
# ╠═5aa6af39-6504-4928-8a36-a4de0281db5a
# ╠═13c8ba14-e94b-46b6-a540-5c40818ceba9
# ╠═0a8842d4-423e-4028-a83f-a4803f68c4a4
# ╠═c814a821-3b73-499c-b4a2-d47d8fa7b00c
# ╠═0628346f-3d56-49a4-9470-079fbab67376
# ╠═cdf1d103-e72d-4159-ae5f-bebeba51faeb
# ╠═7e00fb11-c11d-4491-b173-07421aa6160f
