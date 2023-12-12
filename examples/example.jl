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

# ╔═╡ 1baea98c-98f4-11ee-0d37-1de73460bc8b
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	using Revise
end

# ╔═╡ d50e7470-000a-47fe-8ce5-b03e0ec01287
using KernelAbstractions, CUDA, TestImages, FileIO, ImageShow, ImageTransformations, PlutoUI, PlutoTest, FiniteDifferences, Zygote, NDTools

# ╔═╡ cd42e1cc-66e3-43a7-9cc9-2adb02a4e363
using DiffImageRotation

# ╔═╡ 8eb351c6-5cf1-4e07-a0d0-8aae66f36301
img = Float32.(testimage("resolution_test_512"));

# ╔═╡ 70f74f6d-42c9-40ce-9652-ec1ea4e614e4
@bind angle Slider(0:360, show_value=true)

# ╔═╡ 4253ef7a-24e5-4caa-8767-9b9fe29b8e2e
simshow(DiffImageRotation.imrotate(img, deg2rad(angle)))

# ╔═╡ f81d587d-c5c7-4e5d-82e6-87bb1bc950db
md"""## Compare to ImageTransformations.imrotate
$(PlutoTest.@test res)
"""

# ╔═╡ dd0f988f-ab99-43ac-8e22-21bec828b47e
res = (select_region(DiffImageRotation.imrotate(img, deg2rad(angle)), new_size=(250, 250)) ≈ select_region(ImageTransformations.imrotate(img[begin+1:end, begin+1:end], deg2rad(angle), axes(img[begin+1:end, begin+1:end]), fillvalue=0), new_size=(250, 250)))

# ╔═╡ Cell order:
# ╠═1baea98c-98f4-11ee-0d37-1de73460bc8b
# ╠═d50e7470-000a-47fe-8ce5-b03e0ec01287
# ╠═8eb351c6-5cf1-4e07-a0d0-8aae66f36301
# ╠═cd42e1cc-66e3-43a7-9cc9-2adb02a4e363
# ╟─70f74f6d-42c9-40ce-9652-ec1ea4e614e4
# ╠═4253ef7a-24e5-4caa-8767-9b9fe29b8e2e
# ╟─f81d587d-c5c7-4e5d-82e6-87bb1bc950db
# ╟─dd0f988f-ab99-43ac-8e22-21bec828b47e
