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
using KernelAbstractions, CUDA, TestImages, FileIO, ImageShow, ImageTransformations, PlutoUI, PlutoTest, NDTools, FourierTools

# ╔═╡ cd42e1cc-66e3-43a7-9cc9-2adb02a4e363
using DiffImageRotation

# ╔═╡ ef18be31-0b8c-43c1-97f8-c59da7740934
imrotate = DiffImageRotation.imrotate

# ╔═╡ 8eb351c6-5cf1-4e07-a0d0-8aae66f36301
img = Float32.(testimage("cameraman"));

# ╔═╡ cc42f0d0-9438-4f27-a670-bd9e9101352d
md"# Example
Visually the results are almost identical.

Small difference are visible because *ImageTransformations* rotates around the `size(img) ./ 2 + 0.5` whereas *DiffImageRotation* rotates around `size(img) .÷ 2 + 1`.

Also *DiffImageRotation* cuts everything outside the unit circle.
*FourierTools.jl* uses a FFT based method.
"

# ╔═╡ 70f74f6d-42c9-40ce-9652-ec1ea4e614e4
@bind angle Slider(0:360, show_value=true)

# ╔═╡ 49c86f22-0961-47db-9534-6b01e7b412d2
DiffImageRotation.imrotate(img, deg2rad(angle));

# ╔═╡ 4253ef7a-24e5-4caa-8767-9b9fe29b8e2e
[simshow(DiffImageRotation.imrotate(img, deg2rad(angle))) simshow(ImageTransformations.imrotate(img, deg2rad(angle), axes(img), fillvalue=0)) simshow(FourierTools.rotate(img, deg2rad.(angle)))]

# ╔═╡ dd0f988f-ab99-43ac-8e22-21bec828b47e
res = (select_region(DiffImageRotation.imrotate(img, deg2rad(angle)), new_size=(250, 250)) ≈ select_region(ImageTransformations.imrotate(img[begin+1:end, begin+1:end], deg2rad(angle), axes(img[begin+1:end, begin+1:end]), fillvalue=0), new_size=(250, 250)))

# ╔═╡ f81d587d-c5c7-4e5d-82e6-87bb1bc950db
md"""## Compare to ImageTransformations.imrotate
In case of odd size arrays, the results match.
$(PlutoTest.@test res)
"""

# ╔═╡ 9bc17466-8a9b-4ffd-8898-8ed00d2e385e
md"# CUDA example
An immediate speedup can be observed
"

# ╔═╡ 1a5155ac-cc8d-41fd-bbc2-8d77c1d174b7
img_c = CuArray(img);

# ╔═╡ 26ed1552-ba9f-4203-af54-ad8b60e3572d
@time  imrotate(img, deg2rad(45));

# ╔═╡ b300dde6-1b95-47aa-9027-46aedbcff5fc
@time ImageTransformations.imrotate(img, deg2rad(45));

# ╔═╡ 75a9b76a-4ac6-444d-adc9-f6c5526c645b
@time FourierTools.rotate(img, deg2rad(45));

# ╔═╡ 386f327e-1a34-445c-b19e-b688d284e5a2
@time CUDA.@sync imrotate(img_c, deg2rad(45));

# ╔═╡ 8a83f412-2a47-4f1c-ad34-054cf5ca01bf
simshow(Array(imrotate(img_c, deg2rad(45))))

# ╔═╡ 67cc89f7-0e29-4bf4-bff7-f20cac12a14e
md"# More drastical for bigger 3D arrays"

# ╔═╡ d19cb77b-21c8-40bd-a84d-6a1260911241
arr = randn(Float32, (1024, 1024, 64));

# ╔═╡ 9f5cf42e-6ffa-4611-88d0-41227d0344d7
arr_c = CuArray(arr);

# ╔═╡ dac9ea4c-ae16-4ae0-b8b6-332447d33529
@time imrotate(arr, deg2rad(45));

# ╔═╡ 14d1497e-3673-436f-a371-eb7ce58d9843
@time CUDA.@sync imrotate(arr_c, deg2rad(45));

# ╔═╡ Cell order:
# ╠═1baea98c-98f4-11ee-0d37-1de73460bc8b
# ╠═d50e7470-000a-47fe-8ce5-b03e0ec01287
# ╠═cd42e1cc-66e3-43a7-9cc9-2adb02a4e363
# ╠═ef18be31-0b8c-43c1-97f8-c59da7740934
# ╠═8eb351c6-5cf1-4e07-a0d0-8aae66f36301
# ╟─cc42f0d0-9438-4f27-a670-bd9e9101352d
# ╟─70f74f6d-42c9-40ce-9652-ec1ea4e614e4
# ╠═49c86f22-0961-47db-9534-6b01e7b412d2
# ╠═4253ef7a-24e5-4caa-8767-9b9fe29b8e2e
# ╟─f81d587d-c5c7-4e5d-82e6-87bb1bc950db
# ╟─dd0f988f-ab99-43ac-8e22-21bec828b47e
# ╟─9bc17466-8a9b-4ffd-8898-8ed00d2e385e
# ╠═1a5155ac-cc8d-41fd-bbc2-8d77c1d174b7
# ╠═26ed1552-ba9f-4203-af54-ad8b60e3572d
# ╠═b300dde6-1b95-47aa-9027-46aedbcff5fc
# ╠═75a9b76a-4ac6-444d-adc9-f6c5526c645b
# ╠═386f327e-1a34-445c-b19e-b688d284e5a2
# ╠═8a83f412-2a47-4f1c-ad34-054cf5ca01bf
# ╟─67cc89f7-0e29-4bf4-bff7-f20cac12a14e
# ╠═d19cb77b-21c8-40bd-a84d-6a1260911241
# ╠═9f5cf42e-6ffa-4611-88d0-41227d0344d7
# ╠═dac9ea4c-ae16-4ae0-b8b6-332447d33529
# ╠═14d1497e-3673-436f-a371-eb7ce58d9843
