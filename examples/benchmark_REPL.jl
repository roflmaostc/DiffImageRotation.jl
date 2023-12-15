using CUDA, ImageTransformations,  FourierTools
using DiffImageRotation 
using BenchmarkTools 
using FFTW


imrotate = DiffImageRotation.imrotate



println("(2048, 2048)")
img = randn(Float32, (2048, 2048))
img_c = CuArray(img) 
println("imrotate")
@btime $imrotate($img, deg2rad(30))
@btime $imrotate($img, deg2rad(30), method=:nearest)
@btime DiffImageRotation.imrotate_adj($img, deg2rad(30))
@btime CUDA.@sync $imrotate($img_c, deg2rad(30))
@btime CUDA.@sync $imrotate($img_c, deg2rad(30), method=:nearest)
@btime CUDA.@sync DiffImageRotation.imrotate_adj($img_c, deg2rad(30))
#println("ImageTransformations")
#@btime ImageTransformations.imrotate($img, deg2rad(30))
#println("FourierTools")
#@btime FourierTools.rotate($img, deg2rad(30))
#p = plan_fft(img_c)
#println("CUDA FFT")
#@btime CUDA.@sync $p * $img_c

println("(256, 256)")
img = randn(Float32, (256, 256))
img_c = CuArray(img) 
println("imrotate")
@btime $imrotate($img, deg2rad(30))
@btime $imrotate($img, deg2rad(30))
@btime DiffImageRotation.imrotate_adj($img, deg2rad(30))
@btime CUDA.@sync $imrotate($img_c, deg2rad(30))
@btime CUDA.@sync $imrotate($img_c, deg2rad(30), method=:nearest)
@btime CUDA.@sync DiffImageRotation.imrotate_adj($img_c, deg2rad(30))
#println("ImageTransformations")
#@btime ImageTransformations.imrotate($img, deg2rad(30))
#println("FourierTools")
#@btime FourierTools.rotate($img, deg2rad(30))
#p = plan_fft(img_c)
#println("CUDA FFT")
#@btime CUDA.@sync $p * $img_c

println("(512, 512, 100)")
img = randn(Float32, (256, 256, 100))
img_c = CuArray(img) 
println("imrotate")
@btime $imrotate($img, deg2rad(30))
@btime $imrotate($img, deg2rad(30), method=:nearest)
@btime DiffImageRotation.imrotate_adj($img, deg2rad(30))
@btime CUDA.@sync $imrotate($img_c, deg2rad(30))
@btime CUDA.@sync $imrotate($img_c, deg2rad(30), method=:nearest)
@btime CUDA.@sync DiffImageRotation.imrotate_adj($img_c, deg2rad(30))
#println("FourierTools")
#@btime FourierTools.rotate($img, deg2rad(30))
#p = plan_fft(img_c)
#println("CUDA FFT")
#@btime CUDA.@sync $p * $img_c


"""
Tested on a AMD Ryzen 9 5900X 12-Core Processor with 24 Threads and a NVIDIA GeForce RTX 3060 with Julia 1.9.4 on Ubuntu 22.04.

julia> include("examples/benchmark_REPL.jl")
(2048, 2048)
imrotate
  2.363 ms (256 allocations: 16.02 MiB)
  323.365 μs (96 allocations: 4.27 KiB)
ImageTransformations
  31.476 ms (3 allocations: 29.86 MiB)
FourierTools
  862.269 ms (99 allocations: 480.02 MiB)
CUDA FFT
  857.193 μs (88 allocations: 4.64 KiB)
(256, 256)
imrotate
  64.761 μs (251 allocations: 275.92 KiB)
  21.169 μs (81 allocations: 4.00 KiB)
ImageTransformations
  463.687 μs (3 allocations: 478.66 KiB)
FourierTools
  6.745 ms (87 allocations: 7.56 MiB)
CUDA FFT
  24.836 μs (85 allocations: 4.56 KiB)
(512, 512, 100)
imrotate
  4.964 ms (255 allocations: 25.02 MiB)
  470.069 μs (81 allocations: 3.91 KiB)
FourierTools
  892.931 ms (13209887 allocations: 1.09 GiB)
CUDA FFT
  1.549 ms (88 allocations: 5.02 KiB)

(2048, 2048)
imrotate
  2.484 ms (260 allocations: 16.02 MiB)
  327.553 μs (108 allocations: 4.58 KiB)
ImageTransformations
  30.345 ms (3 allocations: 29.86 MiB)
FourierTools
  847.974 ms (99 allocations: 480.02 MiB)
CUDA FFT
  862.948 μs (88 allocations: 4.64 KiB)
(256, 256)
imrotate
  75.802 μs (255 allocations: 276.81 KiB)
  22.332 μs (81 allocations: 4.12 KiB)
ImageTransformations
  469.751 μs (3 allocations: 478.66 KiB)
FourierTools
  6.654 ms (87 allocations: 7.56 MiB)
CUDA FFT
  23.805 μs (85 allocations: 4.56 KiB)
(512, 512, 100)
imrotate
  6.078 ms (255 allocations: 25.02 MiB)
  484.408 μs (82 allocations: 4.06 KiB)
FourierTools
  810.206 ms (13209887 allocations: 1.09 GiB)
CUDA FFT
  1.547 ms (88 allocations: 5.02 KiB)






julia> include("examples/benchmark_REPL.jl")
(2048, 2048)
imrotate
  2.352 ms (259 allocations: 16.02 MiB)
  1.997 ms (259 allocations: 16.02 MiB)
  4.092 ms (259 allocations: 16.02 MiB)
  319.398 μs (108 allocations: 4.58 KiB)
^[^[  316.171 μs (108 allocations: 4.58 KiB)
  925.843 μs (108 allocations: 4.77 KiB)
(256, 256)
imrotate
  63.629 μs (254 allocations: 276.77 KiB)
  64.220 μs (253 allocations: 276.73 KiB)
  135.664 μs (253 allocations: 277.12 KiB)
  21.109 μs (87 allocations: 4.22 KiB)
  20.248 μs (87 allocations: 4.22 KiB)
  32.200 μs (87 allocations: 4.41 KiB)
(512, 512, 100)
imrotate
  4.980 ms (257 allocations: 25.02 MiB)
  4.397 ms (257 allocations: 25.02 MiB)
  8.045 ms (258 allocations: 25.02 MiB)
  469.318 μs (85 allocations: 4.27 KiB)
  451.374 μs (86 allocations: 4.30 KiB)
  1.364 ms (88 allocations: 4.36 KiB)



"""

nothing
