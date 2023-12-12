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
@btime CUDA.@sync $imrotate($img_c, deg2rad(30))
println("ImageTransformations")
@btime ImageTransformations.imrotate($img, deg2rad(30))
println("FourierTools")
@btime FourierTools.rotate($img, deg2rad(30))
p = plan_fft(img_c)
println("CUDA FFT")
@btime CUDA.@sync $p * $img_c

println("(256, 256)")
img = randn(Float32, (256, 256))
img_c = CuArray(img) 
println("imrotate")
@btime $imrotate($img, deg2rad(30))
@btime CUDA.@sync $imrotate($img_c, deg2rad(30))
println("ImageTransformations")
@btime ImageTransformations.imrotate($img, deg2rad(30))
println("FourierTools")
@btime FourierTools.rotate($img, deg2rad(30))
p = plan_fft(img_c)
println("CUDA FFT")
@btime CUDA.@sync $p * $img_c

println("(512, 512, 100)")
img = randn(Float32, (256, 256, 100))
img_c = CuArray(img) 
println("imrotate")
@btime $imrotate($img, deg2rad(30))
@btime CUDA.@sync $imrotate($img_c, deg2rad(30))
println("FourierTools")
@btime FourierTools.rotate($img, deg2rad(30))
p = plan_fft(img_c)
println("CUDA FFT")
@btime CUDA.@sync $p * $img_c


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
"""

nothing
