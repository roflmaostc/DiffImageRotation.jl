using CUDA, ImageTransformations,  FourierTools
using DiffImageRotation 
using BenchmarkTools 
using FFTW


imrotate = DiffImageRotation.imrotate


function f(arr)
    img = arr
    img_c = CuArray(arr)

    println("size:", size(img))
    println("CPU")
    @btime $imrotate($img, deg2rad(30))
    println("nearest")
    @btime $imrotate($img, deg2rad(30), method=:nearest)
    println("adjoint")
    @btime DiffImageRotation.∇imrotate($img, $img, deg2rad(30))
    println("adjoint nearest")
    @btime DiffImageRotation.∇imrotate($img, $img, deg2rad(30), method=:nearest)
    println("\n")
    println("CUDA")
    @btime CUDA.@sync $imrotate($img_c, deg2rad(30))
    println("nearest")
    @btime CUDA.@sync $imrotate($img_c, deg2rad(30), method=:nearest)
    println("adjoint")
    @btime CUDA.@sync DiffImageRotation.∇imrotate($img_c, $img_c, deg2rad(30))
    println("adjoint nearest")
    @btime CUDA.@sync DiffImageRotation.∇imrotate($img_c, $img_c, deg2rad(30), method=:nearest)
    println("\n\n")
end


f(randn(Float32, (2048, 2048)))
f(randn(Float32, (256, 256)))
f(randn(Float32, (512, 512, 100)))






"""
    Tested on an AMD Ryzen 5 5600X 6-Core Processo with 12 Threads and a NVIDIA GeForce RTX 3060 with Julia 1.10 on Ubuntu 22.10

```
julia> include("examples/benchmark_REPL.jl")
size:(2048, 2048)
CPU
  3.837 ms (135 allocations: 16.01 MiB)
nearest
  3.072 ms (135 allocations: 16.01 MiB)
adjoint
  7.463 ms (137 allocations: 16.01 MiB)
adjoint nearest
  4.423 ms (137 allocations: 16.01 MiB)


CUDA
  316.692 μs (103 allocations: 4.28 KiB)
nearest
  305.352 μs (103 allocations: 4.28 KiB)
adjoint
  735.298 μs (106 allocations: 4.39 KiB)
adjoint nearest
  327.322 μs (105 allocations: 4.38 KiB)



size:(256, 256)
CPU
  82.163 μs (133 allocations: 266.81 KiB)
nearest
  72.573 μs (133 allocations: 266.81 KiB)
adjoint
  131.955 μs (135 allocations: 266.91 KiB)
adjoint nearest
  94.314 μs (135 allocations: 266.91 KiB)


CUDA
  18.081 μs (85 allocations: 4.00 KiB)
nearest
  17.801 μs (85 allocations: 4.00 KiB)
adjoint
  26.091 μs (87 allocations: 4.09 KiB)
adjoint nearest
  18.141 μs (87 allocations: 4.09 KiB)



size:(512, 512, 100)
CPU
  38.045 ms (132 allocations: 100.01 MiB)
nearest
  33.698 ms (132 allocations: 100.01 MiB)
adjoint
  54.156 ms (132 allocations: 100.01 MiB)
adjoint nearest
  41.783 ms (132 allocations: 100.01 MiB)


CUDA
  1.804 ms (100 allocations: 4.31 KiB)
nearest
  1.722 ms (100 allocations: 4.31 KiB)
adjoint
  4.519 ms (100 allocations: 4.31 KiB)
adjoint nearest
  1.860 ms (100 allocations: 4.31 KiB)
```


"""

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
