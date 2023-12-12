# DiffImageRotation.jl

[![Build Status](https://github.com/roflmaostc/DiffImageRotation.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/roflmaostc/DiffImageRotation.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/roflmaostc/DiffImageRotation.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/roflmaostc/DiffImageRotation.jl) 

This package serves only one purpose.
To provide a bilinear interpolation based image rotation which works with CUDA and multithreaded CPUs
(thanks to [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)).
It can rotate images but also 3D arrays with a trailing batch dimension.

Further, it has registered adjoints with [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl).

Try it out with:
```julia
# install package with typing ]
# or ]add https://github.com/roflmaostc/DiffImageRotation.jl/
julia> ]add DiffImageRotation 

julia> using DiffImageRotation

julia> arr = zeros((32, 32)); arr[15:19, 10:23] .= 1 

julia> imrotate(arr, rad2deg(45));

julia> imrotate(arr, deg2rad(90));

# access the docs
julia> ?imrotate  
```

![](examples/example.png)

To learn more about the interpolation scheme, see this [webpage](http://www.leptonica.org/rotation.html). We implement rotation by area mapping (RAM).

## Related Packages
There is `imrotate` by [ImageTransformations.jl](https://github.com/JuliaImages/ImageTransformations.jl).
For standard image processing rather use this. It has better handling and way more options.
But, it doesn't run with `CuArrays` and does not provide an adjoint/gradient rule.

There is `rotate` by [FourierTools.jl](https://nanoimaging.de/FourierTools.jl/dev/rotate/).
It's based on FFTs. It is based on a sinc interpolation.


## Benchmarks
Tested on a AMD Ryzen 9 5900X 12-Core Processor with 24 Threads and a NVIDIA GeForce RTX 3060 with Julia 1.9.4 on Ubuntu 22.04.
|                 | DiffImageRotation.jl | CUDA DiffImageRotation.jl | FourierTools.jl | ImageTransformations.jl | CUDA FFT (as CUDA reference) | torchvision CUDA | torchvision CPU |
|-----------------|----------------------|---------------------------|-----------------|-------------------------|------------------------------|------------------|-----------------|
| (2048, 2048)    | 2.4ms                | 0.32ms                    | 860ms           | 31ms                    | 0.86ms                       | 2.1ms            | 45ms            |
| (256, 256)      | 65µs                 | 21µs                      | 6700µs          | 463µs                   | 25µs                         | 168µs            | 640µs           |
| (512, 512, 100) | 5.0ms                | 0.47ms                    | 890ms           | not possible            | 1.5ms                        | 0.9ms           | 27.1ms          |

