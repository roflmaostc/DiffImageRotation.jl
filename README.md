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
julia> ]add DiffImageRotation

julia> using DiffImageRotation

julia> arr = zeros((32, 32)); arr[15:19, 10:23] .= 1 

julia> imrotate(arr, rad2deg(45));

julia> imrotate(arr, deg2rad(90));

# access the docs
julia> ?imroate  
```

![](examples/example.png)

To learn more about the interpolation scheme, see this [webpage](http://www.leptonica.org/rotation.html). We implement rotation by area mapping (RAM).

## Related Packages
There is `imrotate` by [ImageTransformations.jl](https://github.com/JuliaImages/ImageTransformations.jl).
For standard image processing rather use this. It has better handling, way more options and is faster on CPUs.
But, it doesn't run with `CuArrays` and does not provide an adjoint/gradient rule.

There is `rotate` by [FourierTools.jl](https://nanoimaging.de/FourierTools.jl/dev/rotate/).
It's based on FFTs. It is based on a sinc interpolation.
