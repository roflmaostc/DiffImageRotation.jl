# DiffImageRotation.jl

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

julia> arr = zeros((32, 32)); zeros[15:19, 10:23] .= 1

julia> imrotate(arr, rad2deg(45))
```

![](examples/example.png)

To learn more about the interpolation scheme, see this [webpage](http://www.leptonica.org/rotation.html). We implement rotation by area mapping (RAM).

## Related Packages
There is `imrotate` by [ImageTransformations.jl](https://github.com/JuliaImages/ImageTransformations.jl).
For standard image processing rather use this. It has better handling, way more options and is faster on CPUs.
But, it doesn't run with `CuArrays` and does not provide an adjoint/gradient rule.
