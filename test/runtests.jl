using DiffImageRotation
using Test

using Interpolations
using Zygote, FiniteDifferences, ImageTransformations


@testset "DiffImageRotation.jl" begin
    @testset "SImple test" begin
        arr = zeros((6, 6)); arr[3:4, 4] .= 1;
        @test all(DiffImageRotation.imrotate(arr, deg2rad(45)) .≈ [0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.29289321881345254 0.585786437626905 0.0; 0.0 0.0 0.08578643762690495 1.0 0.2928932188134524 0.0; 0.0 0.0 0.0 0.08578643762690495 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0])

    end

    @testset "Compare with ImageTransformations" begin
        arr = zeros((51, 51))
        arr[15:40, 15:40] .= 1 .+ randn((26, 26))
        
        arr2 = zeros((51, 51, 5))
        arr2[15:40, 15:40, :] .= arr[15:40, 15:40] 


        for method in [:nearest, :bilinear]
            for angle in deg2rad.([0, 35, 45, 90, 135, 170, 180, 270, 360])
                res1 = DiffImageRotation.imrotate(arr, angle; method)
                res3 = DiffImageRotation.imrotate(arr2, angle; method)
                if method == :nearest
                    res2 = ImageTransformations.imrotate(arr, angle, axes(arr), method=Constant(), fillvalue=0)
                elseif method == :bilinear
                    res2 = ImageTransformations.imrotate(arr, angle, axes(arr), fillvalue=0)
                end
                @test all(1 .+ res1[20:35, 20:35] .≈ 1 .+ res2[20:35, 20:35])
                @test all(1 .+ res1[20:35, 20:35] .≈ 1 .+ res3[20:35, 20:35, 1])
                @test all(1 .+ res1[20:35, 20:35] .≈ 1 .+ res3[20:35, 20:35, 2])
            end
        end
    end

    @testset "Compare for plausibilty" begin
        arr = zeros((10, 10))
        arr[6, 6] = 1

        for method in [:bilinear, :nearest]
            @test arr ≈ DiffImageRotation.imrotate(arr, deg2rad(0); method)
            @test arr ≈ DiffImageRotation.imrotate(arr, deg2rad(90); method)
            @test arr ≈ DiffImageRotation.imrotate(arr, deg2rad(180); method)
            @test arr ≈ DiffImageRotation.imrotate(arr, deg2rad(270); method)
            @test arr ≈ DiffImageRotation.imrotate(arr, deg2rad(360); method)
        end
    end


    @testset "Test gradients" begin

        for method in [:nearest, :bilinear]
            f(x) = sum(abs2.(DiffImageRotation.imrotate(x, deg2rad(137); method)))
       
            img2 = randn((32, 32));
            grad = FiniteDifferences.grad(central_fdm(7, 1), f, img2)[1]
            grad2 = Zygote.gradient(f, img2)[1];
            @test all(.≈(1 .+ grad, 1 .+ grad2, rtol=1f-8))
            
            img2 = randn((21, 21));
            grad = FiniteDifferences.grad(central_fdm(7, 1), f, img2)[1]
            grad2 = Zygote.gradient(f, img2)[1];
            @test all(.≈(1 .+ grad, 1 .+ grad2, rtol=1f-8))

            img2 = randn((20, 14, 3));
            grad = FiniteDifferences.grad(central_fdm(7, 1), f, img2)[1]
            grad2 = Zygote.gradient(f, img2)[1];
            @test all(.≈(1 .+ grad, 1 .+ grad2, rtol=1f-8))
        end
    end




end
