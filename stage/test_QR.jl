@testset "Test set for linear_system_QR" begin
    A₁ = [2 -2 2 ; -1 2 -9 ; -2 0 -4]
    b₁ = [2, 4, -3]

    A₂ = [-8 21 3 ; 4 -3 3 ; -2 3 -1]
    b₂ = [8, 20, -8]

    σ = 10^(-12)

    d₁ = linear_system_QR(A₁,b₁)
    d₂ = linear_system_QR(A₂,b₂)

    sol₁ = [5/2, 1, -1/2]
    sol₂ = [-1, -1, 7]

    for k = 1:3
        @test abs(sol₁[k] - d₁[k]) < σ
        @test abs(sol₂[k] - d₂[k]) < σ
    end
end