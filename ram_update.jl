import LinearAlgebra.Cholesky
import LinearAlgebra.lowrankdowndate!
import LinearAlgebra.lowrankupdate!

"""
    RAM_updating!(Σ_curr::Cholesky, u::Array, α_curr::Real, iter::Int, dim::Int, α_target::Real=0.234, γ::Real=2/3)


Efficient RAM updating step via a rank-1 updating. 

The code is adapted from from https://github.com/helske/ramcmc/blob/master/inst/include/ramcmc.h)

See als: https://cran.r-project.org/web/packages/ramcmc/vignettes/ramcmc.html and http://users.jyu.fi/~mvihola/vihola_-_ram.pdf

Inputs:

- `Σ_curr`: The Chol. comp. of the current cov matrix of the proposal kernel
- `u`: vector of random numbers used in the proposal kernel 
- `α_curr`: current accaptace prob
- `iter`: idex for current iteration
- `dim`: dim of target 
- `α_target`: target accaptace prob
- `γ`: decay rate of the adaptation 
    
Returns:
    
- `Σ_new`: the Chol. comp. of the cov matrix of the proposal kernel (returned inplace)
"""
function RAM_updating!(Σ_curr::Cholesky, 
                       u::Array, 
                       α_curr::Real,
                       iter::Int,
                       dim::Int,    
                       α_target::Real=0.234,   
                       γ::Real=2/3)


    α_diff = α_curr - α_target            
    u_update = Σ_curr.L * u / sqrt(dot(u,u)) * sqrt(min(1, dim * 1/iter^γ)*abs(α_diff))
 
    if α_diff < 0. 
        return lowrankdowndate!(Σ_curr, u_update)
    else    
        return lowrankupdate!(Σ_curr, u_update)
    end


end 