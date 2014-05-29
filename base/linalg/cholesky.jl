##########################
# Cholesky Factorization #
##########################

immutable Cholesky{T} <: Factorization{T}
    UL::Matrix{T}
    uplo::Char
end
immutable CholeskyPivoted{T} <: Factorization{T}
    UL::Matrix{T}
    uplo::Char
    piv::Vector{BlasInt}
    rank::BlasInt
    tol::Real
    info::BlasInt
end

function cholfact!{T<:BlasFloat}(A::StridedMatrix{T}, uplo::Symbol=:U; pivot=false, tol=0.0)
    uplochar = string(uplo)[1]
    if pivot
        A, piv, rank, info = LAPACK.pstrf!(uplochar, A, tol)
        return CholeskyPivoted{T}(A, uplochar, piv, rank, tol, info)
    else
        C, info = LAPACK.potrf!(uplochar, A)
        return @assertposdef Cholesky(C, uplochar) info
    end
end
cholfact{T<:BlasFloat}(A::StridedMatrix{T}, uplo::Symbol=:U; pivot=false, tol=0.0) = cholfact!(copy(A), uplo, pivot=pivot, tol=tol)
cholfact{T}(A::StridedMatrix{T}, uplo::Symbol=:U; pivot=false, tol=0.0) = (S = promote_type(typeof(sqrt(one(T))),Float32); S != T ? cholfact!(convert(AbstractMatrix{S},A), uplo, pivot=pivot, tol=tol) : cholfact!(copy(A), uplo, pivot=pivot, tol=tol)) # When julia Cholesky has been implemented, the promotion should be changed.
cholfact(x::Number) = @assertposdef Cholesky(fill(sqrt(x), 1, 1), :U) !(imag(x) == 0 && real(x) > 0)

chol(A::Union(Number, AbstractMatrix), uplo::Symbol) = cholfact(A, uplo)[uplo]
chol(A::Union(Number, AbstractMatrix)) = triu!(cholfact(A, :U).UL)

convert{T}(::Type{Cholesky{T}},C::Cholesky) = Cholesky(convert(AbstractMatrix{T},C.UL),C.uplo)
convert{T}(::Type{Factorization{T}}, C::Cholesky) = convert(Cholesky{T}, C)
convert{T}(::Type{CholeskyPivoted{T}},C::CholeskyPivoted) = CholeskyPivoted(convert(AbstractMatrix{T},C.UL),C.uplo,C.piv,C.rank,C.tol,C.info)
convert{T}(::Type{Factorization{T}}, C::CholeskyPivoted) = convert(CholeskyPivoted{T}, C)

function full{T<:BlasFloat}(C::Cholesky{T})
    if C.uplo == 'U'
        BLAS.trmm!('R', C.uplo, 'N', 'N', one(T), C.UL, tril!(C.UL'))
    else
        BLAS.trmm!('L', C.uplo, 'N', 'N', one(T), C.UL, triu!(C.UL'))
    end
end

size(C::Union(Cholesky, CholeskyPivoted)) = size(C.UL)
size(C::Union(Cholesky, CholeskyPivoted), d::Integer) = size(C.UL,d)

function getindex(C::Cholesky, d::Symbol)
    d == :U && return Triangular(triu!(symbol(C.uplo) == d ? C.UL : C.UL'),:U)
    d == :L && return Triangular(tril!(symbol(C.uplo) == d ? C.UL : C.UL'),:L)
    d == :UL && return Triangular(C.UL, symbol(C.uplo))
    throw(KeyError(d))
end
function getindex{T<:BlasFloat}(C::CholeskyPivoted{T}, d::Symbol)
    d == :U && return triu!(symbol(C.uplo) == d ? C.UL : C.UL')
    d == :L && return tril!(symbol(C.uplo) == d ? C.UL : C.UL')
    d == :p && return C.piv
    if d == :P
        n = size(C, 1)
        P = zeros(T, n, n)
        for i=1:n
            P[C.piv[i],i] = one(T)
        end
        return P
    end
    throw(KeyError(d))
end

show(io::IO, C::Cholesky) = (println("$(typeof(C)) with factor:");show(io,C[symbol(C.uplo)]))

A_ldiv_B!{T<:BlasFloat}(C::Cholesky{T}, B::StridedVecOrMat{T}) = LAPACK.potrs!(C.uplo, C.UL, B)
A_ldiv_B!(C::Cholesky, B::StridedVecOrMat) = C.uplo=='L' ? Ac_ldiv_B!(Triangular(C.UL,C.uplo,'N'), A_ldiv_B!(Triangular(C.UL,C.uplo,'N'), B)) : A_ldiv_B!(Triangular(C.UL,C.uplo,'N'), Ac_ldiv_B!(Triangular(C.UL,C.uplo,'N'), B))

function A_ldiv_B!{T<:BlasFloat}(C::CholeskyPivoted{T}, B::StridedVector{T})
    chkfullrank(C)
    ipermute!(LAPACK.potrs!(C.uplo, C.UL, permute!(B, C.piv)), C.piv)
end
function A_ldiv_B!{T<:BlasFloat}(C::CholeskyPivoted{T}, B::StridedMatrix{T})
    chkfullrank(C)
    n = size(C, 1)
    for i=1:size(B, 2)
        permute!(sub(B, 1:n, i), C.piv)
    end
    LAPACK.potrs!(C.uplo, C.UL, B)
    for i=1:size(B, 2)
        ipermute!(sub(B, 1:n, i), C.piv)
    end
    B
end
A_ldiv_B!(C::CholeskyPivoted, B::StridedVector) = C.uplo=='L' ? Ac_ldiv_B!(Triangular(C.UL,C.uplo,'N'), A_ldiv_B!(Triangular(C.UL,C.uplo,'N'), B[C.piv]))[invperm(C.piv)] : A_ldiv_B!(Triangular(C.UL,C.uplo,'N'), Ac_ldiv_B!(Triangular(C.UL,C.uplo,'N'), B[C.piv]))[invperm(C.piv)]
A_ldiv_B!(C::CholeskyPivoted, B::StridedMatrix) = C.uplo=='L' ? Ac_ldiv_B!(Triangular(C.UL,C.uplo,'N'), A_ldiv_B!(Triangular(C.UL,C.uplo,'N'), B[C.piv,:]))[invperm(C.piv),:] : A_ldiv_B!(Triangular(C.UL,C.uplo,'N'), Ac_ldiv_B!(Triangular(C.UL,C.uplo,'N'), B[C.piv,:]))[invperm(C.piv),:]

function det{T}(C::Cholesky{T})
    dd = one(T)
    for i in 1:size(C.UL,1) dd *= abs2(C.UL[i,i]) end
    dd
end

det{T}(C::CholeskyPivoted{T}) = C.rank<size(C.UL,1) ? real(zero(T)) : prod(abs2(diag(C.UL)))

function logdet{T}(C::Cholesky{T})
    dd = zero(T)
    for i in 1:size(C.UL,1) dd += log(C.UL[i,i]) end
    dd + dd # instead of 2.0dd which can change the type
end

inv(C::Cholesky) = copytri!(LAPACK.potri!(C.uplo, copy(C.UL)), C.uplo, true)

function inv(C::CholeskyPivoted)
    chkfullrank(C)
    ipiv = invperm(C.piv)
    copytri!(LAPACK.potri!(C.uplo, copy(C.UL)), C.uplo, true)[ipiv, ipiv]
end

chkfullrank(C::CholeskyPivoted) = C.rank<size(C.UL, 1) && throw(RankDeficientException(C.info))

rank(C::CholeskyPivoted) = C.rank
