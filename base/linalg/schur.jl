immutable Schur{Ty<:BlasFloat} <: Factorization{Ty}
    T::Matrix{Ty}
    Z::Matrix{Ty}
    values::Vector
end

schurfact!{T<:BlasFloat}(A::StridedMatrix{T}) = Schur(LinAlg.LAPACK.gees!('V', A)...)
schurfact{T<:BlasFloat}(A::StridedMatrix{T}) = schurfact!(copy(A))
schurfact{T}(A::StridedMatrix{T}) = (S = promote_type(Float32,typeof(one(T)/norm(one(T)))); S != T ? schurfact!(convert(AbstractMatrix{S},A)) : schurfact!(copy(A)))

function getindex(F::Schur, d::Symbol)
    (d == :T || d == :Schur) && return F.T
    (d == :Z || d == :vectors) && return F.Z
    d == :values && return F.values
    throw(KeyError(d))
end

function schur(A::AbstractMatrix)
    SchurF = schurfact(A)
    SchurF[:T], SchurF[:Z], SchurF[:values]
end

immutable GeneralizedSchur{Ty<:BlasFloat} <: Factorization{Ty}
    S::Matrix{Ty}
    T::Matrix{Ty}
    alpha::Vector
    beta::Vector{Ty}
    Q::Matrix{Ty}
    Z::Matrix{Ty}
end

schurfact!{T<:BlasFloat}(A::StridedMatrix{T}, B::StridedMatrix{T}) = GeneralizedSchur(LinAlg.LAPACK.gges!('V', 'V', A, B)...)
schurfact{T<:BlasFloat}(A::StridedMatrix{T},B::StridedMatrix{T}) = schurfact!(copy(A),copy(B))
schurfact{TA,TB}(A::StridedMatrix{TA}, B::StridedMatrix{TB}) = (S = promote_type(Float32,typeof(one(TA)/norm(one(TA))),TB); schurfact!(S != TA ? convert(AbstractMatrix{S},A) : copy(A), S != TB ? convert(AbstractMatrix{S},B) : copy(B)))

function getindex(F::GeneralizedSchur, d::Symbol)
    d == :S && return F.S
    d == :T && return F.T
    d == :alpha && return F.alpha
    d == :beta && return F.beta
    d == :values && return F.alpha./F.beta
    (d == :Q || d == :left) && return F.Q
    (d == :Z || d == :right) && return F.Z
    throw(KeyError(d))
end

function schur(A::AbstractMatrix, B::AbstractMatrix)
    SchurF = schurfact(A, B)
    SchurF[:S], SchurF[:T], SchurF[:Q], SchurF[:Z]
end
