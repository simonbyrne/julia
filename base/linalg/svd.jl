# SVD
immutable SVD{T<:BlasFloat,Tr} <: Factorization{T}
    U::Matrix{T}
    S::Vector{Tr}
    Vt::Matrix{T}
end
function svdfact!{T<:BlasFloat}(A::StridedMatrix{T}; thin::Bool=true)
    m,n = size(A)
    if m == 0 || n == 0
        u,s,vt = (eye(T, m, thin ? n : m), real(zeros(T,0)), eye(T,n,n))
    else
        u,s,vt = LAPACK.gesdd!(thin ? 'S' : 'A', A)
    end
    SVD(u,s,vt)
end
svdfact{T<:BlasFloat}(A::StridedMatrix{T};thin=true) = svdfact!(copy(A),thin=thin)
svdfact{T}(A::StridedVecOrMat{T};thin=true) = (S = promote_type(Float32,typeof(one(T)/norm(one(T)))); S != T ? svdfact!(convert(AbstractMatrix{S},A),thin=thin) : svdfact!(copy(A),thin=thin))
svdfact(x::Number; thin::Bool=true) = SVD(x == 0 ? fill(one(x), 1, 1) : fill(x/abs(x), 1, 1), [abs(x)], fill(one(x), 1, 1))
svdfact(x::Integer; thin::Bool=true) = svdfact(float(x), thin=thin)

function svd(A::Union(Number, AbstractArray); thin::Bool=true)
    F = svdfact(A, thin=thin)
    F.U, F.S, F.Vt'
end

function getindex(F::SVD, d::Symbol)
    d == :U && return F.U
    d == :S && return F.S
    d == :Vt && return F.Vt
    d == :V && return F.Vt'
    throw(KeyError(d))
end

svdvals!{T<:BlasFloat}(A::StridedMatrix{T}) = any([size(A)...].==0) ? zeros(T, 0) : LAPACK.gesdd!('N', A)[2]
svdvals{T<:BlasFloat}(A::StridedMatrix{T}) = svdvals!(copy(A))
svdvals{T}(A::StridedMatrix{T}) = (S = promote_type(Float32,typeof(one(T)/norm(one(T)))); S != T ? svdvals!(convert(AbstractMatrix{S}, A)) : svdvals!(copy(A)))
svdvals(x::Number) = [abs(x)]

# SVD least squares
function \{T<:BlasFloat}(A::SVD{T}, B::StridedVecOrMat{T})
    n = length(A.S)
    Sinv = zeros(T, n)
    Sinv[A.S .> sqrt(eps())] = 1.0 ./ A.S
    scale(A.Vt', Sinv) * A.U[:,1:n]'B
end

# Generalized svd
immutable GeneralizedSVD{T} <: Factorization{T}
    U::Matrix{T}
    V::Matrix{T}
    Q::Matrix{T}
    a::Vector
    b::Vector
    k::Int
    l::Int
    R::Matrix{T}
end

function svdfact!{T<:BlasFloat}(A::StridedMatrix{T}, B::StridedMatrix{T})
    U, V, Q, a, b, k, l, R = LAPACK.ggsvd!('U', 'V', 'Q', A, B)
    GeneralizedSVD(U, V, Q, a, b, int(k), int(l), R)
end
svdfact{T<:BlasFloat}(A::StridedMatrix{T}, B::StridedMatrix{T}) = svdfact!(copy(A),copy(B))
svdfact{TA,TB}(A::StridedMatrix{TA}, B::StridedMatrix{TB}) = (S = promote_type(Float32,typeof(one(TA)/norm(one(TA))),TB); svdfact!(S != TA ? convert(AbstractMatrix{S},A) : copy(A), S != TB ? convert(AbstractMatrix{S},B) : copy(B)))

function svd(A::AbstractMatrix, B::AbstractMatrix)
    F = svdfact(A, B)
    F[:U], F[:V], F[:Q], F[:D1], F[:D2], F[:R0]
end

function getindex{T}(obj::GeneralizedSVD{T}, d::Symbol)
    d == :U && return obj.U
    d == :V && return obj.V
    d == :Q && return obj.Q
    (d == :alpha || d == :a) && return obj.a
    (d == :beta || d == :b) && return obj.b
    (d == :vals || d == :S) && return obj.a[1:obj.k + obj.l] ./ obj.b[1:obj.k + obj.l]
    if d == :D1
        m = size(obj.U, 1)
        if m - obj.k - obj.l >= 0
            return [eye(T, obj.k) zeros(T, obj.k, obj.l); zeros(T, obj.l, obj.k) diagm(obj.a[obj.k + 1:obj.k + obj.l]); zeros(T, m - obj.k - obj.l, obj.k + obj.l)]
        else
            return [eye(T, m, obj.k) [zeros(T, obj.k, m - obj.k); diagm(obj.a[obj.k + 1:m])] zeros(T, m, obj.k + obj.l - m)]
        end
    end
    if d == :D2
        m = size(obj.U, 1)
        p = size(obj.V, 1)
        if m - obj.k - obj.l >= 0
            return [zeros(T, obj.l, obj.k) diagm(obj.b[obj.k + 1:obj.k + obj.l]); zeros(T, p - obj.l, obj.k + obj.l)]
        else
            return [zeros(T, p, obj.k) [diagm(obj.b[obj.k + 1:m]); zeros(T, obj.k + p - m, m - obj.k)] [zeros(T, m - obj.k, obj.k + obj.l - m); eye(T, obj.k + p - m, obj.k + obj.l - m)]]
        end
    end
    d == :R && return obj.R
    if d == :R0
        n = size(obj.Q, 1)
        return [zeros(T, obj.k + obj.l, n - obj.k - obj.l) obj.R]
    end
    throw(KeyError(d))
end

function svdvals!{T<:BlasFloat}(A::StridedMatrix{T}, B::StridedMatrix{T})
    _, _, _, a, b, k, l, _ = LAPACK.ggsvd!('N', 'N', 'N', A, B)
    a[1:k + l] ./ b[1:k + l]
end
svdvals{T<:BlasFloat}(A::StridedMatrix{T},B::StridedMatrix{T}) = svdvals!(copy(A),copy(B))
svdvals{TA,TB}(A::StridedMatrix{TA}, B::StridedMatrix{TB}) = (S = promote_type(Float32,typeof(one(T)/norm(one(TA))),TB); svdvals!(S != TA ? convert(AbstractMatrix{S}, A) : copy(A), S != TB ? convert(AbstractMatrix{S}, B) : copy(B)))
