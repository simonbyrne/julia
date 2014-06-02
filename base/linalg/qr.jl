####################
# QR Factorization #
####################

immutable QR{T} <: Factorization{T}
    factors::Matrix{T}
    τ::Vector{T}
end
# Note. For QRCompactWY factorization without pivoting, the WY representation based method introduced in LAPACK 3.4
immutable QRCompactWY{S} <: Factorization{S}
    factors::Matrix{S}
    T::Matrix{S}
end

immutable QRPivoted{T} <: Factorization{T}
    factors::Matrix{T}
    τ::Vector{T}
    jpvt::Vector{BlasInt}
end

qrfact!{T<:BlasFloat}(A::StridedMatrix{T}; pivot=false) = pivot ? QRPivoted{T}(LAPACK.geqp3!(A)...) : QRCompactWY(LAPACK.geqrt!(A, min(minimum(size(A)), 36))...)
function qrfact!{T}(A::AbstractMatrix{T}; pivot=false)
    pivot && warn("pivoting only implemented for Float32, Float64, Complex64 and Complex128")
    m, n = size(A)
    τ = zeros(T, min(m,n))
    @inbounds begin
        for k = 1:min(m-1+!(T<:Real),n)
            τk = elementaryLeft!(A, k, k)
            τ[k] = τk
            for j = k+1:n
                vAj = A[k,j]
                for i = k+1:m
                    vAj += conj(A[i,k])*A[i,j]
                end
                vAj = conj(τk)*vAj
                A[k,j] -= vAj
                for i = k+1:m
                    A[i,j] -= A[i,k]*vAj
                end
            end
        end
    end
    QR(A, τ)
end
qrfact{T<:BlasFloat}(A::StridedMatrix{T}; pivot=false) = qrfact!(copy(A),pivot=pivot)
qrfact{T}(A::StridedMatrix{T}; pivot=false) = (S = typeof(one(T)/norm(one(T)));S != T ? qrfact!(convert(AbstractMatrix{S},A), pivot=pivot) : qrfact!(copy(A),pivot=pivot))
qrfact(x::Number) = qrfact(fill(x,1,1))

function qr(A::Union(Number, AbstractMatrix); pivot=false, thin::Bool=true)
    F = qrfact(A, pivot=pivot)
    full(F[:Q], thin=thin), F[:R]
end

convert{T}(::Type{QR{T}},A::QR) = QR(convert(AbstractMatrix{T}, A.factors), convert(Vector{T}, A.τ))
convert{T}(::Type{Factorization{T}}, A::QR) = convert(QR{T}, A)
convert{T}(::Type{QRCompactWY{T}},A::QRCompactWY) = QRCompactWY(convert(AbstractMatrix{T}, A.factors), convert(AbstractMatrix{T}, A.T))
convert{T}(::Type{Factorization{T}}, A::QRCompactWY) = convert(QRCompactWY{T}, A)
convert{T}(::Type{QRPivoted{T}},A::QRPivoted) = QRPivoted(convert(AbstractMatrix{T}, A.factors), convert(Vector{T}, A.τ), A.jpvt)
convert{T}(::Type{Factorization{T}}, A::QRPivoted) = convert(QRPivoted{T}, A)

function getindex(A::QR, d::Symbol)
    d == :R && return triu(A.factors[1:minimum(size(A)),:])
    d == :Q && return QRPackedQ(A.factors,A.τ)
    throw(KeyError(d))
end
function getindex(A::QRCompactWY, d::Symbol)
    d == :R && return triu(A.factors[1:minimum(size(A)),:])
    d == :Q && return QRCompactWYQ(A.factors,A.T)
    throw(KeyError(d))
end
function getindex{T}(A::QRPivoted{T}, d::Symbol)
    d == :R && return triu(A.factors[1:minimum(size(A)),:])
    d == :Q && return QRPackedQ(A.factors,A.τ)
    d == :p && return A.jpvt
    if d == :P
        p = A[:p]
        n = length(p)
        P = zeros(T, n, n)
        for i in 1:n
            P[p[i],i] = one(T)
        end
        return P
    end
    throw(KeyError(d))
end

immutable QRPackedQ{T} <: AbstractMatrix{T}
    factors::Matrix{T}
    τ::Vector{T}
end
immutable QRCompactWYQ{S} <: AbstractMatrix{S} 
    factors::Matrix{S}                      
    T::Matrix{S}                       
end

convert{T}(::Type{QRPackedQ{T}}, Q::QRPackedQ) = QRPackedQ(convert(AbstractMatrix{T}, Q.factors), convert(Vector{T}, Q.τ))
convert{T}(::Type{AbstractMatrix{T}}, Q::QRPackedQ) = convert(QRPackedQ{T}, Q)
convert{S}(::Type{QRCompactWYQ{S}}, Q::QRCompactWYQ) = QRCompactWYQ(convert(AbstractMatrix{S}, Q.factors), convert(AbstractMatrix{S}, Q.T))
convert{S}(::Type{AbstractMatrix{S}}, Q::QRCompactWYQ) = convert(QRCompactWYQ{S}, Q)

size(A::Union(QR,QRCompactWY,QRPivoted), dim::Integer) = size(A.factors, dim)
size(A::Union(QR,QRCompactWY,QRPivoted)) = size(A.factors)
size(A::Union(QRPackedQ,QRCompactWYQ), dim::Integer) = 0 < dim ? (dim <= 2 ? size(A.factors, 1) : 1) : throw(BoundsError())
size(A::Union(QRPackedQ,QRCompactWYQ)) = size(A, 1), size(A, 2)

full{T}(A::Union(QRPackedQ{T},QRCompactWYQ{T}); thin::Bool=true) = mul!(Inplace(2), A, thin ? eye(T, size(A.factors,1), minimum(size(A.factors))) : eye(T, size(A.factors,1)))



## Multiplication by Q

function mul!(C::AbstractMatrix, A::QRPackedQ, B::AbstractMatrix)
    copy!(C,B)
    mul!(Inplace(2),A,C)
end
function mul!(C::AbstractMatrix, A::QRCompactWYQ, B::AbstractMatrix)
    copy!(C,B)
    mul!(Inplace(2),A,C)
end

### QB
for (QRQ, lpkfn) in
    ((:QRCompactWYQ,:gemqrt!),
     (:QRPackedQ,:ormqr!))
    for (QRT,BlasType) in 
        ((:($QRQ{T}),:BlasFloat),
         (:(Transpose{T,$QRQ{T}}),:BlasReal),
         (:(ConjTranspose{T,$QRQ{T}}),:BlasComplex))
        @eval begin
            function mul!{T<:$BlasType}(::Inplace{2}, A::$QRT, B::StridedVecOrMat{T})
                Q = blasdata(A)
                (LAPACK.$lpkfn)('L', blastrans(A), Q.factors, Q.T, B)
            end
            function mul!{T<:$BlasType}(::Inplace{1}, A::StridedMatrix{T}, B::$QRT)
                Q = blasdata(B)
                (LAPACK.$lpkfn)('R', blastrans(B), Q.factors, Q.T, A)
            end
        end
    end
    @eval begin
        mul!(::Inplace{2},A::$QRQ, B::Transpose) = mul!(Inplace(1), B.', A.').'
        mul!(::Inplace{2},A::$QRQ, B::ConjTranspose) = mul!(Inplace(1), B', A')'
    end
end


function mul!(::Inplace{2}, A::QRPackedQ, B::AbstractVecOrMat)
    mA, nA = size(A.factors)
    mB, nB = size(B,1), size(B,2)
    mA == mB || throw(DimensionMismatch("")) 
    @inbounds begin
        for k = min(mA,nA):-1:1
            for j = 1:nB
                vBj = B[k,j]
                for i = k+1:mB
                    vBj += conj(A.factors[i,k])*B[i,j]
                end
                vBj = A.τ[k]*vBj
                B[k,j] -= vBj
                for i = k+1:mB
                    B[i,j] -= A.factors[i,k]*vBj
                end
            end
        end
    end
    B
end
function mul!{T<:Real}(::Inplace{2}, A::HermTranspose{T,QRPackedQ}, B::AbstractVecOrMat)
    Q = A.data
    mA, nA = size(Q.factors)
    mB, nB = size(B,1), size(B,2)
    mA == mB || throw(DimensionMismatch(""))
    @inbounds begin
        for k = 1:min(mA,nA)
            for j = 1:nB
                vBj = B[k,j]
                for i = k+1:mB
                    vBj += conj(Q.factors[i,k])*B[i,j]
                end
                vBj = conj(Q.τ[k])*vBj
                B[k,j] -= vBj
                for i = k+1:mB
                    B[i,j] -= Q.factors[i,k]*vBj
                end
            end
        end
    end
    B
end

function mul!(::Inplace{1}, A::AbstractMatrix, Q::QRPackedQ)
    mQ, nQ = size(Q.factors)
    mA, nA = size(A,1), size(A,2)
    nA == mQ || throw(DimensionMismatch(""))
    @inbounds begin
        for k = 1:min(mQ,nQ)
            for i = 1:mA
                vAi = A[i,k]
                for j = k+1:mQ
                    vAi += A[i,j]*Q.factors[j,k]
                end
                vAi = vAi*Q.τ[k]
                A[i,k] -= vAi
                for j = k+1:nA
                    A[i,j] -= vAi*conj(Q.factors[j,k])
                end
            end
        end
    end
    A
end

function mul!{T<:Real}(::Inplace{1}, A::AbstractMatrix, B::HermTranspose{T,QRPackedQ})
    Q = B.data
    mQ, nQ = size(Q.factors)
    mA, nA = size(A,1), size(A,2)
    nA == mQ || throw(DimensionMismatch(""))
    @inbounds begin
        for k = min(mQ,nQ):-1:1
            for i = 1:mA
                vAi = A[i,k]
                for j = k+1:mQ
                    vAi += A[i,j]*Q.factors[j,k]
                end
                vAi = vAi*conj(Q.τ[k])
                A[i,k] -= vAi
                for j = k+1:nA
                    A[i,j] -= vAi*conj(Q.factors[j,k])
                end
            end
        end
    end
    A
end




# Julia implementation similarly to xgelsy
function A_ldiv_B!{T<:BlasFloat}(A::Union(QRCompactWY{T},QRPivoted{T}), B::StridedMatrix{T}, rcond::Real)
    mA, nA = size(A.factors)
    nr = min(mA,nA)
    nrhs = size(B, 2)
    if nr == 0 return zeros(0, nrhs), 0 end
    ar = abs(A.factors[1])
    if ar == 0 return zeros(nr, nrhs), 0 end
    rnk = 1
    xmin = ones(T, nr)
    xmax = ones(T, nr)
    tmin = tmax = ar
    while rnk < nr
        tmin, smin, cmin = LAPACK.laic1!(2, sub(xmin, 1:rnk), tmin, sub(A.factors, 1:rnk, rnk + 1), A.factors[rnk + 1, rnk + 1])
        tmax, smax, cmax = LAPACK.laic1!(1, sub(xmax, 1:rnk), tmax, sub(A.factors, 1:rnk, rnk + 1), A.factors[rnk + 1, rnk + 1])
        tmax*rcond > tmin && break
        xmin[1:rnk + 1] = [smin*sub(xmin, 1:rnk), cmin]
        xmax[1:rnk + 1] = [smax*sub(xmin, 1:rnk), cmax]
        rnk += 1
        # if cond(r[1:rnk, 1:rnk])*rcond < 1 break end
    end
    C, τ = LAPACK.tzrzf!(A.factors[1:rnk,:])
    A_ldiv_B!(Triangular(C[1:rnk,1:rnk],:U),sub(mul!(Inplace(2),A[:Q]',sub(B, 1:mA, 1:nrhs)),1:rnk,1:nrhs))
    B[rnk+1:end,:] = zero(T)
    LAPACK.ormrz!('L', iseltype(B, Complex) ? 'C' : 'T', C, τ, sub(B,1:nA,1:nrhs))
    return isa(A,QRPivoted) ? B[invperm(A[:p]),:] : B[1:nA,:], rnk
end
A_ldiv_B!{T<:BlasFloat}(A::Union(QRCompactWY{T},QRPivoted{T}), B::StridedVector{T}) = A_ldiv_B!(A,reshape(B,length(B),1))[:]
A_ldiv_B!{T<:BlasFloat}(A::Union(QRCompactWY{T},QRPivoted{T}), B::StridedVecOrMat{T}) = A_ldiv_B!(A, B, sqrt(eps(real(float(one(eltype(B)))))))[1]
function A_ldiv_B!{T}(A::QR{T},B::StridedMatrix{T})
    m, n = size(A)
    minmn = min(m,n)
    mB, nB = size(B)
    mul!(Inplace(2),A[:Q]',sub(B,1:m,1:nB)) # Reconsider when arrayviews are merged.
    R = A[:R]
    @inbounds begin
        if n > m # minimum norm solution
            τ = zeros(T,m)
            for k = m:-1:1 # Trapezoid to triangular by elementary operation
                τ[k] = elementaryRightTrapezoid!(R,k)
                for i = 1:k-1
                    vRi = R[i,k]
                    for j = m+1:n
                        vRi += R[i,j]*R[k,j]
                    end
                    vRi *= τ[k]
                    R[i,k] -= vRi
                    for j = m+1:n
                        R[i,j] -= vRi*R[k,j]
                    end
                end
            end
        end
        for k = 1:nB # solve triangular system. When array views are implemented, consider exporting    to function.
            for i = minmn:-1:1
                for j = i+1:minmn
                    B[i,k] -= R[i,j]*B[j,k]
                end
                B[i,k] /= R[i,i]
            end
        end
        if n > m # Apply elemenary transformation to solution
            B[m+1:mB,1:nB] = zero(T)
            for j = 1:nB
                for k = 1:m
                    vBj = B[k,j]
                    for i = m+1:n
                        vBj += B[i,j]*conj(R[k,i])
                    end
                    vBj *= τ[k]
                    B[k,j] -= vBj
                    for i = m+1:n
                        B[i,j] -= R[k,i]*vBj
                    end
                end
            end
        end
    end
    return B[1:n,:]
end
A_ldiv_B!(A::QR, B::StridedVector) = A_ldiv_B!(A, reshape(B, length(B), 1))[:]
A_ldiv_B!(A::QRPivoted, B::StridedVector) = A_ldiv_B!(QR(A.factors,A.τ),B)[invperm(A.jpvt)]
A_ldiv_B!(A::QRPivoted, B::StridedMatrix) = A_ldiv_B!(QR(A.factors,A.τ),B)[invperm(A.jpvt),:]
function \{TA,Tb}(A::Union(QR{TA},QRCompactWY{TA},QRPivoted{TA}),b::StridedVector{Tb})
    S = promote_type(TA,Tb)
    m,n = size(A)
    m == length(b) || throw(DimensionMismatch("left hand side has $(m) rows, but right hand side has length $(length(b))"))
    n > m ? A_ldiv_B!(convert(Factorization{S},A),[b,zeros(S,n-m)]) : A_ldiv_B!(convert(Factorization{S},A), S == Tb ? copy(b) : convert(AbstractVector{S}, b))
end
function \{TA,TB}(A::Union(QR{TA},QRCompactWY{TA},QRPivoted{TA}),B::StridedMatrix{TB})
    S = promote_type(TA,TB)
    m,n = size(A)
    m == size(B,1) || throw(DimensionMismatch("left hand side has $(m) rows, but right hand side has $(size(B,1)) rows"))
    n > m ? A_ldiv_B!(convert(Factorization{S},A),[B;zeros(S,n-m,size(B,2))]) : A_ldiv_B!(convert(Factorization{S},A), S == TB ? copy(B) : convert(AbstractMatrix{S}, B))
end

##TODO:  Add methods for rank(A::QRP{T}) and adjust the (\) method accordingly
##       Add rcond methods for Cholesky, LU, QR and QRP types
## Lower priority: Add LQ, QL and RQ factorizations

# FIXME! Should add balancing option through xgebal
immutable Hessenberg{T} <: Factorization{T}
    factors::Matrix{T}
    τ::Vector{T}
end
Hessenberg(A::StridedMatrix) = Hessenberg(LAPACK.gehrd!(A)...)

hessfact!{T<:BlasFloat}(A::StridedMatrix{T}) = Hessenberg(A)
hessfact{T<:BlasFloat}(A::StridedMatrix{T}) = hessfact!(copy(A))
hessfact{T}(A::StridedMatrix{T}) = (S = promote_type(Float32,typeof(one(T)/norm(one(T)))); S != T ? hessfact!(convert(AbstractMatrix{S},A)) : hessfact!(copy(A)))

immutable HessenbergQ{T} <: AbstractMatrix{T}
    factors::Matrix{T}
    τ::Vector{T}
end
HessenbergQ(A::Hessenberg) = HessenbergQ(A.factors, A.τ)
size(A::HessenbergQ, args...) = size(A.factors, args...)

function getindex(A::Hessenberg, d::Symbol)
    d == :Q && return HessenbergQ(A)
    d == :H && return triu(A.factors, -1)
    throw(KeyError(d))
end

full(A::HessenbergQ) = LAPACK.orghr!(1, size(A.factors, 1), copy(A.factors), A.τ)

# Also printing of QRQs
print_matrix(io::IO, A::Union(QRPackedQ,QRCompactWYQ,HessenbergQ), rows::Integer, cols::Integer, punct...) = print_matrix(io, full(A), rows, cols, punct...)
