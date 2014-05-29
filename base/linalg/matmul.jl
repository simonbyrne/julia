# matmul.jl: Everything to do with dense matrix multiplication

arithtype(T) = T
arithtype(::Type{Bool}) = Int

# multiply by diagonal matrix as vector
function scale!(C::AbstractMatrix, A::AbstractMatrix, b::AbstractVector)
    m, n = size(A)
    n==length(b) || throw(DimensionMismatch(""))
    for j = 1:n
        bj = b[j]
        for i = 1:m
            C[i,j] = A[i,j]*bj
        end
    end
    C
end

function scale!(C::AbstractMatrix, b::AbstractVector, A::AbstractMatrix)
    m, n = size(A)
    m==length(b) || throw(DimensionMismatch(""))
    for j=1:n, i=1:m
        C[i,j] = A[i,j]*b[i]
    end
    C
end
scale(A::AbstractMatrix, b::AbstractVector) = scale!(similar(A, promote_type(eltype(A),eltype(b))), A, b)
scale(b::AbstractVector, A::AbstractMatrix) = scale!(similar(b, promote_type(eltype(A),eltype(b)), size(A)), b, A)


## Covec * Vec = scalar (dot product)

*{T<:BlasReal}(x::CovectorVector{T}, y::Vector{T}) = BLAS.dot(x.data, y)
*{T<:BlasComplex}(x::ConjCovectorVector{T}, y::Vector{T}) = BLAS.dotc(x.data, y)
*{T<:BlasComplex}(x::CovectorVector{T}, y::Vector{T}) = BLAS.dotu(x.data, y)

function *(x::AbstractCovector, y::AbstractVector)
    lx = length(x)
    lx==length(y) || throw(DimensionMismatch(""))
    if lx == 0
        return zero(eltype(x))*zero(eltype(y))
    end
    s = x[1]*y[1]
    @inbounds for i = 2:lx
        s += x[i]*y[i]
    end
    s
end


## Vec * Covec = Matrix
# TODO


## Matrix * Vec = Vec

function (*){T,S}(A::AbstractMatrix{T}, x::AbstractVector{S})
    TS = promote_type(arithtype(T),arithtype(S))
    mul!(similar(x,TS,size(A,1)),A,x)
end

for elty in (Float32,Float64)
    @eval begin
        function mul!(y::StridedVector{Complex{$elty}}, A::StridedMatrix{Complex{$elty}}, x::StridedVector{$elty})
            Afl = reinterpret($elty,A,(2size(A,1),size(A,2)))
            yfl = reinterpret($elty,y)
            mul!(yfl, Afl, x)
            return y
        end
    end
end

function mul!{T<:BlasFloat}(y::StridedVector{T}, A::NTC{T,StridedMatrix}, x::StridedVector{T})
    stride(A, 1)==1 || return invoke(mul!,(AbstractVector{T},typeof(A),AbstractVector{T}), y, A, x)

    (mA, nA) = size(A)
    nA==length(x) && mA==length(y)|| throw(DimensionMismatch(""))

    (mA == 0 || nA == 0) && return fill!(C,zero(T))

    if Base.blas_vendor() == :openblas
        ## Avoid calling BLAS.gemv! when OpenBLAS is being used until #6941 is fixed.
        invoke(mul!,(AbstractVector{T},typeof(A),AbstractVector{T}), y, A, x)
    else
        BLAS.gemv!(blastrans(A), one(T), blasdata(A), x, zero(T), y)
    end
end

function mul!{R}(y::AbstractVector{R}, A::AbstractMatrix, x::AbstractVector)
    lx = length(x)
    mA, nA = size(A)
    lx==nA && mA==length(y) || throw(DimensionMismatch("*"))

    fill!(y, zero(R))
    @inbounds for k = 1:nA
        b = x[k]
        for i = 1:mA
            y[i] += A[i,k] * b
        end
    end
    y
end

function mul!{R}(y::AbstractVector{R}, A::AbstractTranspose, x::AbstractVector)
    lx = length(x)
    mA, nA = size(A)
    lx==nA && mA==length(y) || throw(DimensionMismatch("*"))

    @inbounds for k = 1:mA
        s = zero(R)
        for i = 1:nA
            s += A[k,i] * x[i]
        end
        y[k] = s
    end
    y
end

## Covec * Matrix = Covec

*(x::Covector, A::AbstractMatrix) = (A.' * x.').'
*(x::ConjCovector, A::AbstractMatrix) = (A' * x')'


## Matrix * Matrix = Matrix

function *{T,S}(A::AbstractMatrix{T},B::AbstractMatrix{S})
    TS = promote_type(arithtype(T),arithtype(S))
    mul!(similar(B,TS,(size(A,1),size(B,2))), A, B)
end


for elty in (Float32,Float64)
    @eval begin
        function mul!(C::StridedMatrix{Complex{$elty}}, A::StridedMatrix{Complex{$elty}}, B::StridedMatrix{$elty})
            Afl = reinterpret($elty,A,(2size(A,1),size(A,2)))
            Cfl = reinterpret($elty,C,(2size(C,1),size(C,2)))
            mul!(Cfl,Afl,B)
            return C
        end
    end
end

for elty in (Float32,Float64)
    @eval begin
        function mul!(C::StridedMatrix{Complex{$elty}}, A::StridedMatrix{Complex{$elty}}, B::TransposeStridedMatrix{$elty})
            Afl = reinterpret($elty,A,(2size(A,1),size(A,2)))
            Cfl = reinterpret($elty,C,(2size(C,1),size(C,2)))
            mul!(Cfl,Afl,B.data)
            return C
        end
    end
end



# Supporting functions for matrix multiplication

function copytri!(A::StridedMatrix, uplo::Char, conjugate::Bool=false)
    n = chksquare(A)
    @chkuplo
    if uplo == 'U'
        for i = 1:(n-1), j = (i+1):n
            A[j,i] = conjugate ? conj(A[i,j]) : A[i,j]
        end
    elseif uplo == 'L'
        for i = 1:(n-1), j = (i+1):n
            A[i,j] = conjugate ? conj(A[j,i]) : A[j,i]
        end
    else
        throw(ArgumentError("second argument must be 'U' or 'L'"))
    end
    A
end

function mul!{T<:BlasFloat}(C::StridedMatrix{T}, A::NTC{T,StridedMatrix}, B::NTC{T,StridedMatrix})
    (stride(A, 1) == stride(B, 1) == 1) || return invoke(mul!,(AbstractMatrix{T},AbstractMatrix{T},AbstractMatrix{T}), C, A, B)
    
    mA, nA = size(A)
    mB, nB = size(B)
    mC, nC = size(C)

    mB == nA && mC == mA && nC == nB || throw(DimensionMismatch("*"))

    (mA == 0 || nA == 0 || nB == 0) && return fill!(C,zero(T))
    mA == 2 && nA == 2 && nB == 2 && return mul22!(C,A,B)
    mA == 3 && nA == 3 && nB == 3 && return mul33!(C,A,B)
    
    if istranspose(A,B)
        copytri!(BLAS.syrk!('U', blastrans(A), one(T), blasdata(A), zero(T), C), 'U')
    elseif isctranspose(A,B)
        copytri!(BLAS.herk!('U', blastrans(A), one(T), blasdata(A), zero(T), C), 'U', true)
    else
        BLAS.gemm!(blastrans(A), blastrans(B), one(T), blasdata(A), blasdata(B), zero(T), C)
    end
end

# blas.jl defines matmul for floats; other integer and mixed precision
# cases are handled here

function copy!(B::AbstractMatrix, ir_dest::UnitRange{Int}, jr_dest::UnitRange{Int}, M::Transpose, ir_src::UnitRange{Int}, jr_src::UnitRange{Int})
    copy_transpose!(B, ir_dest, jr_dest, M.data, jr_src, ir_src)
end
function copy!(B::AbstractMatrix, ir_dest::UnitRange{Int}, jr_dest::UnitRange{Int}, M::ConjTranspose, ir_src::UnitRange{Int}, jr_src::UnitRange{Int})
    copy_transpose!(B, ir_dest, jr_dest, M.data, jr_src, ir_src)
    conj!(B)
end

function copy_transpose!(B::AbstractMatrix, ir_dest::UnitRange{Int}, jr_dest::UnitRange{Int}, M::Transpose, ir_src::UnitRange{Int}, jr_src::UnitRange{Int})
    copy!(B, ir_dest, jr_dest, M, jr_src, ir_src)
end
function copy_transpose!(B::AbstractMatrix, ir_dest::UnitRange{Int}, jr_dest::UnitRange{Int}, M::ConjTranspose, ir_src::UnitRange{Int}, jr_src::UnitRange{Int})
    copy!(B, ir_dest, jr_dest, M, jr_src, ir_src)
    conj!(B)
end

# TODO: It will be faster for large matrices to convert to float,
# call BLAS, and convert back to required type.

# NOTE: the generic version is also called as fallback for
#       strides != 1 cases


const tilebufsize = 10800  # Approximately 32k/3
const Abuf = Array(Uint8, tilebufsize)
const Bbuf = Array(Uint8, tilebufsize)
const Cbuf = Array(Uint8, tilebufsize)

function mul!{R}(C::AbstractMatrix{R}, A::AbstractMatrix, B::AbstractMatrix)
    mA, nA = size(A)
    mB, nB = size(B)
    mC, nC = size(C)

    mB == nA && mC == mA && nC == nB || throw(DimensionMismatch("*"))

    (mA == 0 || nA == 0 || nB == 0) && return fill!(C,zero(T))
    mA == 2 && nA == 2 && nB == 2 && return mul22!(C,A,B)
    mA == 3 && nA == 3 && nB == 3 && return mul33!(C,A,B)

    @inbounds begin
    if isbits(R)
        tile_size = int(ifloor(sqrt(tilebufsize/sizeof(R))))
        sz = (tile_size, tile_size)
        Atile = pointer_to_array(convert(Ptr{R}, pointer(Abuf)), sz)
        Btile = pointer_to_array(convert(Ptr{R}, pointer(Bbuf)), sz)

        z = zero(R)

        if mA < tile_size && nA < tile_size && nB < tile_size
            copy_transpose!(Atile, 1:nA, 1:mA, A, 1:mA, 1:nA)
            copy!(Btile, 1:mB, 1:nB, B, 1:mB, 1:nB)
            for j = 1:nB
                boff = (j-1)*tile_size
                for i = 1:mA
                    aoff = (i-1)*tile_size
                    s = z
                    for k = 1:nA
                        s += Atile[aoff+k] * Btile[boff+k]
                    end
                    C[i,j] = s
                end
            end
        else
            Ctile = pointer_to_array(convert(Ptr{R}, pointer(Cbuf)), sz)
            for jb = 1:tile_size:nB
                jlim = min(jb+tile_size-1,nB)
                jlen = jlim-jb+1
                for ib = 1:tile_size:mA
                    ilim = min(ib+tile_size-1,mA)
                    ilen = ilim-ib+1
                    fill!(Ctile, z)
                    for kb = 1:tile_size:nA
                        klim = min(kb+tile_size-1,mB)
                        klen = klim-kb+1
                        copy_transpose!(Atile, 1:klen, 1:ilen, A, ib:ilim, kb:klim)
                        copy!(Btile, 1:klen, 1:jlen, B, kb:klim, jb:jlim)
                        for j=1:jlen
                            bcoff = (j-1)*tile_size
                            for i = 1:ilen
                                aoff = (i-1)*tile_size
                                s = z
                                for k = 1:klen
                                    s += Atile[aoff+k] * Btile[bcoff+k]
                                end
                                Ctile[bcoff+i] += s
                            end
                        end
                    end
                    copy!(C, ib:ilim, jb:jlim, Ctile, 1:ilen, 1:jlen)
                end
            end
        end
    else
        # Multiplication for non-plain-data uses the naive algorithm
        for i = 1:mA, j = 1:nB
            Ctmp = A[i, 1]*B[1, j]
            for k = 2:nA
                Ctmp += A[i, k]*B[k, j]
            end
            C[i,j] = Ctmp
        end
    end
    end # @inbounds
    C
end


# multiply 2x2 matrices
# function matmul2x2{T,S}(tA, tB, A::AbstractMatrix{T}, B::AbstractMatrix{S})
#     matmul2x2!(similar(B, promote_type(T,S), 2, 2), tA, tB, A, B)
# end

function mul22!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
    A11 = A[1,1]; A12 = A[1,2]; A21 = A[2,1]; A22 = A[2,2]
    B11 = B[1,1]; B12 = B[1,2]; B21 = B[2,1]; B22 = B[2,2]
    
    C[1,1] = A11*B11 + A12*B21
    C[1,2] = A11*B12 + A12*B22
    C[2,1] = A21*B11 + A22*B21
    C[2,2] = A21*B12 + A22*B22
    C
end

# # Multiply 3x3 matrices
# function matmul3x3{T,S}(tA, tB, A::AbstractMatrix{T}, B::AbstractMatrix{S})
#     matmul3x3!(similar(B, promote_type(T,S), 3, 3), tA, tB, A, B)
# end

function mul33!{T,S,R}(C::AbstractMatrix{R}, A::AbstractMatrix{T}, B::AbstractMatrix{S})
    A11 = A[1,1]; A12 = A[1,2]; A13 = A[1,3];
    A21 = A[2,1]; A22 = A[2,2]; A23 = A[2,3];
    A31 = A[3,1]; A32 = A[3,2]; A33 = A[3,3];

    B11 = B[1,1]; B12 = B[1,2]; B13 = B[1,3];
    B21 = B[2,1]; B22 = B[2,2]; B23 = B[2,3];
    B31 = B[3,1]; B32 = B[3,2]; B33 = B[3,3];

    C[1,1] = A11*B11 + A12*B21 + A13*B31
    C[1,2] = A11*B12 + A12*B22 + A13*B32
    C[1,3] = A11*B13 + A12*B23 + A13*B33

    C[2,1] = A21*B11 + A22*B21 + A23*B31
    C[2,2] = A21*B12 + A22*B22 + A23*B32
    C[2,3] = A21*B13 + A22*B23 + A23*B33

    C[3,1] = A31*B11 + A32*B21 + A33*B31
    C[3,2] = A31*B12 + A32*B22 + A33*B32
    C[3,3] = A31*B13 + A32*B23 + A33*B33

    C
end
