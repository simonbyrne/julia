abstract AbstractTranspose{T} <: AbstractMatrix{T}

immutable Transpose{T, S <: AbstractMatrix{T}} <: AbstractTranspose{T}
    data::S
end
immutable ConjTranspose{T, S <: AbstractMatrix{T}} <: AbstractTranspose{T}
    data::S
end

abstract AbstractCovector{T}

immutable Covector{T, S <: AbstractVector{T}} <: AbstractCovector{T}
    data::S
end
immutable ConjCovector{T, S <: AbstractVector{T}} <: AbstractCovector{T}
    data::S
end


typealias TransposeMatrix{T} Transpose{T,Matrix{T}}
typealias ConjTransposeMatrix{T} ConjTranspose{T,Matrix{T}}

typealias CovectorVector{T} Covector{T,Vector{T}}
typealias ConjCovectorVector{T} ConjCovector{T,Vector{T}}

typealias TransposeStridedMatrix{T,S<:StridedMatrix} Transpose{T,S}
typealias ConjTransposeStridedMatrix{T,S<:StridedMatrix} ConjTranspose{T,S}
typealias NTCStridedMatrix{T} Union(StridedMatrix{T},TransposeStridedMatrix{T},ConjTransposeStridedMatrix{T})

typealias NTC{T,S} Union(S,Transpose{T,S},ConjTranspose{T,S})
typealias HermTranspose{T<:Real,S} Union(Transpose{T,S},ConjTranspose{Complex{T},S})

typealias CovectorStrided{T,S<:StridedVector} Covector{T,S}
typealias ConjCovectorStrided{T,S<:StridedVector} ConjCovector{T,S}
#typealias TCStridedVector{T} Union(CovectorStrided{T},ConjCovectorStrided{T})

blastrans(A::AbstractMatrix) = 'N'
blastrans(A::Transpose) = 'T'
blastrans(A::ConjTranspose) = 'C'

blasdata(A::AbstractMatrix) = A
blasdata(A::Transpose) = A.data
blasdata(A::ConjTranspose) = A.data


# methods called by ' and .'
# should really have different names,
# ('){T<:Complex}(A::AbstractMatrix{T}) = ConjTranspose{T, typeof(A)}(A) 
ctranspose{T<:Complex}(A::AbstractMatrix{T}) = ConjTranspose{T, typeof(A)}(A)
ctranspose{T}(A::AbstractMatrix{T}) = Transpose{T, typeof(A)}(A)
transpose{T}(A::AbstractMatrix{T}) = Transpose{T, typeof(A)}(A)

ctranspose(A::ConjTranspose) = A.data
ctranspose(A::Transpose) = A.data
ctranspose{T<:Complex}(A::Transpose{T}) = error("Cannot alternate Transpose and ConjTranspose")
transpose(A::ConjTranspose) = error("Cannot alternate Transpose and ConjTranspose")

ctranspose{T<:Complex}(A::AbstractVector{T}) = ConjCovector{T, typeof(A)}(A)
ctranspose{T}(A::AbstractVector{T}) = Covector{T, typeof(A)}(A)
transpose{T}(A::AbstractVector{T}) = Covector{T, typeof(A)}(A)

ctranspose(A::ConjCovector) = A.data
ctranspose(A::Covector) = A.data
transpose(A::Covector) = A.data
ctranspose{T<:Complex}(A::Covector{T}) = error("Cannot alternate Covector and ConjCovector")
transpose(A::ConjCovector) = error("Cannot alternate Covector and ConjCovector")



size(A::AbstractTranspose) = reverse(size(A.data))
size(A::AbstractTranspose, dim::Integer) = dim == 1 ? size(A.data, 2) : (dim == 2 ? size(A.data, 1) : size(A.data, dim))

length(A::AbstractTranspose) = length(A.data)
length(A::AbstractCovector) = length(A.data)

getindex(A::Covector, i) = getindex(A.data, i).'
getindex(A::ConjCovector, i) = getindex(A.data, i)'

getindex(A::Transpose, i, j) = getindex(A.data, j, i).'
getindex(A::ConjTranspose, i, j) = getindex(A.data, j, i)'

# keep linear indexing the same?
# getindex(A::Transpose, i::Integer) = getindex(A.data, i).'
# getindex(A::ConjTranspose, i::Integer) = getindex(A.data, i)'

# for BLAS calls
convert{T}(::Type{Ptr{T}},A::AbstractTranspose{T}) = convert(Ptr{T},A.data)

# technically this is incorrect, but more useful.
stride(A::AbstractTranspose,i::Integer) = stride(A.data,i)

# extenstion of `is`
istranspose(A::AbstractMatrix, B::AbstractMatrix) = false
istranspose(A::Transpose, B::Transpose) = false
istranspose(A::AbstractMatrix, B::Transpose) = is(A,B.data)
istranspose(A::Transpose, B::AbstractMatrix) = is(A.data,B)


isctranspose(A::AbstractMatrix, B::AbstractMatrix) = false
isctranspose(A::ConjTranspose, B::ConjTranspose) = false
isctranspose(A::AbstractMatrix, B::ConjTranspose) = is(A,B.data)
isctranspose(A::ConjTranspose, B::AbstractMatrix) = is(A.data,B)


## Transpose ##

const sqrthalfcache = 1<<7
function transpose!{T<:Number}(B::Matrix{T}, A::Matrix{T})
    m, n = size(A)
    if size(B) != (n,m)
        error("input and output must have same size")
    end
    elsz = isbits(T) ? sizeof(T) : sizeof(Ptr)
    blocksize = ifloor(sqrthalfcache/elsz/1.4) # /1.4 to avoid complete fill of cache
    if m*n <= 4*blocksize*blocksize
        # For small sizes, use a simple linear-indexing algorithm
        for i2 = 1:n
            j = i2
            offset = (j-1)*m
            for i = offset+1:offset+m
                B[j] = A[i]
                j += n
            end
        end
        return B
    end
    # For larger sizes, use a cache-friendly algorithm
    for outer2 = 1:blocksize:size(A, 2)
        for outer1 = 1:blocksize:size(A, 1)
            for inner2 = outer2:min(n,outer2+blocksize)
                i = (inner2-1)*m + outer1
                j = inner2 + (outer1-1)*n
                for inner1 = outer1:min(m,outer1+blocksize)
                    B[j] = A[i]
                    i += 1
                    j += n
                end
            end
        end
    end
    B
end

function full{T, S<:DenseMatrix}(A::Transpose{T,S})
   	B = similar(A, size(A, 2), size(A, 1))
   	transpose!(B, A)
end
function full{T, S<:DenseMatrix}(A::ConjTranspose{T,S})
   	B = similar(A, size(A, 2), size(A, 1))
   	transpose!(B, A)
   	return conj!(B)
end

full(X::Transpose) = [ X[i,j] for i=1:size(X,1), j=1:size(X,2) ]
full(X::ConjTranspose) = [ conj(X[i,j]) for i=1:size(X,1), j=1:size(X,2) ]

# Covector ?

