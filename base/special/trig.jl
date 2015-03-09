immutable DoubleFloat64
    hi::Float64
    lo::Float64
end
immutable DoubleFloat32
    hi::Float64
end

# kernel_* functions are only valid for |x| < pi/4 = 0.7854
# translated from openlibm code: k_sin.c, k_cos.c, k_sinf.c, k_cosf.c
# rem_pio2 is based on e_rem_pio2.c
# which are made available under the following licence:

## Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
##
## Developed at SunPro, a Sun Microsystems, Inc. business.
## Permission to use, copy, modify, and distribute this
## software is freely granted, provided that this notice
## is preserved.

function sin_kernel(x::DoubleFloat64)
    S1 = -1.66666666666666324348e-01
    S2 =  8.33333333332248946124e-03
    S3 = -1.98412698298579493134e-04
    S4 =  2.75573137070700676789e-06
    S5 = -2.50507602534068634195e-08
    S6 =  1.58969099521155010221e-10

    z = x.hi*x.hi
    w = z*z
    r = S2+z*(S3+z*S4) + z*w*(S5+z*S6)
    v = z*x.hi
    x.hi-((z*(0.5*x.lo-v*r)-x.lo)-v*S1)
end

function cos_kernel(x::DoubleFloat64)
    C1 =  4.16666666666666019037e-02
    C2 = -1.38888888888741095749e-03
    C3 =  2.48015872894767294178e-05
    C4 = -2.75573143513906633035e-07
    C5 =  2.08757232129817482790e-09
    C6 = -1.13596475577881948265e-11

    z = x.hi*x.hi
    w = z*z
    r = z*(C1+z*(C2+z*C3)) + w*w*(C4+z*(C5+z*C6))
    hz = 0.5*z
    w = 1.0-hz
    w + (((1.0-w)-hz) + (z*r-x.hi*x.lo))
end

function sin_kernel(x::DoubleFloat32)
    S1 = -0.16666666641626524
    S2 = 0.008333329385889463
    S3 = -0.00019839334836096632
    S4 = 2.718311493989822e-6

    z = x.hi*x.hi
    w = z*z
    r = S3+z*S4
    s = z*x.hi
    Float32((x.hi + s*(S1+z*S2)) + s*w*r)
end

function cos_kernel(x::DoubleFloat32)
    C0 = -0.499999997251031
    C1 = 0.04166662332373906
    C2 = -0.001388676377460993
    C3 = 2.439044879627741e-5

    z = x.hi*x.hi
    w = z*z
    r = C2+z*C3
    Float32(((1.0+z*C0) + w*C1) + (w*z)*r)
end

# fallback methods
sin_kernel(x::Real) = sin(x)
cos_kernel(x::Real) = cos(x)

# multiply in extended precision
function mulpi_ext(x::Float64)
    m = 3.141592653589793
    m_hi = 3.1415926218032837
    m_lo = 3.178650954705639e-8

    u = 134217729.0*x # 0x1p27 + 1
    x_hi = u-(u-x)
    x_lo = x-x_hi

    y_hi = m*x
    y_lo = x_hi * m_lo + (x_lo* m_hi + ((x_hi*m_hi-y_hi) + x_lo*m_lo))

    DoubleFloat64(y_hi,y_lo)
end
mulpi_ext(x::Float32) = DoubleFloat32(pi*Float64(x))
mulpi_ext(x::Rational) = mulpi_ext(float(x))
mulpi_ext(x::Real) = pi*x # Fallback

function sinpi(x::Real)
    if isinf(x)
        return throw(DomainError())
    elseif isnan(x)
        return oftype(x,NaN)
    end

    rx = copysign(rem(x,2),x)
    arx = abs(rx)

    if rx == zero(rx)
        return copysign(float(zero(rx)),x)
    elseif arx < oftype(rx,0.25)
        return sin_kernel(mulpi_ext(rx))
    elseif arx <= oftype(rx,0.75)
        y = mulpi_ext(oftype(rx,0.5) - arx)
        return copysign(cos_kernel(y),rx)
    elseif arx == one(x)
        return copysign(float(zero(rx)),rx)
    elseif arx < oftype(rx,1.25)
        y = mulpi_ext((one(rx) - arx)*sign(rx))
        return sin_kernel(y)
    elseif arx <= oftype(rx,1.75)
        y = mulpi_ext(oftype(rx,1.5) - arx)
        return -copysign(cos_kernel(y),rx)
    else
        y = mulpi_ext(rx - copysign(oftype(rx,2.0),rx))
        return sin_kernel(y)
    end
end

function cospi(x::Real)
    if isinf(x)
        return throw(DomainError())
    elseif isnan(x)
        return oftype(x,NaN)
    end

    rx = abs(float(rem(x,2)))

    if rx <= oftype(rx,0.25)
        return cos_kernel(mulpi_ext(rx))
    elseif rx < oftype(rx,0.75)
        y = mulpi_ext(oftype(rx,0.5) - rx)
        return sin_kernel(y)
    elseif rx <= oftype(rx,1.25)
        y = mulpi_ext(one(rx) - rx)
        return -cos_kernel(y)
    elseif rx < oftype(rx,1.75)
        y = mulpi_ext(rx - oftype(rx,1.5))
        return sin_kernel(y)
    else
        y = mulpi_ext(oftype(rx,2.0) - rx)
        return cos_kernel(y)
    end
end

sinpi(x::Integer) = x >= 0 ? zero(float(x)) : -zero(float(x))
cospi(x::Integer) = isodd(x) ? -one(float(x)) : one(float(x))

function sinpi(z::Complex)
    zr, zi = reim(z)
    if !isfinite(zi) && zr == 0 return complex(zr, zi) end
    if isnan(zr) && !isfinite(zi) return complex(zr, zi) end
    if !isfinite(zr) && zi == 0 return complex(oftype(zr, NaN), zi) end
    if !isfinite(zr) && isfinite(zi) return complex(oftype(zr, NaN), oftype(zi, NaN)) end
    if !isfinite(zr) && !isfinite(zi) return complex(zr, oftype(zi, NaN)) end
    pizi = pi*zi
    complex(sinpi(zr)*cosh(pizi), cospi(zr)*sinh(pizi))
end

function cospi(z::Complex)
    zr, zi = reim(z)
    if !isfinite(zi) && zr == 0
        return complex(isnan(zi) ? zi : oftype(zi, Inf),
                       isnan(zi) ? zr : zr*-sign(zi))
    end
    if !isfinite(zr) && isinf(zi)
        return complex(oftype(zr, Inf), oftype(zi, NaN))
    end
    if isinf(zr)
        return complex(oftype(zr, NaN), zi==0 ? -copysign(zi, zr) : oftype(zi, NaN))
    end
    if isnan(zr) && zi==0 return complex(zr, abs(zi)) end
    pizi = pi*zi
    complex(cospi(zr)*cosh(pizi), -sinpi(zr)*sinh(pizi))
end
@vectorize_1arg Number sinpi
@vectorize_1arg Number cospi


sinc(x::Number) = x==0 ? one(x)  : oftype(x,sinpi(x)/(pi*x))
sinc(x::Integer) = x==0 ? one(x) : zero(x)
sinc{T<:Integer}(x::Complex{T}) = sinc(float(x))
@vectorize_1arg Number sinc
cosc(x::Number) = x==0 ? zero(x) : oftype(x,(cospi(x)-sinpi(x)/(pi*x))/x)
cosc(x::Integer) = cosc(float(x))
cosc{T<:Integer}(x::Complex{T}) = cosc(float(x))
@vectorize_1arg Number cosc

for (finv, f) in ((:sec, :cos), (:csc, :sin), (:cot, :tan),
                  (:sech, :cosh), (:csch, :sinh), (:coth, :tanh),
                  (:secd, :cosd), (:cscd, :sind), (:cotd, :tand))
    @eval begin
        ($finv){T<:Number}(z::T) = one(T) / (($f)(z))
        ($finv){T<:Number}(z::AbstractArray{T}) = one(T) ./ (($f)(z))
    end
end

for (fa, fainv) in ((:asec, :acos), (:acsc, :asin), (:acot, :atan),
                    (:asech, :acosh), (:acsch, :asinh), (:acoth, :atanh))
    @eval begin
        ($fa){T<:Number}(y::T) = ($fainv)(one(T) / y)
        ($fa){T<:Number}(y::AbstractArray{T}) = ($fainv)(one(T) ./ y)
    end
end


# multiply in extended precision
function deg2rad_ext(x::Float64)
    m = 0.017453292519943295
    m_hi = 0.01745329238474369
    m_lo = 1.3519960527851425e-10

    u = 134217729.0*x # 0x1p27 + 1
    x_hi = u-(u-x)
    x_lo = x-x_hi

    y_hi = m*x
    y_lo = x_hi * m_lo + (x_lo* m_hi + ((x_hi*m_hi-y_hi) + x_lo*m_lo))

    DoubleFloat64(y_hi,y_lo)
end
deg2rad_ext(x::Float32) = DoubleFloat32(deg2rad(Float64(x)))
deg2rad_ext(x::Real) = deg2rad(x) # Fallback

function sind(x::Real)
    if isinf(x)
        return throw(DomainError())
    elseif isnan(x)
        return oftype(x,NaN)
    end

    rx = copysign(float(rem(x,360)),x)
    arx = abs(rx)

    if rx == zero(rx)
        return rx
    elseif arx < oftype(rx,45)
        return sin_kernel(deg2rad_ext(rx))
    elseif arx <= oftype(rx,135)
        y = deg2rad_ext(oftype(rx,90) - arx)
        return copysign(cos_kernel(y),rx)
    elseif arx == oftype(rx,180)
        return copysign(zero(rx),rx)
    elseif arx < oftype(rx,225)
        y = deg2rad_ext((oftype(rx,180) - arx)*sign(rx))
        return sin_kernel(y)
    elseif arx <= oftype(rx,315)
        y = deg2rad_ext(oftype(rx,270) - arx)
        return -copysign(cos_kernel(y),rx)
    else
        y = deg2rad_ext(rx - copysign(oftype(rx,360),rx))
        return sin_kernel(y)
    end
end
@vectorize_1arg Real sind

function cosd(x::Real)
    if isinf(x)
        return throw(DomainError())
    elseif isnan(x)
        return oftype(x,NaN)
    end

    rx = abs(float(rem(x,360)))

    if rx <= oftype(rx,45)
        return cos_kernel(deg2rad_ext(rx))
    elseif rx < oftype(rx,135)
        y = deg2rad_ext(oftype(rx,90) - rx)
        return sin_kernel(y)
    elseif rx <= oftype(rx,225)
        y = deg2rad_ext(oftype(rx,180) - rx)
        return -cos_kernel(y)
    elseif rx < oftype(rx,315)
        y = deg2rad_ext(rx - oftype(rx,270))
        return sin_kernel(y)
    else
        y = deg2rad_ext(oftype(rx,360) - rx)
        return cos_kernel(y)
    end
end
@vectorize_1arg Real cosd

tand(x::Real) = sind(x) / cosd(x)
@vectorize_1arg Real tand

for (fd, f) in ((:sind, :sin), (:cosd, :cos), (:tand, :tan))
    @eval begin
        ($fd)(z) = ($f)(deg2rad(z))
    end
end

for (fd, f) in ((:asind, :asin), (:acosd, :acos), (:atand, :atan),
                (:asecd, :asec), (:acscd, :acsc), (:acotd, :acot))
    @eval begin
        ($fd)(y) = rad2deg(($f)(y))
        @vectorize_1arg Real $fd
    end
end


function rem_pio2(x::Float64)
    two24 =  1.67772160000000000000e+07   # 0x4170_0000_0000_0000
    invpio2 =  6.36619772367581382433e-01 # 0x3FE4_5F30_6DC9_C883
    pio2_1  =  1.57079632673412561417e+00 # 0x3FF9_21FB_5440_0000
    pio2_1t =  6.07710050650619224932e-11 # 0x3DD0_B461_1A62_6331
    pio2_2  =  6.07710050630396597660e-11 # 0x3DD0_B461_1A60_0000
    pio2_2t =  2.02226624879595063154e-21 # 0x3BA3_198A_2E03_7073
    pio2_3  =  2.02226624871116645580e-21 # 0x3BA3_198A_2E00_0000
    pio2_3t =  8.47842766036889956997e-32 # 0x397B_839A_2520_49C1

    ux = reinterpret(UInt64,x)
    hx = (ux >> 32) % UInt
    ix = hx & 0x7fff_ffff
    
    if ix <= 0x3fe9_21fb # |x| ~<= pi/4
        return 0, DoubleFloat64(x, 0.0)
    elseif ix <= 0x400f_6a7a # |x| ~<= 5pi/4
        if ((ix & 0xf_ffff) == 0x9_21fb) # |x| ~= pi/2, pi/4 
            @goto medium
        elseif ix <= 0x4002_d97c # |x| ~<= 3pi/4
            if hx > 0
                z = x - pio2_1
                y_hi = z - pio2_1t
                y_lo = (z - y_hi) - pio2_1t
                return 1, DoubleFloat64(y_hi, y_lo)
            else
                z = x + pio2_1
                y_hi = z + pio2_1t
                y_lo = (z - y_hi) + pio2_1t
                return -1, DoubleFloat64(y_hi, y_lo)
            end
        else # 3pi/4 ~<= |x| ~<= 5pi/4
            if hx > 0
                z = x - 2pio2_1
                y_hi = z - 2pio2_1t
                y_lo = (z - y_hi) - 2pio2_1t
                return 2, DoubleFloat64(y_hi,y_lo)
            else
                z = x + 2pio2_1
                y_hi = z + 2pio2_1t
                y_lo = (z - y_hi) + 2pio2_1t
                return -2, DoubleFloat64(y_hi,y_lo)
            end
        end
    elseif ix <= 0x401c_463b # |x| ~<= 9pi/4
        if ix <= 0x4015_fdbc # |x| ~<= 7pi/4
            if ix == 0x4012_d97c # |x| ~= 3pi/2
                @goto medium
            elseif hx > 0
                z = x - 3pio2_1
                y_hi = z - 3pio2_1t
                y_lo = (z - y_hi) - 3pio2_1t
                return 3, DoubleFloat64(y_hi,y_lo)
            else
                z = x + 3pio2_1
                y_hi = z + 3pio2_1t
                y_lo = (z - y_hi) + 3pio2_1t
                return -3, DoubleFloat64(y_hi,y_lo)
            end
        else # 7pi/4 ~<= |x| ~<= 9pi/4
            if ix == 0x4019_21fb # |x| ~= 4pi/2
                @goto medium
            elseif hx > 0
                z = x - 4pio2_1
                y_hi = z - 4pio2_1t
                y_lo = (z - y_hi) - 4pio2_1t
                return 4, DoubleFloat64(y_hi,y_lo)
            else
                z = x + 4pio2_1
                y_hi = z + 4pio2_1t
                y_lo = (z - y_hi) + 4pio2_1t
                return -4, DoubleFloat64(y_hi,y_lo)
            end
        end
    end

    if ix < 0x413921fb # |x| ~< 2^20*(pi/2)
        @label medium

        fn = round(x*invpio2) # assumes fast, ties-to-even
        n = unsafe_trunc(Int,fn)
        r = x-fn*pio2_1
        w = fn*pio2_1t

        j = ix >> 20 # biased exponent of x
        y_hi = r-w
        k = ((reinterpret(UInt64,y_hi) >> 52) % UInt) & 0x7ff # biased exponent of y_hi
        i = j - k
        if i > 16 # require 2nd iteration
            t = r
            w = fn*pio2_2
            r = t-w
            w = fn*pio2_2t - ((t-r)-w)
            y_hi = r-w
            k = UInt(reinterpret(UInt64,y_hi) >> 52) & 0x7ff # biased exponent of y_hi
            i = j - k
            if i > 49 # require 3rd iteration
                t = r
                w = fn*pio2_3
                r = t-w
                w = fn*pio2_3t - ((t-r)-w)
                y_hi = r-w
            end
        end
        y_lo = (r-y_hi)-w
        return n, DoubleFloat64(y_hi, y_lo)

    elseif ix >= 0x7ff00000 # Inf or NaN
        y = x-x
        return 0, DoubleFloat64(y,y)

    else # slow method for large exponents (|x| > 1e6 or so)
        # TODO: implement Payne-Hanek in Julia
        y = [0.0,0.0]
        n = ccall((:__ieee754_rem_pio2, openspecfun), Cint, (Float64,Ptr{Float64}), x, y)            
        return n, DoubleFloat64(y[1],y[2])
    end
end

function sincos(x::Float64)
    n, r = rem_pio2(x)
    n &= 3
    s = sin_kernel(r)
    c = cos_kernel(r)
    
    if n == 0
        s,c
    elseif n == 1
        c,-s
    elseif n == 2
        -s,-c
    else
        -c,s
    end
end
