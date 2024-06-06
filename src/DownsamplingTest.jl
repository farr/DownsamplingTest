module DownsamplingTest

using CairoMakie
using FFTW
using LaTeXStrings
using Printf

function h(t, f, tau, a, b)
    return exp.(.-abs.(t)./tau) .* (a .* cos.(2π.*f.*t) .+ b .* sin.(2π.*f.*t))
end

function snr_td(t, h, sigma)
    return sqrt(sum(h[t .>= 0].*h[t .>= 0]))/sigma
end

function snr_fd(hf, sigma, df, fny)
    sqrt(real(4*sum(hf .* conj(hf)) / (sigma^2/fny) * df))
end

function reflect(h)
    vcat(h[end:-1:2], h)
end

function taper(h, alpha)
    n = length(h)
    i = Int(round(n*alpha/2))

    w = ones(n)
    w[1:i] = 0.5 .* (1 .- cos.(2π.*(0:i-1)./(2*i)))
    w[end-i+1:end] = w[i:-1:1]

    return h .* w
end

function next_pow_2(n)
    np = 1
    while np < n
        np = np << 1
    end
    np
end

function zero_pad(h)
    n = next_pow_2(length(h))

    return vcat(h, zeros(n - length(h)))
end

function downsample(h, nd=2)
    hh = zero_pad(taper(h, 0.1))
    hf = rfft(hh)

    n = length(hf)
    n2 = div((n-1), 2)
    hf[n2+1:end] .= 0.0 + 0.0im

    return irfft(hf, length(hh))[1:nd:length(h)]
end

function do_plot()
    sigma = 1.0
    
    ts = collect(-10:0.01:10)
    T = ts[end]
    dt = ts[2]-ts[1]
    fs = 1/dt
    fny = fs/2

    f0 = 2.0
    tau = 2.0

    a = 2*randn()
    b = 2*randn()

    hh = h(ts, f0, tau, a, b)

    f_h = Figure()
    a_h = Axis(f_h[1,1], xlabel=L"t", ylabel=L"h(t)")
    lines!(a_h, ts, hh)

    hf = dt .* rfft(hh[ts .>= 0])
    fs = rfftfreq(length(hh[ts .>= 0]), fs)

    snr = snr_td(ts[ts.>=0], hh[ts.>=0], sigma)

    f = Figure()
    a = Axis(f[1,1], xscale=log10, yscale=log10,
             xlabel=L"f", ylabel=L"$4 f \left| h \right|^2$ or $S_h(f)$")

    lines!(a, fs[2:end], real(4 .* fs[2:end] .* hf[2:end] .* conj(hf[2:end])))
    hlines!(a, [sigma^2/fny], color=:black)
    vlines!(a, [fny, fny/2, fny/4, fny/8, fny/16], color=:black)

    ypos = maximum(real(hf .* conj(hf)))
    xpos = 0.9*fny
    rotation = pi/2
    text!(a, xpos, ypos, text=@sprintf("SNR^2 = %.1f", snr^2), rotation=rotation, align=(:right, :baseline))

    println(@sprintf("SNR^2 = %.1f", snr^2))
    for ds in [2, 4, 8, 16]
        hd = downsample(hh, ds)
        td = ts[1:ds:end]

        snr_d = snr_td(td[td.>=0], hd[td.>=0], sigma/sqrt(ds))

        println(@sprintf("ds = %d: SNR^2 = %.1f", ds, snr_d^2))
        xpos = 0.9*fny/ds
        text!(a, xpos, ypos, text=@sprintf("SNR^2 = %.1f", snr_d^2), rotation=rotation, align=(:right, :baseline))
    end

    f_h, f
end

end # module DownsamplingTest
