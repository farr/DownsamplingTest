module DownsamplingTest

using CairoMakie
using DataFrames
using FFTW
using LaTeXStrings
using PairPlots
using Printf
using Turing
using Zygote

export do_plot

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

function do_plot(; amplitude=2.0, f0=2.0, tau=2.0, T=10.0, dt=0.01, sigma=1.0)
    ts = collect(-T:dt:T)

    a = amplitude*randn()
    b = amplitude*randn()

    hh = h(ts, f0, tau, a, b)

    do_plot(ts, hh, amplitude=amplutude, f0=f0, tau=tau, T=T, dt=dt, sigma=sigma)
end

function do_plot(ts, hh; amplitude=2.0, f0=2.0, tau=2.0, T=10.0, dt=0.01, sigma=1.0)
    fs = 1/dt
    fny = fs/2

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

@model function ds_model(ts, data, sigma; amplitude=2.0, f0=2.0, tau0=2.0)
    a ~ Normal(0, amplitude)
    b ~ Normal(0, amplitude)

    log_f ~ Uniform(log(f0/2), log(2*f0))
    log_tau ~ Uniform(log(tau0/2), log(2*tau0))

    f = exp(log_f)
    tau = exp(log_tau)

    hh = h(ts, f, tau, a, b)
    data .~ Normal.(hh, sigma)

    (f=f, tau=tau, h=hh)
end

function do_sampling(; amplitude=2.0, f0=2.0, tau=2.0, T=10.0, dt=0.01, sigma=1.0)
    ts = collect(-T:dt:T)
    fs = 1/dt

    a = amplitude*randn()
    b = amplitude*randn()

    hh = h(ts, f0, tau, a, b)

    do_sampling(ts, hh; amplitude=amplitude, f0=f0, tau=tau, T=T, dt=dt, sigma=sigma)
end

function do_sampling(ts, hh; amplitude=2.0, f0=2.0, tau=2.0, T=10.0, dt=0.01, sigma=1.0)
    dfs = []
    for ds in [1, 2, 4, 8, 16]
        hd = downsample(hh, ds)
        td = ts[1:ds:end]

        data = hd[td.>=0]

        model = ds_model(td[td.>=0], data, sigma, amplitude=amplitude, f0=f0, tau0=tau)
        chain = sample(model, NUTS(), 1000)
        genq = generated_quantities(model, chain)

        df = DataFrame(chain)
        df[:, :f] = vec([x.f for x in genq])
        df[:, :tau] = vec([x.tau for x in genq])
        df[:, :ds] .= ds

        push!(dfs, df)
    end
    vcat(dfs...)
end

T = 10.0
dt = 0.01
ts = collect(-T:dt:T)

amplitude = 2.0
f0 = 2.0
tau0 = 2.0

a,b = amplitude*randn(2)
hh = h(ts, f0, tau0, a, b)
_, f = do_plot(ts, hh)
f

df = do_sampling(ts, hh)
pairplot([PairPlots.Series(d[:,[:a,:b,:f,:tau]], label=string(d[1,:ds]), color=c, strokecolor=c) for (d,c) in zip(groupby(df, :ds), Makie.wong_colors(0.5))]..., PairPlots.Truth((a=a, b=b, f=f0, tau=tau0), label="Truth", color=:black))

end # module DownsamplingTest
