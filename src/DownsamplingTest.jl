module DownsamplingTest

using CairoMakie
using CSV
using DataFrames
using FFTW
using LaTeXStrings
using Optim
using PairPlots
using Printf
using ProgressLogging
using Turing
using Zygote

function h(t, f, tau, a, b)
    return exp.(.-abs.(t)./tau) .* (a .* cos.(2π.*f.*t) .+ b .* sin.(2π.*f.*t))
end

function snr_td(t, h, sigma)
    return sqrt(sum(h[t .>= 0].*h[t .>= 0]))/sigma
end

function snr_fd(hf, sigma, df, fny)
    sqrt(real(4*sum(hf .* conj(hf)) / (sigma^2/fny) * df))
end

function relative_log_likelihood(t, d, h, sigma)
    r = d .- h 
    mask = t .>= 0

    ll = -0.5 * sum(r[mask] .* r[mask]) / (sigma*sigma) + 0.5 * sum(d[mask] .* d[mask]) / (sigma*sigma)
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

function fourier_downsample(h, nd=2)
    nh = length(h)
    hf = rfft(h)

    nhd = div(nh, nd)
    nfd = div(nhd, 2) + 1

    irfft(hf[1:nfd], nhd)/nd
end

function downsample(h, nd=2)
    hh = zero_pad(taper(h, 0.1))
    hf = rfft(hh)

    n = length(hf)
    n2 = div((n-1), 2)
    hf[n2+1:end] .= 0.0 + 0.0im

    return irfft(hf, length(hh))[1:nd:length(h)]
end

function downsample_times(t, nd=2)
    t[1:nd:end]
end

function find_differences_between_noise_and_signal_terms()
    ll_diffs = []
    @progress for i in 1:1000
        a,b = 2.0*randn(2)

        f0 = 2.0
        tau = 2.0
        T = 10.0
        dt = 0.01
        sigma = 1.0

        ts = collect(-T:dt:T)
        hh = h(ts, f0, tau, a, b)

        d0 = hh
        dd = hh + sigma*randn(length(hh))

        d0_down = downsample(d0, 2)
        dd_down = downsample(dd, 2)
        ts_down = downsample_times(ts, 2)
        hh_down = h(ts_down, f0, tau, a, b)

        ll0 = relative_log_likelihood(ts, d0, hh, sigma)
        lld = relative_log_likelihood(ts, dd, hh, sigma)
        
        ll0_down = relative_log_likelihood(ts_down, d0_down, hh_down, sigma/sqrt(2))
        lld_down = relative_log_likelihood(ts_down, dd_down, hh_down, sigma/sqrt(2))

        ll_diff_0 = ll0 - ll0_down
        ll_diff = lld - lld_down

        push!(ll_diffs, (ll0=ll0, lld=lld, ll0_down=ll0_down, lld_down=lld_down, ll_diff_0=ll_diff_0, ll_diff=ll_diff))
    end
    DataFrame(ll_diffs)
end

function do_plot(; amplitude=2.0, f0=2.0, tau=2.0, T=10.0, dt=0.01, sigma=1.0)
    ts = collect(-T:dt:T)

    a = amplitude*randn()
    b = amplitude*randn()

    hh = h(ts, f0, tau, a, b)

    do_plot(ts, hh; sigma=sigma)
end

function do_plot(ts, hh; sigma=1.0)
    dt = ts[2]-ts[1]
    fs = 1/dt
    fny = fs/2

    f_h = Figure()
    a_h = Axis(f_h[1,1], xlabel=L"t", ylabel=L"h(t)")
    lines!(a_h, ts, hh)

    hf = dt .* rfft(hh[ts .>= 0])
    fs = rfftfreq(length(hh[ts .>= 0]), fs)

    snr = snr_td(ts, hh, sigma)

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

function do_sampling(; amplitude=2.0, f0=2.0, tau0=2.0, T=10.0, dt=0.01, sigma=1.0, ds=[1,2,4,8,16])
    ts = collect(-T:dt:T)
    fs = 1/dt

    a = amplitude*randn()
    b = amplitude*randn()

    hh = h(ts, f0, tau, a, b)

    do_sampling(ts, hh; amplitude=amplitude, f0=f0, tau0=tau, T=T, dt=dt, sigma=sigma, ds=ds)
end

function do_sampling(ts, hh; amplitude=2.0, f0=2.0, tau0=2.0, T=10.0, dt=0.01, sigma=1.0, ds=[1,2,4,8,16])
    dfs = []
    for ds in ds
        hd = downsample(hh, ds)
        td = ts[1:ds:end]

        data = hd[td.>=0]

        model = ds_model(td[td.>=0], data, sigma/sqrt(ds), amplitude=amplitude, f0=f0, tau0=tau0)
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

function random_params(; amplitude=2.0, f0=2.0, tau0=2.0, T=10.0, dt=0.01, sigma=1.0)
    a = amplitude*randn()
    b = amplitude*randn()

    (a=a, b=b, f=f0, tau=tau)
end

function safe_and_unsafe_downsampling(ts, hh, sigma; safety_factor = 1.0)
    dt = ts[2]-ts[1]
    fsamp = 1/dt
    fny = fsamp/2

    S_h = sigma^2/fny

    sel = ts .> 0
    hpos = hh[sel]
    fs = rfftfreq(sum(sel), fsamp)
    hf = dt .* rfft(hpos)
    
    snr_numer = real.(4 .* fs .* hf .* conj(hf))

    if snr_numer[end] > S_h / safety_factor
        return (1,1) # There is no good downsampling
    end

    ds_safe = 1
    ds_unsafe = 2
    while snr_numer[div(length(snr_numer), ds_unsafe)] < S_h / safety_factor
        ds_safe = ds_unsafe
        ds_unsafe = ds_unsafe * 2
    end

    return (ds_safe, ds_unsafe)
end

function playground() 
    amplitude = 4.0
    f0 = 1.0
    tau0 = 2.0

    a,b = amplitude*randn(2)

    factor = 1.0
    sigma = 1.0 * sqrt(factor)
    dt = 0.01 / factor

    T = 10.0
    ts = collect(-T:dt:T)

    h0 = h(ts, f0, tau0, a, b)

    data = h0 .+ sigma*randn(length(h0))

    (ds_safe, ds_unsafe) = safe_and_unsafe_downsampling(ts, h0, sigma)

    f_h, f = do_plot(ts, h0, dt=dt, sigma=sigma)
    f
    f_h

    for d in [1, 2, 4, 8, 16]
        hd = fourier_downsample(data, d)
        td = ts[1:d:end]

        h0d = fourier_downsample(h0, d)
        x0 = [log(f0), log(tau0), a, b]
        sel = td .> 0

        ll = relative_log_likelihood(td[sel], hd[sel], h0d[sel], sigma/sqrt(d))

        ll_func = x -> -relative_log_likelihood(td[sel], hd[sel], h(td, exp(x[1]), exp(x[2]), x[3], x[4])[sel], sigma/sqrt(d))
        opt_result = optimize(ll_func, x0, inplace=false) 
        logf_opt, logtau_opt, a_opt, b_opt = Optim.minimizer(opt_result)
        ll_opt = -Optim.minimum(opt_result)

        println(@sprintf("ds = %d: LL = %.1f \t LL_opt = %.1f", d, ll, ll_opt))
    end

    df = do_sampling(ts, data, ds=[1, 2, 4]; f0=f0, tau0=tau0, sigma=sigma)
    pairplot(PairPlots.Truth((a=a, b=b, f=f0, tau=tau0), label="Truth", color=:black), [PairPlots.Series(d[:,[:a,:b,:f,:tau]], label=string(d[1,:ds]), color=c, strokecolor=c) for (d,c) in zip(groupby(df, :ds), Makie.wong_colors(0.5))]...)

    # bad_waveform = DataFrame(Dict(:t => ts, :h => hh))
    # CSV.write("bad_waveform.csv", bad_waveform)
end

end # module DownsamplingTest
