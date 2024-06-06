# DownsamplingTest

Comparing time-domain and frequency-domain S/N of damped sinusoids at various
sample rates.

Installation: run `julia` within the root directory, and enter package mode (type `]`), and issue
```julia
activate .
instantiate
```

Usage: to produce plots like the one below, run `julia` in the root directory, activate the package (`] activate .`), and run 
```julia
using DownsamplingTest
figure_h_of_t, figure_frequency_domain_snr = do_plot()
```

The particular ring-up-ring-down sinusoid will be chosen with random phase and
amplitude.  You can see it plotted in the time-domain in the first figure, and
then the frequency-domain approximate S/N integrand compared to the white noise
spectral density in the second figure.  The second figure will include vertical
lines at Nyquist and downsampling by factors of 2, 4, ..., with the computed
time-domain S/N for each downsampling factor indicated.  The function will also
print the computed time-domain S/N for each downsampling factor.