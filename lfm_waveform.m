function lfm_sig = lfm_waveform(N, Fs, BW, center_freq)
    T = N / Fs; % Duration matches pulse width (64μs)
    t = linspace(0, T, N).';
    
    % Chirp rate calculation
    k_r = BW / T; % 1e6 / 64e-6 = 15.625e9 Hz/s
    
    % Baseband LFM with Hamming window
    lfm_sig = hamming(N) .* exp(1j*pi*k_r*t.^2);
    
    % Upconvert to 1.57 GHz carrier
    lfm_sig = lfm_sig .* exp(1j*2*pi*center_freq*t);
    
    % Add target effects (optional)
    % delay = 2*target_range/3e8; % Round-trip delay (24μs for 3.6km)
    % doppler_shift = 2*target_speed*center_freq/3e8; % ~418.7 Hz
    % (Implement using circshift and frequency shift if needed)
end