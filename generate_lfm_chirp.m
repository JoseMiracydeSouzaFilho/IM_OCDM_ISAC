function lfm_signal = generate_lfm_chirp(BW, pulse_width, Fs, Nc_radar)
    % Inputs:
    %   BW = Bandwidth (1e6 Hz)
    %   pulse_width = Pulse duration (64e-6 s)
    %   Fs = Sampling frequency (2e6 Hz)
    %   Nc_radar = Number of radar subchirps (128)
    
    t = linspace(0, pulse_width, Nc_radar);
    lfm_signal = chirp(t, -BW/2, pulse_width, BW/2, 'linear', 90);
    lfm_signal = lfm_signal(:); % Ensure column vector
end