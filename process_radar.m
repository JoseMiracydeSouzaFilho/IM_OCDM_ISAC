function [range_est, velocity_est] = process_radar(radar_signal, lfm_template, Fs, pulse_width)
    % Inputs:
    %   radar_signal   : Received radar signal (time domain)
    %   lfm_template   : Transmitted LFM chirp template
    %   Fs             : Sampling frequency (Hz)
    %   pulse_width    : Duration of the LFM pulse (seconds)
    %
    % Outputs:
    %   range_est      : Estimated target range (meters)
    %   velocity_est   : Estimated target velocity (m/s)

    %% 1. Matched Filtering for Range Estimation
    matched_filter = conj(flip(lfm_template)); % Reverse conjugate for correlation
    mf_output = conv(radar_signal, matched_filter, 'same');
    
    % Find peak index and compute time delay
    [~, peak_idx] = max(abs(mf_output));
    time_delay = (peak_idx / Fs); % Delay in seconds
    
    % Range calculation (using pulse_width for valid detection window)
    c = 3e8; % Speed of light
    range_est = (time_delay * c) / 2; % Round-trip time
    
    % Validate range within pulse_width limits
    max_detectable_range = (pulse_width * c) / 2;
    if range_est > max_detectable_range
        range_est = mod(range_est, max_detectable_range); % Handle ambiguity
    end

    %% 2. Doppler Processing for Velocity Estimation (Placeholder)
    % Requires multiple pulses and carrier frequency (not implemented here)
    velocity_est = 0; % Replace with actual Doppler processing
    
    % Example stub for future implementation:
    % PRF = 1 / PRP; % Pulse Repetition Frequency
    % Doppler_shift = ... % FFT-based phase change across pulses
    % velocity_est = (Doppler_shift * c) / (2 * radar_center_freq);
end