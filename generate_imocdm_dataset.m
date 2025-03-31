function [X_cplx, Y_labels] = generate_imocdm_dataset(num_samples, num_classes, Phi_H, Fs)
    % System parameters
    Nc = 256;                   % Total subchirps
    Ns_comm = 128;              % Communication subchirps
    Ns_radar = 128;             % Radar subchirps
    radar_BW = 1e6;             % 1 MHz bandwidth
    radar_center_freq = 0; % 1.57 GHz center frequency
    fixed_noise_power = -20;    % -20 dB fixed noise power

    % Generate IM patterns
    patterns = generate_im_patterns(Ns_comm, num_classes);
    
    % Initialize output arrays
    X_cplx = complex(zeros(Nc, num_samples));
    Y_labels_numeric = zeros(1, num_samples);

    % Configure Rayleigh channel
    channel = comm.RayleighChannel(...
        'SampleRate', Fs, ...
        'PathDelays', [0, 0.5e-6], ...
        'AveragePathGains', [0, -3], ...
        'MaximumDopplerShift', 50, ...
        'RandomStream', 'Global stream');

    parfor i = 1:num_samples
        % 1. Generate Index-Modulated Comm Signal
        class_id = randi(num_classes);
        comm_signal = zeros(Ns_comm, 1);
        active_idx = patterns{class_id};
        
        % 16-QAM Modulation
        comm_signal(active_idx) = qammod(...
            randi([0 15], length(active_idx), 1), ...
            16, 'UnitAveragePower', true);

        % 2. Generate Radar Signal (LFM chirp)
        radar_signal = lfm_waveform(Ns_radar, Fs, radar_BW, radar_center_freq);

        % 3. Composite Signal Generation
        tx_symbols = [comm_signal; radar_signal];
        tx_signal = Phi_H * tx_symbols;  % IDFnT

        % 4. Channel Effects
        rx_signal = channel(tx_signal);
        
        % 5. Add Fixed -20 dB Noise
        rx_signal = awgn(rx_signal, fixed_noise_power, 'measured');
        
        % 6. Normalize and Store
        X_cplx(:, i) = rx_signal / max(abs(rx_signal));
        Y_labels_numeric(i) = class_id;
    end
    
    % Convert to categorical labels
    Y_labels = categorical(Y_labels_numeric, 1:num_classes);
end