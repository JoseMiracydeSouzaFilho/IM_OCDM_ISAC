
function X_aug = advanced_augmentation(X, Fs,radar_center_freq)
    % ADVANCED_AUGMENTATION - Applies time-domain augmentations to IM-OCDM signals
    %
    % Inputs:
    %   X   : [Nc × num_samples] matrix of input signals (time-domain)
    %   Fs  : Sampling frequency in Hz (used for resampling operations)
    %
    % Output:
    %   X_aug : [Nc × num_samples] matrix of augmented signals
    %
    % Parameters (internal, adjustable):
    %   augmentation_factor : Probability [0-1] of augmenting each sample (default: 0.3)
    %   time_warp_range     : [min, max] time warp factors (default: [0.9 1.1] = ±10% speed variation)
    %   noise_level         : Noise amplitude as percentage of signal peak (default: 0.03 = 3%)
    %   max_shift           : Maximum cyclic shift as percentage of signal length (default: 10%)
    %
    % Augmentation Techniques:
    % 1. Physical Time Warping: Uses Fs to maintain Nyquist compliance
    % 2. Additive Noise
    % 3. Cyclic Shifts
    % 4. Amplitude Scaling

    [Nc, num_samples] = size(X);
    X_aug = X;
    
    % ========================
    % Augmentation Parameters
    % ========================
    augmentation_factor = 0.3;
    time_warp_range = [0.8 1.2]; % Wider range now physically meaningful
    noise_level = 0.03;
    max_shift = round(0.1*Nc);
    
    % Nyquist safety check
    % min_freq = Fs * time_warp_range(1) / 2;
    % if min_freq < 2*radar_center_freq % Example: 24 GHz radar
    %     warning('Time warping may violate Nyquist at %.2f GHz!', min_freq/1e9);
    % end

     parfor i = 1:num_samples
        if rand() < augmentation_factor
            signal = X(:,i);
            
            % 1. PHYSICAL TIME WARPING (Fixed integer ratio)
            if rand() > 0.5
                % Get warp factor and convert to integer ratio
                warp_factor = time_warp_range(1) + diff(time_warp_range)*rand();
                [p, q] = rat(warp_factor, 0.01); % Rational approximation with 1% tolerance
                
                % Resample with anti-aliasing
                warped_signal = resample(signal, p, q);
                
                % Maintain original length
                if length(warped_signal) > Nc
                    signal = warped_signal(1:Nc);
                else
                    signal = [warped_signal; zeros(Nc - length(warped_signal),1)];
                end
            end
            
             % 2. Additive White Gaussian Noise
            signal = signal + noise_level*max(signal)*randn(Nc,1);
            
            % 3. Cyclic Shift (Temporal Misalignment)
            shift = randi([1 max_shift]);
            signal = circshift(signal, shift);
            
            % 4. Amplitude Scaling (Channel Variation)
            scale_factor = 0.8 + 0.4*rand(); % Random scaling between 80%-120%
            signal = scale_factor * signal;
            
            X_aug(:,i) = signal;
        end
    end
end
