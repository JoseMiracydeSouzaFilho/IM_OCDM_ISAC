%% 1. Initial Configuration
clear; clc; close all;
fprintf('=== IM-OCDM with LSTM - Signal Classification System ===\n');
fprintf('Based on paper: LSTM Framework for Classification of Radar and Communications Signals\n');

%% Initial Sanity Tests
fprintf('Running sanity tests...\n');
test_signal = lfm_waveform(256, 15.6e12, 64e-6);
assert(~any(isnan(test_signal)), 'Basic signal generation contains NaN');
assert(~any(isinf(test_signal)), 'Basic signal generation contains Inf');
assert(abs(max(test_signal)) <= 1, 'LFM signal exceeds expected amplitude');

%% 2. Dataset Generation with NaN Prevention
num_train_samples = 20e3;  % 20k training samples
num_val_samples = 3e3;     % 3k validation samples
SNR_range = 0:5:40;        % SNR range (0-40 dB)
num_classes = 2;           % 1=Radar, 2=Communication

fprintf('Generating dataset with NaN checks...\n');
[X_train_cplx, Y_train_labels] = generate_imocdm_dataset(num_train_samples, SNR_range, num_classes);
[X_val_cplx, Y_val_labels] = generate_imocdm_dataset(num_val_samples, SNR_range, num_classes);

%% 2.1 Advanced Data Augmentation (Tested Version)
fprintf('Applying data augmentation...\n');
augmentation_factor = 0.3; % 30% of samples get extra noise

% Initialization
num_samples = size(X_train_cplx, 2);
augment_mask = false(1, num_samples);
snr_levels = zeros(num_samples, 1);

% Sample-by-sample application with logging
for i = 1:num_samples
    if rand() < augmentation_factor
        augment_mask(i) = true;
        snr_levels(i) = 10 + 3*randn(); % Variable SNR
        X_train_cplx(:,i) = awgn(X_train_cplx(:,i), snr_levels(i), 'measured');
    end
end

% Consistent noise for validation
X_val_cplx = awgn(X_val_cplx, 20); % Fixed high SNR

% Statistics calculation
num_augmented = sum(augment_mask);
used_snr_levels = snr_levels(augment_mask);

fprintf('=== Augmentation Statistics ===\n');
fprintf('Augmented samples: %d/%d (%.1f%%)\n',...
    num_augmented, num_samples, 100*num_augmented/num_samples);
fprintf('Mean SNR: %.1f dB (Std: %.1f dB)\n',...
    mean(used_snr_levels), std(used_snr_levels));

%% 3. Advanced Preprocessing (STFT + Normalization)
fprintf('Processing signals with quality checks...\n');

% STFT transform with error handling
X_train_stft = apply_stft_transform(X_train_cplx);
X_val_stft = apply_stft_transform(X_val_cplx);

% NaN verification after STFT
assert(~any(isnan(X_train_stft(:))), 'NaN detected after STFT (train)');
assert(~any(isnan(X_val_stft(:))), 'NaN detected after STFT (validation)');

% Concatenate real and imaginary parts
X_train = [real(X_train_stft); imag(X_train_stft)];
X_val = [real(X_val_stft); imag(X_val_stft)];

% Robust normalization (with zero fallback)
[X_train, train_fixes] = normalize_with_checks(X_train);
[X_val, val_fixes] = normalize_with_checks(X_val);
fprintf('Applied corrections: %d (train), %d (validation)\n', train_fixes, val_fixes);

% Reshape data for LSTM [1×features×samples]
X_train = reshape(X_train, [1, size(X_train,1), size(X_train,2)]);
X_val = reshape(X_val, [1, size(X_val,1), size(X_val,2)]);

% Convert labels to categorical
Y_train = categorical(Y_train_labels');
Y_val = categorical(Y_val_labels');

%% 4. Optimized LSTM Architecture
inputSize = size(X_train, 2);

layers = [
    sequenceInputLayer(numFeatures)
    
    % Conv1D + BiLSTM
    convolution1dLayer(3, 128, 'Padding', 'same', 'Stride', 1)
    batchNormalizationLayer()
    reluLayer()
    bilstmLayer(256, 'OutputMode','sequence')
    dropoutLayer(0.4)
    
    % Attention mechanism
    globalAveragePooling1dLayer()  % Optional: Reduces sequence length
    attentionLayer(128, 'Attention')  % Use custom layer
    
    % Classifier
    fullyConnectedLayer(128)
    layerNormalizationLayer()
    reluLayer()
    dropoutLayer(0.3)
    fullyConnectedLayer(num_classes)
    softmaxLayer()
    classificationLayer()
];

%% 5. Adjusted Training Options
options = trainingOptions('adam',...
    'MaxEpochs', 200,...
    'MiniBatchSize', 256,...
    'InitialLearnRate', 1e-4,...
    'LearnRateSchedule', 'piecewise',...
    'LearnRateDropPeriod', 25,...
    'LearnRateDropFactor', 0.5,...
    'GradientThreshold', 1.5,...
    'ValidationData', {X_val, Y_val},...
    'ValidationFrequency', 100,...
    'Shuffle', 'every-epoch',...
    'Plots', 'training-progress',...
    'ExecutionEnvironment', 'auto',...
    'OutputNetwork', 'best-validation-loss',...
    'Verbose', true);

%% 6. Training with Monitoring
try
    fprintf('Starting training...\n');
    [net, info] = trainnet(X_train, Y_train, layers, "crossentropy", options);
    
    % Safe model saving
    model_name = sprintf('imocdm_lstm_%s.mat', datestr(now, 'ddmmyy_HHMM'));
    save(model_name, 'net', 'info', '-v7.3');
    fprintf('Model saved as: %s\n', model_name);
    
    % Robust evaluation
    [accuracy, confMat] = evaluate_model_robust(net, X_val, Y_val);
    fprintf('\n=== Final Results ===\n');
    fprintf('Accuracy: %.2f%%\n', accuracy*100);
    disp('Confusion Matrix:');
    disp(confMat);
    
catch ME
    fprintf('Error during training: %s\n', ME.message);
    save('error_context.mat', 'X_train', 'Y_train', 'layers', '-v7.3');
end

%% ================= UPDATED SUPPORT FUNCTIONS ====================

function [X_cplx, Y_labels, SNR_stats] = generate_imocdm_dataset(num_samples, SNR_range, num_classes)
    % Parameters from IM-OCDM paper
    Ns = 256;               % Number of subcarriers
    num_active = 128;       % Active subcarriers
    k_r = 15.6e12;          % Chirp rate (Hz/s)
    Fs = 2e6;               % Sampling frequency (Hz)
    T = 64e-6;              % Symbol duration (s)
    
    % Safe initialization
    X_cplx = zeros(Ns, num_samples);
    Y_labels = zeros(1, num_samples);
    current_SNR_record = zeros(1, num_samples); % To record SNRs
    
    % Robust Rayleigh channel configuration
    channel = comm.RayleighChannel(...
        'SampleRate', Fs, ...
        'PathDelays', [0, 1e-6, 2.3e-6], ...
        'AveragePathGains', [0, -3, -6], ...
        'MaximumDopplerShift', 200, ...
        'RandomStream', 'mt19937ar with seed', ...
        'Seed', randi(10000));
    
    % Extreme SNR definitions (now used)
    extreme_low_SNRs = [min(SNR_range)-5, -3, 0, 2]; % Low values
    extreme_high_SNRs = [max(SNR_range)+5, max(SNR_range)+15, max(SNR_range)+20]; % High values
    
    for i = 1:num_samples
        try
            % 1. Signal generation with amplitude check
            Y_labels(i) = randi(num_classes);
            
            if Y_labels(i) == 1 % Radar (LFM)
                tx_signal = lfm_waveform(Ns, k_r, T);
            else % Communication (QAM)
                tx_data = qammod(randi([0 15], num_active, 1), 16, 'UnitAveragePower', true);
                tx_signal = zeros(Ns, 1);
                active_idx = randperm(Ns, num_active);
                tx_signal(active_idx) = tx_data;
            end
            
            % Amplitude limiting
            tx_signal = max(min(tx_signal, 1e3), -1e3);
            
            % 2. Enhanced SNR control
            if rand() > 0.9 % 10% chance for extreme SNR
                if rand() > 0.5 % 50% for low/high
                    current_SNR = extreme_low_SNRs(randi(length(extreme_low_SNRs)));
                else
                    current_SNR = extreme_high_SNRs(randi(length(extreme_high_SNRs)));
                end
            else
                current_SNR = SNR_range(randi(length(SNR_range)));
            end
            current_SNR = max(round(current_SNR), 0); % Final guarantee
            current_SNR_record(i) = current_SNR;
            
            % 3. Channel processing with error handling
            rx_signal = awgn(channel(tx_signal), current_SNR, 'measured');
            reset(channel);
            
            % 4. Safe normalization
            max_val = max(abs(rx_signal));
            if max_val <= 0 || isnan(max_val) || isinf(max_val)
                error('Invalid signal detected');
            end
            X_cplx(:,i) = rx_signal / (max_val + eps);
            
            % Final verification
            if any(isnan(X_cplx(:,i))) || any(isinf(X_cplx(:,i)))
                error('Invalid values after normalization');
            end
            
        catch ME
            % Safe fallback
            X_cplx(:,i) = complex(randn(Ns,1)*0.1, randn(Ns,1)*0.1);
            Y_labels(i) = randi(num_classes);
            current_SNR_record(i) = NaN;
            warning('Sample %d: %s. Replaced with fallback.', i, ME.message);
        end
    end
    
    % Final statistics
    valid_SNRs = current_SNR_record(~isnan(current_SNR_record));
    SNR_stats = struct(...
        'min', min(valid_SNRs), ...
        'max', max(valid_SNRs), ...
        'mean', mean(valid_SNRs), ...
        'std', std(valid_SNRs));
    
    fprintf('=== SNR Statistics ===\n');
    fprintf('Minimum: %.1f dB, Maximum: %.1f dB\n', SNR_stats.min, SNR_stats.max);
    fprintf('Mean: %.1f dB, Std: %.1f dB\n', SNR_stats.mean, SNR_stats.std);
    
    % Final dataset verification
    assert(~any(isnan(X_cplx(:))), 'Final dataset contains NaN');
    assert(~any(isinf(X_cplx(:))), 'Final dataset contains Inf');
end

function [data, fixes] = normalize_with_checks(data)
    fixes = 0;
    for i = 1:size(data,1)
        sample = data(i,:);
        
        % Handle problematic values
        sample(isnan(sample)) = 0;
        sample(isinf(sample)) = 0;
        
        % L2 normalization with verification
        norm_val = norm(sample, 2);
        if norm_val > 0
            data(i,:) = sample / norm_val;
        else
            data(i,:) = sample;
            fixes = fixes + 1;
        end
    end
end

function [accuracy, confMat] = evaluate_model_robust(net, X_val, Y_val)
    classes = categories(Y_val);
    num_classes = length(classes);
    num_samples = size(X_val, 3);
    
    try
        % Correct prediction for LSTM
        Y_pred_raw = predict(net, X_val);
        [~, Y_pred] = max(Y_pred_raw, [], 1); % Corrected: extracts class indices
        
        % Convert to categorical
        Y_pred = categorical(Y_pred', 1:num_classes, classes);
        
        % Calculate metrics
        accuracy = mean(Y_pred == Y_val);
        confMat = confusionmat(Y_val, Y_pred);
        
        fprintf('\n=== Detailed Metrics ===\n');
        fprintf('Accuracy: %.2f%%\n', accuracy*100);
        
        % Add precision/recall per class
        for c = 1:num_classes
            idx = (Y_val == classes(c));
            if any(idx)
                prec = mean(Y_pred(idx) == classes(c));
                rec = mean(Y_pred(Y_val == classes(c)) == classes(c));
                fprintf('Class %d: Precision=%.2f, Recall=%.2f\n', c, prec, rec);
            end
        end
        
    catch ME
        fprintf('\nEvaluation error: %s\n', ME.message);
        accuracy = 0;
        confMat = zeros(num_classes);
    end
end

function X_stft = apply_stft_transform(X_cplx)
    fs = 1; window_length = 64; overlap = 32; nfft = 256;
    X_stft = zeros(size(X_cplx));
    
    for i = 1:size(X_cplx, 2)
        try
            [s, ~] = spectrogram(X_cplx(:,i), hamming(window_length), overlap, nfft);
            mag = abs(s);
            X_stft(:,i) = mean(mag, 2);
            
            % Post-STFT verification
            if any(isnan(X_stft(:,i)))
                error('NaN after STFT');
            end
        catch
            X_stft(:,i) = zeros(size(X_cplx,1),1);
            warning('STFT failed for sample %d - replaced with zeros', i);
        end
    end
end

function lfm_sig = lfm_waveform(N, k_r, T)
    t = linspace(0, T, N).';
    lfm_sig = exp(1j*pi*k_r*t.^2);
    
    % Quality check
    if any(isnan(lfm_sig)) || any(isinf(lfm_sig))
        error('Invalid LFM signal generated');
    end
end