%% 0. Configuração Inicial
clear; clc; close all;
fprintf('=== Enhanced IM-OCDM with Index Modulation + LSTM - Signal Classification System ===\n');
fprintf('Transmissão IM-OCDM: 128 slots para comunicação (Index Modulation) e 128 slots para radar (LFM chirp).\n');
fprintf('Recepção via LSTM conforme: https://arxiv.org/pdf/2305.03192\n');

%% 1. System Parameters
SNR_range = 0:5:40;         % Intervalo de SNR (0-40 dB)
num_classes = 4;            % 4 padrões de index modulation
Nc = 256;                   % Total de subchirps
numHiddenUnits = 128;       % LSTM hidden units

%% 2. DFnT Matrix Setup
Phi = generate_dfnt_matrix(Nc);  % DFnT matrix 
Phi_H = Phi';                    % IDFnT matrix

%% 3. Testes de Sanidade Iniciais
fprintf('Running initial sanity tests...\n');

% Test basic DFnT/IDFnT operations
x_test = zeros(Nc,1); 
x_test(1:10) = 1;
s_tx = Phi_H * x_test;     % Transmission
x_rx = Phi * s_tx;         % Reception
assert(norm(x_test - x_rx) < 1e-10, 'DFnT/IDFnT failed basic test');

% Test STFT detection
stft_out = abs(spectrogram(s_tx, hamming(64), 48, 256, 1));
assert(any(stft_out(1:10,:),'all'), 'STFT failed to detect active subchirps');

% Test LFM generation
test_chirp = lfm_waveform(128, 15.6e12, 64e-6);
assert(~any(isnan(test_chirp)), 'LFM generation error');

fprintf('All sanity checks passed!\n');

%% 4. Dataset Generation (Updated)
num_train_samples = 20000;  
num_val_samples = 3000;   

% Generate original dataset
[X_train, Y_train] = generate_imocdm_dataset(num_train_samples, SNR_range, num_classes, Phi_H);
[X_val, Y_val] = generate_imocdm_dataset(num_val_samples, SNR_range, num_classes, Phi_H);

% --- Start of Class Balancing ---
% Check current class distribution
[class_counts, edges] = histcounts(Y_train);
class_labels = edges(1:end-1);  % Get actual class labels
median_count = median(class_counts);

fprintf('Original Class Distribution:\n');
disp(table(class_labels', class_counts', 'VariableNames', {'Class', 'Count'}));

% Oversample minority classes
for c = 1:num_classes
    current_count = class_counts(c);
    
    % If class has <80% of median samples, create synthetic ones
    if current_count < 0.8*median_count
        needed = round(median_count - current_count);
        
        % Get indices of current class samples
        class_indices = find(Y_train == class_labels(c));
        
        % Randomly duplicate existing samples with noise (simple oversampling)
        synthetic_indices = randi(length(class_indices), [1, needed]);
        synthetic_samples = X_train(:, class_indices(synthetic_indices));
        
        % Add Gaussian noise to duplicates
        synthetic_samples = synthetic_samples + 0.02*randn(size(synthetic_samples));
        
        % Append to dataset
        X_train = [X_train synthetic_samples];
        Y_train = [Y_train repelem(class_labels(c), needed)];
    end
end

fprintf('Balanced Class Distribution:\n');
balanced_counts = histcounts(Y_train);
disp(table(class_labels', balanced_counts', 'VariableNames', {'Class', 'NewCount'}));
% --- End of Balancing ---

% Convert to categorical
Y_train = categorical(Y_train);
Y_val = categorical(Y_val);

%% 5. Enhanced Data Augmentation 
fprintf('Applying advanced data augmentation...\n');
augmentation_factor = 0.8; % Increased from 0.3

parfor i = 1:num_samples
    % Time-domain warping
    if rand() < 0.4
        warp_factor = 0.9 + 0.2*rand();
        X_train(:,i) = resample(X_train(:,i), warp_factor, 1);
    end
    
    % Additive noise with adaptive SNR
    if rand() < augmentation_factor
        valid_snr = 5 + 15*rand(); % SNR between 5-20 dB
        X_train(:,i) = awgn(X_train(:,i), valid_snr, 'measured');
    end
    
    % Random cyclic shifts (10% of samples)
    if rand() < 0.1
        shift = randi([1 Nc/4]);
        X_train(:,i) = circshift(X_train(:,i), shift);
    end
end

%% 6. STFT Preprocessing 
fprintf('Processing signals with STFT...\n');
win_sizes = [64, 128, 256];
nfft = 256;

% Calculate actual feature dimensions
[~,~,t] = spectrogram(X_train(:,1), hamming(win_sizes(1)), round(win_sizes(1)*0.75), nfft, 1);
num_frames = length(t);
num_features = length(win_sizes) * nfft; % 3 windows × 256 frequency bins = 768

% Preallocate arrays with correct dimensions
X_train_stft = zeros(num_features, num_frames, size(X_train,2));
X_val_stft = zeros(num_features, num_frames, size(X_val,2));

% Process training data
parfor i = 1:size(X_train,2)
    X_train_stft(:,:,i) = compute_stft(X_train(:,i), win_sizes, nfft, num_frames);
end

% Process validation data
parfor i = 1:size(X_val,2)
    X_val_stft(:,:,i) = compute_stft(X_val(:,i), win_sizes, nfft, num_frames);
end

% Convert 3D array [num_features × num_frames × num_samples] to cell array
X_train_cell = squeeze(num2cell(X_train_stft, [1 2]));  % Each cell: [num_features × num_frames]
X_val_cell = squeeze(num2cell(X_val_stft, [1 2]));


for i = 1:num_train_samples
    if size(X_train_cell{i}, 2) ~= num_frames
        error('Inconsistent frame count in sample %d', i);
    end
end

%% 7. LSTM Model Setup
inputSize = size(X_train_stft, 1);  % Should be num_features (768)
numFeatures = inputSize;  % Add clarity

% Revised LSTM Architecture (simplify and regularize)
layers = [
    sequenceInputLayer(numFeatures)
    
    % First LSTM with dropout
    lstmLayer(64, 'OutputMode','sequence', 'InputWeightsInitializer','he')
    dropoutLayer(0.6)
    
    % Temporal BatchNorm
    sequenceBatchNormalizationLayer()
    
    % Second LSTM (reduced units)
    lstmLayer(32, 'OutputMode','last')
    
    % Dense layers
    fullyConnectedLayer(48, 'BiasInitializer','narrow-normal')
    reluLayer()
    dropoutLayer(0.5)
    
    fullyConnectedLayer(num_classes)
    softmaxLayer()
    classificationLayer()
];
% Sanity checks
assert(iscell(X_train_cell), 'Training data must be cell array');
assert(numel(X_train_cell) == num_train_samples, 'Cell count mismatch');
assert(size(X_train_cell{1},1) == numFeatures, 'Feature dimension error')

%% 8. Training Configuration
options = trainingOptions('adam', ...
    'MaxEpochs', 150, ...
    'MiniBatchSize', 128, ...          % Reduced from 256
    'InitialLearnRate', 3e-5, ...      % Lower initial LR
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.3, ...    % More aggressive decay
    'LearnRateDropPeriod', 15, ...
    'GradientThreshold', 0.8, ...      % Prevent exploding gradients
    'L2Regularization', 0.005, ...     % Explicit weight regularization
    'ValidationData', {X_val_cell, Y_val}, ...
    'ValidationFrequency', 50, ...     % More frequent checks
    'Shuffle', 'every-epoch', ...
    'ExecutionEnvironment', 'gpu', ...
    'OutputNetwork', 'best-validation-loss', ...
    'Plots', 'training-progress', ...
    'Verbose', true);

%% 9. Training & Evaluation

% Early stopping callback
early_stop = createEarlyStopping(...
    'ValidationPatience', 8, ...       % Stop if no improvement in 8 val checks
    'MiniBatchSize', options.MiniBatchSize, ...
    'Verbose', true);

% Modify training call
[net, info] = trainNetwork(X_train_cell, Y_train, layers, options, ...
    'CheckpointPath', 'temp_checkpoints', ...
    'TrainingPlots', 'progress', ...
    'ExecutionEnvironment', 'auto', ...
    'OutputFcn', early_stop);


% Final evaluation
Y_pred = classify(net, X_val_cell);
accuracy = sum(Y_pred == Y_val)/numel(Y_val);
confMat = confusionmat(Y_val, Y_pred);

fprintf('\n=== Final Results ===\n');
fprintf('Accuracy: %.2f%%\n', accuracy*100);
disp('Confusion Matrix:');
disp(confMat);

%% ================== ENHANCED FUNÇÕES AUXILIARES =====================

function [X_cplx, Y_labels, SNR_stats] = generate_imocdm_dataset(num_samples, SNR_range, num_classes, Phi_H)
    % Enhanced IM-OCDM parameters
    Ns_total   = 256;    % Total de slots
    Ns_comm    = 128;    % Slots para comunicação
    Ns_radar   = 128;    % Slots para radar (LFM chirp)
    active_block_length = 16;
    
    % Enhanced pattern generation - random but non-overlapping
    pattern_indices = generate_non_overlapping_patterns(Ns_comm, 16, num_classes);
    used_indices = [];
    for p = 1:num_classes
        available_indices = setdiff(1:Ns_comm-active_block_length+1, used_indices);
        if isempty(available_indices)
            error('Cannot generate more non-overlapping patterns');
        end
        start_idx = available_indices(randi(length(available_indices)));
        pattern_indices{p} = start_idx:(start_idx+active_block_length-1);
        used_indices = [used_indices, start_idx:(start_idx+active_block_length-1)];
    end
    
    % Enhanced radar parameters with variation
    k_r = 15.6e12;    % Base chirp rate
    T   = 64e-6;      % Base duration
    
    % Enhanced channel model
    Fs = 2e6;
    channel = comm.RayleighChannel(...
        'SampleRate', Fs, ...
        'PathDelays', [0, 1e-6, 2.3e-6, 3.7e-6], ... % Added path
        'AveragePathGains', [0, -3, -6, -9], ...      % Added gain
        'MaximumDopplerShift', 200, ...
        'RandomStream', 'mt19937ar with seed', ...
        'Seed', randi(10000));
    
    % Enhanced SNR extremes
    extreme_low_SNRs  = [min(SNR_range)-10, -5, 0, 2];
    extreme_high_SNRs = [max(SNR_range)+10, max(SNR_range)+20, max(SNR_range)+30];
    
    % Initialize outputs
    X_cplx = zeros(Ns_total, num_samples);
    Y_labels = zeros(1, num_samples);
    current_SNR_record = zeros(1, num_samples);
    
    for i = 1:num_samples
        try
            % 1. Select class with enhanced pattern generation
            class_label = randi(num_classes);
            Y_labels(i) = class_label;
            
            % 2. Enhanced communication signal generation
            comm_signal = zeros(Ns_comm, 1);
            active_idx = pattern_indices{class_label};
            % QAM-16 with phase noise
            tx_data = qammod(randi([0 15], length(active_idx), 1), 16, 'UnitAveragePower', true);
            phase_noise = 0.1*randn(size(tx_data)); % Add phase noise
            comm_signal(active_idx) = tx_data .* exp(1j*phase_noise);
            
            % 3. Enhanced radar signal with parameter variation
            radar_signal = lfm_waveform(Ns_radar, k_r*(0.9 + 0.2*rand()), T*(0.95 + 0.1*rand()));
            
            % 4. Combine signals with amplitude balancing
            tx_symbols = [comm_signal; radar_signal];
            tx_signal = Phi_H * tx_symbols; % Transformada inversa de Fresnel
            tx_signal = tx_signal / max(abs(tx_signal)); % Normalize
            
            if mod(i, 1000) == 0 % Verifica a cada 1000 amostras
                stft_test = abs(spectrogram(tx_signal, hamming(64), 48, 256, 1));
                active_freqs = find([comm_signal; radar_signal] ~= 0);
                assert(any(stft_test(active_freqs, :), 'all'), ...
                    'STFT falhou na amostra %d: subchirps ativos não detectados', i);
            end
            
            % 5. Enhanced SNR selection
            if rand() > 0.85  % 15% extreme cases
                if rand() > 0.5
                    current_SNR = extreme_low_SNRs(randi(length(extreme_low_SNRs)));
                else
                    current_SNR = extreme_high_SNRs(randi(length(extreme_high_SNRs)));
                end
            else
                current_SNR = SNR_range(randi(length(SNR_range)));
            end
            current_SNR = max(round(current_SNR), -5); % Allow slightly negative SNR
            current_SNR_record(i) = current_SNR;
            
            % 6. Enhanced channel modeling
            rx_signal = channel(tx_signal);
            rx_signal = awgn(rx_signal, current_SNR, 'measured');
            reset(channel);
            
            % 7. Robust normalization
            max_val = max(abs(rx_signal));
            if max_val <= 0 || isnan(max_val) || isinf(max_val)
                error('Invalid signal detected');
            end
            X_cplx(:,i) = rx_signal / (max_val + eps);
            
        catch ME
            % Enhanced fallback generation
            X_cplx(:,i) = complex(randn(Ns_total,1)*0.05, randn(Ns_total,1)*0.05);
            Y_labels(i) = randi(num_classes);
            current_SNR_record(i) = NaN;
            warning('Sample %d: %s. Replaced with enhanced fallback.', i, ME.message);
        end
    end
    
    % Enhanced SNR statistics
    valid_SNRs = current_SNR_record(~isnan(current_SNR_record));
    SNR_stats = struct(...
        'min', min(valid_SNRs), ...
        'max', max(valid_SNRs), ...
        'mean', mean(valid_SNRs), ...
        'std', std(valid_SNRs), ...
        'percentiles', prctile(valid_SNRs, [0 25 50 75 100]));
    fprintf('=== Enhanced SNR Statistics ===\n');
    fprintf('Range: %.1f dB to %.1f dB\n', SNR_stats.min, SNR_stats.max);
    fprintf('Mean: %.1f dB, Std: %.1f dB\n', SNR_stats.mean, SNR_stats.std);
    fprintf('Percentiles: %.1f, %.1f, %.1f, %.1f, %.1f dB\n', SNR_stats.percentiles);
end

% STFT Preprocessing (Add Normalization)
% function stft_out = compute_stft(signal, win_sizes, nfft, num_frames)
%     stft_multi = [];
% 
%     for win = win_sizes
%         [S,~,~] = spectrogram(signal, hamming(win), round(win*0.75), nfft, 1);
%         S = abs(S(:,1:num_frames));
% 
%         % Per-channel normalization
%         S = (S - mean(S(:))) ./ std(S(:));
% 
%         stft_multi = [stft_multi; S]; 
%     end
% 
%     stft_out = stft_multi;
% end

function lfm_sig = lfm_waveform(N, k_r, T)
    t = linspace(0, T, N).';
    k_r_var = k_r * (0.9 + 0.2*rand());
    win = hamming(N);
    lfm_sig = win .* exp(1j*pi*k_r_var*t.^2);
    phase_noise = 0.05*randn(N,1);
    lfm_sig = lfm_sig .* exp(1j*phase_noise);
end

function [accuracy, confMat] = evaluate_model_robust(net, X_val, Y_val)
    classes = categories(Y_val);
    num_classes = length(classes);
    try
        Y_pred_raw = predict(net, X_val);
        [~, Y_pred] = max(Y_pred_raw, [], 1);
        Y_pred = categorical(Y_pred', 1:num_classes, classes);
        
        % Enhanced evaluation metrics
        accuracy = mean(Y_pred == Y_val);
        confMat = confusionmat(Y_val, Y_pred);
        
        % Per-class metrics
        fprintf('\n=== Enhanced Evaluation Metrics ===\n');
        fprintf('Overall Accuracy: %.2f%%\n', accuracy*100);
        
        % Precision, Recall, F1 for each class
        for c = 1:num_classes
            TP = sum(Y_pred == classes(c) & Y_val == classes(c));
            FP = sum(Y_pred == classes(c) & Y_val ~= classes(c));
            FN = sum(Y_pred ~= classes(c) & Y_val == classes(c));
            
            precision = TP / (TP + FP + eps);
            recall = TP / (TP + FN + eps);
            f1 = 2 * (precision * recall) / (precision + recall + eps);
            
            fprintf('Class %d: Precision=%.2f, Recall=%.2f, F1=%.2f\n', ...
                c, precision, recall, f1);
        end
        
        % Confusion matrix with percentages
        confMatPercent = bsxfun(@rdivide, confMat, sum(confMat,2)) * 100;
        disp('Confusion Matrix (%):');
        disp(confMatPercent);
        
    catch ME
        fprintf('\nEvaluation error: %s\n', ME.message);
        accuracy = 0;
        confMat = zeros(num_classes);
    end
end


function patterns = generate_non_overlapping_patterns(N, block_len, num_patterns)
    patterns = cell(1, num_patterns);
    used = false(1, N);
    for p = 1:num_patterns
        available = find(~used(1:end-block_len+1));
        start_idx = available(randi(length(available)));
        patterns{p} = start_idx:(start_idx+block_len-1);
        used(patterns{p}) = true;
    end
end


function Phi = generate_dfnt_matrix(N)
    [m,n] = meshgrid(0:N-1, 0:N-1);
    if mod(N,2) == 0
        Phi = (1/sqrt(N)) * exp(-1j*pi/4) * exp(1j*pi/N * (m-n).^2);
    else
        Phi = (1/sqrt(N)) * exp(-1j*pi/4) * exp(1j*pi/N * (m+0.5-n).^2);
    end
end

% Updated STFT Computation Function
function stft_matrix = compute_stft(signal, win_sizes, nfft, num_frames)
    fs = 1;
    stft_matrix = zeros(length(win_sizes)*nfft, num_frames);
    row_idx = 1;
    
    for win_size = win_sizes
        overlap = round(win_size * 0.75);
        [s,~,~] = spectrogram(signal, hamming(win_size), overlap, nfft, fs);
        s = abs(s);
        
        if size(s,2) > num_frames
            s = s(:,1:num_frames);
        elseif size(s,2) < num_frames
            s = [s, zeros(size(s,1), num_frames-size(s,2))];
        end
        
        stft_matrix(row_idx:row_idx+nfft-1, :) = s;
        row_idx = row_idx + nfft;
    end
end