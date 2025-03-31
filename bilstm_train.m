%% 0. Configuração Inicial
clear; clc; close all;
fprintf('=== Enhanced IM-OCDM with Dynamic Index Modulation + Hybrid Deep Learning ===\n');
fprintf('Transmissão IM-OCDM: 128 slots adaptativos para comunicação + 128 LFM\n');
fprintf('Recepção via CNN-BiLSTM-Attention: https://arxiv.org/pdf/2305.03192\n');

%% 1. System Parameters (Updated)
SNR_range = -20;            % Noise power = -20 dB (single SNR point)
num_classes = 4;            % 4 index modulation patterns
Nc = 256;                   % Total subchirps (maintained)

% Radar parameters
radar_center_freq = 0;          % base band
radar_BW = 1e6;             % 1 MHz bandwidth
Fs = 2e6;                   % 2 MHz sampling frequency
pulse_width = 64e-6;        % 64 μs pulse width
PRP = 480e-6;               % 480 μs pulse repetition period

% Derived parameters
samples_per_pulse = Fs * pulse_width; % 128 samples (matches Ns_radar = 128)
target_range = 3600;        % 3.6 km target distance
target_speed = 40;          % 40 m/s target speed

%% 2. DFnT Matrix Setup (Unchanged)
[DFnT, IDFnT] = generate_dfnt_matrix(Nc); % Get both matrices
% Phi_H = Phi';                % IDFnT matrix

%% 3. Testes de Sanidade Atualizados
fprintf('Running enhanced sanity tests...\n');

% Test dynamic patterns
test_patterns = generate_im_patterns(128, 4);
all_indices = [test_patterns{:}];
unique_count = length(unique(all_indices));

fprintf('Subchirps utilizados: %d/128\n', unique_count);
assert(unique_count >= 100, ...
    'Padrões IM não estão utilizando subchirps suficientes (%d/128)', unique_count);

%% 4. Geração de Dataset com Padrões Dinâmicos
num_train_samples = 10000;  
num_val_samples = 2000;     % Aumentado para validação mais robusta

[X_train, Y_train] = generate_imocdm_dataset(num_train_samples, num_classes, IDFnT, Fs);
[X_val, Y_val] = generate_imocdm_dataset(num_val_samples, num_classes, Phi_H, Fs);

% Balanceamento de classes (atualizado para padrões dinâmicos)
[Y_train, X_train] = balance_classes(Y_train, X_train, num_classes);

%% 5. Aumento de Dados Avançado
X_train = advanced_augmentation(X_train, Fs, radar_center_freq);

%% 6. STFT Preprocessing
% Compute STFT dimensions
[stft_example, num_frames, num_features] = compute_stft(X_train(:,1), [64, 128, 256], 256);

% Process all samples
X_train_cell = process_stft_batch(X_train, num_frames, num_features);
X_val_cell = process_stft_batch(X_val, num_frames, num_features);

%% 7. Arquitetura Híbrida CNN-BiLSTM-Attention
inputSize = 13; %force
numFeatures = inputSize;

% BiLSTM layer setup (ensure hiddenStateDim matches)
hiddenUnits = 16;
hiddenStateDim = 2 * hiddenUnits; % Bidirectional → 2x hidden units
layers = [
    sequenceInputLayer(inputSize)
    bilstmLayer(hiddenUnits, 'OutputMode', 'last')
    %attentionLayer(128, hiddenStateDim, 'Name', 'attention') % AttentionSize=128
    fullyConnectedLayer(num_classes)
    softmaxLayer
    classificationLayer
];

%% 8. Configuração de Treino Otimizada (CPU)
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 32, ... % Reduced from 256
    'InitialLearnRate', 0.0001, ... % Lower initial rate
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 10, ...
    'LearnRateDropFactor', 0.5, ...
    'ValidationData', {X_val_cell, Y_val}, ...
    'ValidationFrequency', 100, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', true);

% options = trainingOptions('adam', ...
%     'GradientThreshold', 1.5, ...  % Added for stability
%     'MaxEpochs', 100, ...
%     'MiniBatchSize', 256, ...
%     'InitialLearnRate', 1e-3, ...
%     'LearnRateSchedule', 'piecewise', ...
%     'L2Regularization', 0.001, ...
%     'ValidationData', {X_val_cell, Y_val}, ...
%     'ValidationFrequency', 200, ...
%     'Shuffle', 'every-epoch', ...
%     'OutputNetwork', 'best-validation-loss', ...
%     'Plots', 'training-progress', ...
%     'ExecutionEnvironment', 'gpu');

%% 9. Treinamento e Avaliação
[net, info] = trainNetwork(X_train_cell, Y_train, layers, options);

% Avaliação robusta
[accuracy, confMat] = evaluate_model_robust(net, X_val_cell, Y_val);














