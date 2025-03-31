%% IMOCDM-ISAC Simulation Script
% Complete implementation with all required parameters
close all;
clear;
clc;

%% Simulation Parameters (Updated)
% Radar parameters
c = 3e8; % Speed of light (m/s)
target_range = 3600; % 3.6 km (from paper)
target_velocity = 40; % m/s (from paper)

% Waveform parameters
k = 15.6e12; % Chirp rate (Hz/s) = 15.6 MHz/μs
B = 1e6; % Bandwidth (1 MHz)
Fs = 2*B; % Sampling frequency (2 MHz)
T = 64e-6; % Pulse duration (64 μs)
Ns = 256; % Number of subchirps
num_active = 128; % Active subchirps (50%)

% Channel parameters
SNR_range = 0:5:30; % dB
noise_power = -20; % dB (from paper Table I)
trials = 1e4; % Monte Carlo trials

% Initialize results
ber = zeros(size(SNR_range));

%% LFM Waveform Generator (15.6 MHz/μs chirp rate)
lfm_gen = @(N) exp(1j*pi*k*(0:N-1).^2/N); 

%% Main Simulation Loop
for snr_idx = 1:length(SNR_range)
    fprintf('Processing SNR = %d dB...\n', SNR_range(snr_idx));
    
    for trial = 1:trials
        % 1. Generate 16QAM symbols for active subchirps
        tx_sym = qammod(randi([0 15], num_active, 1), 16, 'UnitAveragePower', true);
        
        % 2. IM-OCDM Modulation
        tx_sig = ocdm_modulate(tx_sym, lfm_generator, ...
            'TotalSubchirps', Ns, ...
            'ActivationRatio', num_active/Ns);
        
        % 3. Channel Effects
        rx_sig = awgn(tx_sig, SNR_range(snr_idx), 'measured');
        
        % 4. LSTM-based Signal Separation
        [est_sym, ~] = lstm_separate(rx_sig);
        
        % 5. BER Calculation
        ber(snr_idx) = ber(snr_idx) + ...
            sum(qamdemod(est_sym) ~= tx_sym);
    end
    
    % Average BER over trials
    ber(snr_idx) = ber(snr_idx) / (num_active * trials);
end

%% Plot Results
figure;
semilogy(SNR_range, ber, '-o', 'LineWidth', 2);
grid on;
xlabel('SNR (dB)');
ylabel('Bit Error Rate (BER)');
title('IM-OCDM Communication Performance');
legend('Proposed IM-OCDM', 'Location', 'southwest');

%% support Functions --------------------------------------------------------

function tx_signal = ocdm_modulate(data, lfm_generator, varargin)
% OCDM_MODULATE - IM-OCDM transmitter with LFM radar embedding
%
% Inputs:
%   data - Vector of complex symbols (e.g., 16QAM)
%   lfm_generator - Function handle: @(N) generates LFM waveform
%
% Optional Name-Value Pairs:
%   'TotalSubchirps' - Total subchirps (default 256)
%   'ActivationRatio' - Ratio of active subchirps (default 0.5)
%   'ActivationMode' - 'random' (default), 'uniform', or 'custom'
%   'CustomIndices' - Indices for custom activation

% Parameter parsing with inputParser
p = inputParser;

% Validation functions
validNum = @(x) isnumeric(x) && isscalar(x) && (x > 0);
validRatio = @(x) validNum(x) && (x <= 1);
validMode = @(x) any(validatestring(lower(x), ...
    {'random','uniform','custom'}));

% Add parameters
addRequired(p, 'data', @isvector);
addRequired(p, 'lfm_generator', @(x) isa(x, 'function_handle'));
addParameter(p, 'TotalSubchirps', 256, validNum);
addParameter(p, 'ActivationRatio', 0.5, validRatio);
addParameter(p, 'ActivationMode', 'random', validMode);
addParameter(p, 'CustomIndices', [], @isvector);

parse(p, data, lfm_generator, varargin{:});

% Initialize parameters
N_total = p.Results.TotalSubchirps;
N_active = round(p.Results.ActivationRatio * N_total);
data = data(:); % Ensure column vector

% Validate input sizes
if length(data) ~= N_active
    error('Data length mismatch. Expected %d, got %d', N_active, length(data));
end

% Subchirp activation logic
switch lower(p.Results.ActivationMode)
    case 'random'
        active_idx = randperm(N_total, N_active);
    case 'uniform'
        spacing = floor(N_total/N_active);
        active_idx = (1:spacing:N_total);
        active_idx = active_idx(1:N_active);
    case 'custom'
        if length(p.Results.CustomIndices) ~= N_active
            error('Custom indices must match active subchirps count');
        end
        active_idx = p.Results.CustomIndices;
end

% Generate signal
tx_signal = zeros(N_total, 1);
tx_signal(active_idx) = data; % Communication symbols
inactive_idx = setdiff(1:N_total, active_idx);
tx_signal(inactive_idx) = lfm_generator(length(inactive_idx)); % LFM radar

% Power normalization
tx_signal = tx_signal / std(tx_signal);
end

function [comm_sym, radar_sig] = lstm_separate(rx_signal)
    % Load pretrained LSTM network
    persistent net;
    if isempty(net)
        net = load('trained_lstm.mat');
    end
    
    % Normalize input
    rx_signal = rx_signal./max(abs(rx_signal));
    
    % LSTM processing
    output = predict(net, rx_signal);
    
    % Separate outputs
    comm_sym = output(1:128); % First half: communication
    radar_sig = output(129:end); % Second half: radar
end