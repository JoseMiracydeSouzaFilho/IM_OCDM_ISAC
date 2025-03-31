%% Main Script: ISAC System with Sparse IM-OCDM and Radar
clc; clear; close all;

%% 1. System Parameters
% Communication
Nc_total = 256;             % Total subchirps
Nc_comm = 128;              % Subchirps allocated for communication
g = 8;                      % Number of sub-blocks
k = 8;                      % Active subchirps per sub-block
M = 16;                     % 16-QAM modulation
SNR_dB = 20;                % Signal-to-noise ratio (dB)

% Radar
radar_BW = 1e6;             % 1 MHz bandwidth
Fs = 2e6;                   % Sampling frequency
pulse_width = 64e-6;        % Pulse width

%% 2. Generate IM-OCDM Communication Signal
% Calculate bits and symbols
n_c = Nc_comm / g;          % Subchirps per sub-block (16)
index_bits_per_block = floor(log2(nchoosek(n_c, k))); % 13 bits/block
total_index_bits = g * index_bits_per_block;          % 104 bits
total_constellation_bits = g * k * log2(M);           % 256 bits
n_bits = total_index_bits + total_constellation_bits; % 360 bits

% Generate random bits
input_bits = randi([0 1], 1, n_bits);

% Split into index and constellation bits
index_bits = input_bits(1:total_index_bits);
constellation_bits = input_bits(total_index_bits+1:end); % 1x256 bits

% Map index bits to active subchirps (critical missing step!)
active_subchirps = index_mapper(index_bits, n_c, k, g); % Returns 1x64 indices

% Reshape constellation bits for 16-QAM
bits_per_symbol = log2(M); % 4 bits/symbol
constellation_matrix = reshape(constellation_bits, bits_per_symbol, []); % 4x64 matrix

% Map to 16-QAM symbols
symbols = qammod(constellation_matrix, M, 'InputType', 'bit', 'UnitAveragePower', true); % 1x64 symbols

% Assign symbols to active subchirps
comm_signal = zeros(Nc_total, 1);
comm_signal(active_subchirps) = symbols; % Now both are 1x64

%% 3. Generate Radar Signal (LFM) on Inactive Subchirps
radar_subchirps = setdiff(1:Nc_total, active_subchirps); % 192 subchirps
lfm_signal = generate_lfm_chirp(radar_BW, pulse_width, Fs, numel(radar_subchirps));

% Combine signals
full_signal = comm_signal;
full_signal(radar_subchirps) = lfm_signal;

%% 4. Simulate Noisy Channel
received_signal = awgn(full_signal, SNR_dB, 'measured');

%% 5. Receiver Processing
% Separate communication and radar signals
received_comm = received_signal(active_subchirps);
received_radar = received_signal(radar_subchirps);

% Demodulate communication signal
demod_matrix = qamdemod(received_comm, M, 'OutputType', 'bit', 'UnitAveragePower', true);
demod_bits = reshape(demod_matrix, 1, []); % Flatten to 1x256

% Process radar signal
[range_est, velocity_est] = process_radar(received_radar, lfm_signal, Fs, pulse_width);

%% 6. Results and Visualization
% Communication: Constellation plot
scatterplot(received_comm);
title('Received 16QAM Constellation (Communication)');

% Radar: Plot matched filter output
t = (0:numel(radar_subchirps)-1)/Fs;
figure;
plot(t, abs(received_radar));
xlabel('Time (s)'); ylabel('Amplitude');
title('Radar Echo Signal (Time Domain)');

% Display radar results (example placeholder)
target_range = 3600; % Example value
target_speed = 40;   % Example value
fprintf('True Range: %.2f km, Estimated Range: %.2f km\n', target_range/1e3, range_est/1e3);
fprintf('True Speed: %.2f m/s, Estimated Speed: %.2f m/s\n', target_speed, velocity_est);