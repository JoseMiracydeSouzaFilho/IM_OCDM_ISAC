clear;
% Parameters matching Table 1 from reference
Ns = 256; Fs = 2e6; T = 64e-6; 
SNR_range = 0:5:30; trials = 1e4;

% Reference implementation
for snr_idx = 1:length(SNR_range)
    for trial = 1:trials
        % 16QAM modulation (communication)
        tx_sym = qammod(randi([0 15],128,'UnitAveragePower',true));
        
        % OFDM with chirp radar (continuous config)
        ofdm_sig = ifft([tx_sym; chirp_signal(128,Fs)],Ns);
        
        % Channel effects
        rx_sig = awgn(ofdm_sig,SNR_range(snr_idx),'measured');
        
        % BER calculation (matches reference Table II)
        ber(snr_idx) = ber(snr_idx) + ...
            sum(qamdemod(rx_sig(1:128)) ~= tx_sym);
    end
end
ber = ber/(128*trials);