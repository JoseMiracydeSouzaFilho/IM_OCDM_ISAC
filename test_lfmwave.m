% Teste rápido da função
N_test = 100;
lfm_test = lfm_waveform(N_test, 15.6e12, 64e-6);

% Verificações
assert(iscolumn(lfm_test), "Deveria retornar vetor coluna!");
assert(all(abs(imag(lfm_test)) <= 1), "Valores fora do range esperado!");
disp("Teste LFM passou com sucesso!");

function lfm_sig = lfm_waveform(N, k_r, T)
    % Geração de waveform LFM otimizada
    % Inputs:
    %   N - Número de amostras
    %   k_r - Chirp rate (Hz/s)
    %   T - Duração (s)
    %
    % Output:
    %   lfm_sig - Sinal LFM complexo (N×1)

    t = linspace(0, T, N).'; % Vetor coluna
    lfm_sig = exp(1j*pi*k_r*t.^2);
end