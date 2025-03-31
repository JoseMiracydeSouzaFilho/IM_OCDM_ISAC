% Test with minimal samples
% num_samples = 100;
% [X, Y] = generate_imocdm_dataset(num_samples, 0:10, 4, Phi_H);
% 
% % Check outputs
% assert(size(X, 2) == num_samples, 'X matrix size mismatch');
% assert(length(Y) == num_samples, 'Y labels size mismatch');
% assert(isa(Y, 'categorical'), 'Y must be categorical');
% 
% % Test with imbalanced data
% Y_test = categorical([1 1 2 2 2 3 4]', 1:4); % 2,3,1,1 samples
% X_test = randn(256, 7);
% [Y_bal, X_bal] = balance_classes(Y_test, X_test, 4);
% 
% % Verify counts
% tabulate(Y_bal)
% 
% % Test with 30 GHz sampling and 24 GHz radar
% X_test = randn(256,10);
% X_aug = advanced_augmentation(X_test, 30e9);
% 
% % Check Nyquist compliance
% figure;
% spectrogram(X_aug(:,1), hamming(64), 60, 256, 30e9, 'centered');
% title('Augmented Signal Spectrum (Should stay below 15 GHz)');

% Test the attention layer
% analyzeNetwork(layers);

clear;

%% Validate DFnT/IDFnT duality
N = 2;  % Test for even N
[Phi, Phi_H] = generate_dfnt_matrix(N);
I = Phi * Phi_H';  % Should be close to identity matrix
disp(norm(I - eye(N)));  % Should be very small (e.g., < 1e-14)

