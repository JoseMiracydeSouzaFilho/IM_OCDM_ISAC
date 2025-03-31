function [Phi, Phi_H] = generate_dfnt_matrix(N)
    [m,n] = meshgrid(0:N-1, 0:N-1);
    if mod(N,2) == 0
        Phi = (1/sqrt(N)) * exp(-1j*pi/4) * exp(1j*pi/N * (m-n).^2);
    else
        Phi = (1/sqrt(N)) * exp(-1j*pi/4) * exp(1j*pi/N * (m+0.5-n).^2);
    end

    Phi_H = Phi;
end

