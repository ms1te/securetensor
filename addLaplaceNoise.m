function G_tensor_noisy = addLaplaceNoise(G_tensor, mu, b)
    % Input:
    % G_tensor - A tensor of size 6x30x30
    % mu - Mean of the Laplace noise
    % b - Scale parameter of the Laplace noise
    
    % Get the dimensions of the tensor
    [m, n, p] = size(G_tensor);
    
    % Generate Laplace noise with the same dimensions as G_tensor
    laplace_noise = mu - b * sign(rand(m, n, p) - 0.5) .* log(1 - 2 * abs(rand(m, n, p) - 0.5));
    
    % Add the noise to the tensor
    G_tensor_noisy = G_tensor + laplace_noise;
end
