function [G, ranks, approx_error] = ttmlsvdlast(X, epsilon, max_rank)
    % Input parameters:
    %   X - Input tensor with dimensions I1 x I2 x ... x IN
    %   epsilon - Specified relative error precision
    %   max_rank - Maximum allowable rank (truncation upper limit)
    % Output parameters:
    %   G - Cell array containing TT cores {G1, G2, ..., GN}
    %   ranks - TT rank array [R0, R1, ..., RN]
    %   approx_error - The actual approximation error ||X - B||_F
    
    % Initialization
    N = ndims(X);           % Tensor order
    dims = size(X);         % Size of each dimension of the tensor
    normX = norm(X(:));     % Frobenius norm of the input tensor
    delta = epsilon * normX / sqrt(N - 1); % Dynamic truncation threshold
    ranks = zeros(1, N + 1); % Initialize TT rank array
    ranks(1) = 1;           % R0 = 1
    G = cell(1, N);         % Initialize TT core storage
    
    % Temporary tensor initialization
    C = X;
    
    % Main loop: Decompose layer by layer
    for n = 1:N-1
        % Unfold C into a matrix
        C = reshape(C, [ranks(n) * dims(n), prod(dims(n+1:end))]);
        C = double(C);
        
        % Perform complete MLSVD: SVD + truncation
        [U, S, V] = svd(C, 'econ');
        singular_values = diag(S);
        
        % Compute cumulative energy and truncation index
        cumulative_energy = cumsum(singular_values.^2);
        total_energy = cumulative_energy(end);
        trunc_idx = find(cumulative_energy >= total_energy * (1 - (delta / normX)^2), 1);
        
        % Ensure truncation index is within max_rank
        trunc_idx = min(trunc_idx, max_rank);
        
        % Update TT rank
        ranks(n+1) = trunc_idx;
        
        % Truncate U, S, V
        U = U(:, 1:ranks(n+1));
        S = S(1:ranks(n+1), 1:ranks(n+1));
        V = V(:, 1:ranks(n+1));
        
        % Store the current TT core
        G{n} = reshape(U, [ranks(n), dims(n), ranks(n+1)]);
        
        % Update the temporary tensor
        C = S * V';
        C = reshape(C, [ranks(n+1), dims(n+1:end)]);
    end
    
    % Store the final TT core
    G{N} = reshape(C, [ranks(N), dims(N), 1]);
    
    % Calculate the actual approximation error
    B = tt_reconstruct(G);
    approx_error = norm(X(:) - B(:)) / normX;
end
