epsilon = 0.01; % TT decomposition accuracy parameter
max_rank = 20; % Maximum TT rank
mode_rank = 10;
% X_all = cell(1, 4); % Initialize cell array

% Check and load synthetic data
% X_all = untitled(dims, sparsity, mode_rank,K);
% X_all = data_tensor;
% for i = 1:4
%     X_all{i} = low_rank_tensor_1;
% end
% Run CTT(Ms)
[MSreconstructed_tensor, global_cores, local_cores, ranks, errors] = ctt_ms(MSdata, epsilon, max_rank);

function [reconstructed_tensor, global_cores, local_cores, ranks, errors] = ctt_ms(X_all, epsilon, max_rank)
    K = numel(X_all);  
    local_cores = cell(1, K); 
    shared_cores = cell(1, K); 
    ranks = cell(1, K);        % Store the TT ranks for each client
    errors = zeros(1, K);      % Store the RSE for each client

    % 1. Local TT decomposition for each client
    for k = 1:K
        % Perform TT-SVD on each client's data tensor X^k
        [G, rank_k] = ttmlsvdlast(X_all{k}, epsilon, max_rank);

        ranks{k} = rank_k;
        % Store private mode cores and shared mode cores separately
        local_cores{k} = G{1};
        size(local_cores{k});
        shared_cores{k} = G(2:end);  % Shared mode cores      
    end
    
    W = 0; % Initialize accumulated result to 0
    
    for k = 1:4
        % Get the shared cores
        core1 = shared_cores{k}{1}; 
        core2 = shared_cores{k}{2};  
        core1 = tensor(core1);
        core2 = double(core2);
       
        result = ttm(core1, core2', 3); % Perform tensor-matrix multiplication in the second dimension
    
        % Accumulate the result
        W = W + result;
    end

    % Compute the average
    W = W / 4;
    
    size(W)
    [global_cores, ~] = ttmlsvdlast(W, epsilon, max_rank);

    % 3. Client reconstruction and error evaluation
    for k = 1:K     
        global_cores{1} = squeeze(global_cores{1});
        global_cores{2} = tensor(global_cores{2});
        global_cores{3} = double(global_cores{3});
        
        c = ttm(global_cores{2}, global_cores{1}, 1);

        gd = ttm(c, global_cores{3}', 3);

        gd = tensor(gd);
        local_cores{k} = squeeze(local_cores{k});
        size(local_cores{k});
        local_cores{k} = double(local_cores{k});
        reconstructed_tensor = ttm(gd, local_cores{k}, 1);
        size(reconstructed_tensor);
        % Compute the reconstruction error (RSE)
        original_tensor = X_all{k};
        errors(k) = norm(reconstructed_tensor(:) - original_tensor(:)) / norm(original_tensor(:));
    end
end

function [G, ranks, approx_error] = ttmlsvdlast(X, epsilon, max_rank)
    % Input parameters:
    %   X - Input tensor, with dimensions I1 x I2 x ... x IN
    %   epsilon - Relative error precision
    %   max_rank - Maximum allowable rank (truncation upper limit)
    % Output parameters:
    %   G - Cell array containing TT cores {G1, G2, ..., GN}
    %   ranks - TT rank array [R0, R1, ..., RN]
    %   approx_error - The actual approximation error ||X - B||_F
    
    % Initialize
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
        % Unfold C as a matrix
        C = reshape(C, [ranks(n) * dims(n), prod(dims(n+1:end))]);
        C = double(C);
        
        % Perform complete MLSVD: SVD + truncation
        [U, S, V] = svd(C, 'econ');
        singular_values = diag(S);
        
        % Calculate cumulative energy and truncation index
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

function B = tt_reconstruct(G)
    % Reconstruction function to reconstruct tensor B from TT cores
    % Input parameters:
    %   G - TT core cell array {G1, G2, ..., GN}
    % Output parameters:
    %   B - Reconstructed tensor
    
    N = numel(G); % Number of cores
    B = G{1};     % Start with the first core
    
    % Gradually merge with subsequent cores
    for n = 2:N
        % Size of the current core
        [R_prev, I_n, R_next] = size(G{n});
        
        % Tensor multiplication and unfold
        B = reshape(B, [], R_prev);
        B = B * reshape(G{n}, R_prev, I_n * R_next);
        B = reshape(B, [], I_n, R_next);
    end
    
    % Remove the last rank dimension
    B = squeeze(B);
end
