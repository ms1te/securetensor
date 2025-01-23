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
        % Store the private mode cores and shared mode cores separately
        local_cores{k} = G{1};
        size(local_cores{k});
        shared_cores{k} = G(2:end);  % Shared mode cores      
    end
    
    W = 0; % Initialize the accumulated result to 0
    
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
