function G_k = ctt_dec(K, p, X_k, epsilon1, epsilon2, max_rank, L)
    % ctt_dec: Combine and construct adjacency matrix, generate neighbors, and perform decentralized CTT algorithm
    % 
    % Input:
    %   K: Total number of nodes
    %   p: Probability of connection in the adjacency matrix (value between 0 and 1)
    %   X_k: Tensor data for each node, a Cell array where each cell contains the local tensor of a node
    %   epsilon1: Initial truncation SVD accuracy
    %   epsilon2: TT-SVD accuracy
    %   max_rank: Maximum allowable TT rank
    %   L: Average consistency iteration times
    %
    % Output:
    %   G_k: TT tensor decomposition results for each node, a Cell array

    % Step 1: Generate random neighbor sets
    neighbors = generate_random_neighbors(K, p); % Generate neighbor sets
    
    % Step 2: Construct adjacency matrix
    M = construct_adjacency_matrix(K, neighbors); % Construct adjacency matrix

    % Step 3: Perform decentralized CTT algorithm
    G_k = Decentralized_CTT(X_k, epsilon1, epsilon2, max_rank, L, M); % Perform decentralized CTT
end

function neighbors = generate_random_neighbors(K, p)
    % Generate random neighbors for K nodes
    M = rand(K) < p;  % Generate a KxK matrix with random values between [0, 1] and compare with probability p
    M = triu(M, 1);    % Ensure the matrix is upper triangular, removing the symmetric part
    M = M + M';        % Make the matrix symmetric to ensure an undirected graph structure
    
    % Create neighbor sets
    neighbors = cell(K, 1);  % Initialize a Cell array
    for i = 1:K
        neighbors{i} = find(M(i, :));  % Find neighbors for each node
    end
end

function M = construct_adjacency_matrix(K, neighbors)
    % Construct the adjacency matrix according to the formula (14) in the paper
    M = zeros(K, K); % Initialize the adjacency matrix
    for i = 1:K
        Ni = neighbors{i};        % Get the neighbor set of node i
        di = length(Ni);          % Get the degree (number of neighbors) of node i
        
        if di > 0
            for j = 1:K
                if j == i
                    M(i, j) = (K - di) / K; % Self-loop weight
                elseif ismember(j, Ni)
                    M(i, j) = 1 / K;       % Neighbor weight
                end
            end
        end
    end
    M = (M + M') / 2; % Ensure the matrix is symmetric
    M = M ./ sum(M, 2);  % Normalize each row
end

function [G_k] = Decentralized_CTT(X_k, epsilon1, epsilon2, max_rank, L, M)
    % Decentralized_CTT: Decentralized CTT algorithm
    K = length(X_k);
    G_k = cell(K, 1);
    Z_k = cell(K, 1);  % Data for consistency iteration shared among nodes
    
    % Step 1: Local truncated SVD
    for k = 1:K
        [G_local, ~] = ttmlsvdlast(X_k{k}, epsilon1, max_rank);
        G_k{k} = G_local;
        Z_k{k} = G_local{1}(:);
    end
    
    % Step 2: Average consistency mechanism
    for l = 1:L
        Z_new = cell(K, 1);
        for k = 1:K
            Z_new{k} = zeros(size(Z_k{k}));
            for j = 1:K
                Z_new{k} = Z_new{k} + M(k, j) * Z_k{j};
            end
        end
        Z_k = Z_new;  % Update data for each node
    end
    
    % Step 3: Perform final TT-SVD to extract factors
    for k = 1:K
        reshaped_Z = reshape(Z_k{k}, size(G_k{k}{1}));
        G_k{k}{1} = reshaped_Z;
        [G_final, ~] = ttmlsvdlast(X_k{k}, epsilon2, max_rank);
        G_k{k} = G_final;
    end
end
