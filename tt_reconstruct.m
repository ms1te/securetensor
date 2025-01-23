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
    
    % Remove the final rank dimension
    B = squeeze(B);
end
