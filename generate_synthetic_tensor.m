function synthetic_tensor = generate_synthetic_tensor(tensor_size, non_zero_ratio, num_population_modes)
% Function to generate synthetic data
% 
% Input:
%   tensor_size          - Size of the tensor, e.g., [200, 30, 30]
%   non_zero_ratio       - Ratio of non-zero elements, e.g., 0.4
%   num_population_modes - Number of sparse feature mode matrices, e.g., 5
% 
% Output:
%   synthetic_tensor - Generated sparse low-rank tensor

    % 1. Randomly generate sparse feature mode matrices
    population_modes = cell(1, num_population_modes);
    for i = 1:num_population_modes
        mode_matrix = randn(tensor_size(2), tensor_size(3));  % Generate standard Gaussian distribution matrix
        sparsity_mask = rand(tensor_size(2), tensor_size(3)) < non_zero_ratio;  % Control sparsity
        mode_matrix(~sparsity_mask) = 0;  % Set some elements to zero to achieve sparsity
        population_modes{i} = mode_matrix;
    end

    % 2. Randomly generate personalized mode matrix for each client
    personal_mode = randn(tensor_size(2), tensor_size(3));  % Personalized mode matrix

    % 3. Combine feature mode matrices and personalized mode matrix to generate low-rank tensor
    synthetic_tensor = zeros(tensor_size);  % Initialize the tensor
    for i = 1:tensor_size(1)
        mode_index = randi(num_population_modes);  % Randomly choose a sparse feature mode matrix
        selected_mode = population_modes{mode_index};
        synthetic_tensor(i, :, :) = personal_mode + selected_mode;  % Combine to generate tensor
    end

    % 4. Control the sparsity of the tensor
    sparsity_mask_tensor = rand(tensor_size) < non_zero_ratio;  % Global sparsity mask
    synthetic_tensor(~sparsity_mask_tensor) = 0;  % Set some elements to zero to achieve sparsity
end
