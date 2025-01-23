function result = computeRatio(tensor1, tensor2, tensor3)
    % Input:
    % tensor1, tensor2, tensor3 - Three tensors
    % Output:
    % result - The computed ratio
    
    % Get the total number of elements in each tensor (i.e., the size of the tensor)
    size1 = numel(tensor1);  % Total number of elements in tensor1
    size2 = numel(tensor2);  % Total number of elements in tensor2
    size3 = numel(tensor3);  % Total number of elements in tensor3
    
    % Calculate the ratio
    result = size2 / (size1 + size2 + size3);
end
