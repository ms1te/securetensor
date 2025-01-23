clear;
clc;
addpath('tensor_toolbox-v3.1')
% Parameter settings
opts.tol = 1e-5;       % Convergence tolerance
opts.maxit = 1000;     % Maximum number of iterations
opts.rho = 1e-1;       % Regularization parameter
opts.R = [
    0, 6, 6;
    0, 0, 6;
    0, 0, 0
];
opts.max_R = [
    0, 6, 6;
    0, 0, 6;
    0, 0, 0
];

M = 5; % Number of clients
MSdata = cell(1, M);
DECdata = cell(1, M);
max_rounds = 5; % Maximum number of communication rounds
RSE_clients = zeros(1, M);
F_clients = cell(1, M);
tensor_size = [200, 30, 30];  % Tensor dimensions
non_zero_ratio = 0.4;         % Non-zero element ratio
num_population_modes = 5;     % Number of sparse feature mode matrices
rho = 1e-5;
epsilon = 0.01; % TT decomposition accuracy parameter
max_rank = 20; % Maximum TT rank
mode_rank = 10;
epsilon1 = 0.01;  % Initial truncation SVD accuracy
epsilon2 = 0.1; % TT-SVD accuracy
max_rank = 10;    % Maximum allowed rank
L = 10;           % Average consistency iteration times
p = 0.4;  % Connection probability is 0.4
mu = 0;      % Mean
b = 0.1;     % Scaling parameter, this can be adjusted as needed
M = 5; % Number of clients

for i = 1:M
    rng('shuffle');  % Use timestamp as the random seed to ensure different random numbers
    F_clients{i} = generate_synthetic_tensor(tensor_size, non_zero_ratio, num_population_modes);
    tic;
    [X_clients{i}, G_clients{i}, Out_clients{i}] = inc_FCTN_TC(F_clients{i}, true(size(F_clients{i})), opts);
end

G_global = cell(1, ndims(F_clients{1})); % This refers to 3D, global is 1*3
dim = size(G_clients{i}{2});
G_global_last = zeros(dim);

for i = 1:M
    G_tensor_noisy{i} = addLaplaceNoise(G_clients{i}{2}, mu, b);
    MSdata{i} = G_tensor_noisy{i};
    DECdata{i} = G_tensor_noisy{i};
end

[MSreconstructed_tensor, global_cores, local_cores, ranks, errors] = ctt_ms(MSdata, epsilon, max_rank);
elapsedTime = toc;
G_k = ctt_dec(M, p, DECdata, epsilon1, epsilon2, max_rank, L);

for i = 1:M
    G_clients{i}{2} = double(MSreconstructed_tensor);
end

for i = 1:M
    global_cores = G_k{i};
    global_cores{1} = squeeze(global_cores{1});
    global_cores{2} = tensor(global_cores{2});
    global_cores{3} = double(global_cores{3});
    c = ttm(global_cores{2}, global_cores{1}, 1);
    gd = ttm(c, global_cores{3}', 3);
    gd = tensor(gd);
    G_clients{i}{2} = double(gd);
end

for i = 1:M
    ratio = computeRatio(G_clients{i}{1}, G_clients{i}{2}, G_clients{i}{3});
    X_clients{i} = ratio * tnprod(G_clients{i}) + (1 - ratio) * X_clients{i};
end

RSE_clients = zeros(1, M);

for i = 1:M
    numerator = norm(F_clients{i}(:) - X_clients{i}(:), 'fro'); % Frobenius norm of the difference
    denominator = norm(F_clients{i}(:), 'fro'); % Frobenius norm of the original tensor
    RSE_clients(i) = numerator / denominator;
end

disp('RSE for each client:');
disp(RSE_clients);
