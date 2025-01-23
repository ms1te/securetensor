% Parameter settings
epsilon1 = 0.01;  % Initial truncation SVD accuracy
epsilon2 = 0.1;   % TT-SVD accuracy
max_rank = 10;    % Maximum allowable rank
L = 10;           % Average consistency iteration times
p = 0.4;          % Probability of connection is 0.4
K = 5;
G_k = ctt_dec(K, p, DECdata, epsilon1, epsilon2, max_rank, L);

global_cores = G_k{1};
% original_tensor = X_all{1}; % Original tensor
global_cores{1} = squeeze(global_cores{1});
global_cores{2} = tensor(global_cores{2});
global_cores{3} = double(global_cores{3});
c = ttm(global_cores{2}, global_cores{1}, 1);
gd = ttm(c, global_cores{3}', 3);
gd = tensor(gd);
