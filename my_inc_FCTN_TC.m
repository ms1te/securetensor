%  'inc_FCTN_TC.m' supposes that the size of the factor G_k is 
%  R_{1,k}*R_{2,k}*...*R_{k-1,k}*I_k*R_{k,k+1}*...*R_{k,N}. 
%  (consistent with the paper)

function [X, G, Out] = my_inc_FCTN_TC(C,F,Omega,opts)
if isfield(opts, 'tol');         tol   = opts.tol;              end
if isfield(opts, 'maxit');       maxit = opts.maxit;            end
if isfield(opts, 'rho');         rho   = opts.rho;              end
if isfield(opts, 'R');           R     = opts.R;                end
if isfield(opts, 'max_R');       max_R     = opts.max_R;        end

N = ndims(F); 
Nway = size(F);
X = F;
tempdim = diag(Nway)+R+R';
max_tempdim = diag(Nway)+max_R+max_R';

G = cell(1,N);
for i = 1:N
    rng(0); % 固定随机种子
    G{i} = rand(tempdim(i,:));
end

Out.RSE = [];

r_change =0.01;
G{1}=C;
for k = 1:maxit
    Xold = X;
    % Update G 
    for i = 2:N
        Xi = my_Unfold(X,Nway,i);
        Gi = my_Unfold(G{i},tempdim(i,:),i);
        Girest = tnreshape(tnprod_rest(G,i),N,i);
        
        tempC = Xi*Girest'+rho*Gi;
        tempA = Girest*Girest'+rho*eye(size(Gi,2));
        G{i}  = my_Fold(tempC*pinv(tempA),tempdim(i,:),i);  
    end
    
    % Update X 
    X = (tnprod(G)+rho*Xold)/(1+rho);
    X(Omega) = F(Omega);
    
    %% check the convergence
    rse=norm(X(:)-Xold(:))/norm(Xold(:));
    Out.RSE = [Out.RSE,rse];
    
    if mod(k, 5) == 0  ||   k == 1 
       fprintf('inc_FCTN-TC: iter = %d   RSE=%f   \n', k, rse);
    end
    
    if rse < tol 
        break;
    end
    
    rank_inc=double(tempdim<max_tempdim);
    if rse<r_change && sum(rank_inc(:))~=0
    G = rank_inc_adaptive(G,rank_inc,N);
    tempdim = tempdim+rank_inc;
    r_change = r_change*0.5;
    end
end
end


function [G]=rank_inc_adaptive(G,rank_inc,N)
    % increase the estimated rank
    for j = 1:N
    G{j} = padarray(G{j},rank_inc(j,:),rand(1),'post');
    end
end
    

