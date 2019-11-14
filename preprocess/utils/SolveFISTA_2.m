
function x_hat = SolveFISTA_2(A,b,maxIter,tolerance)


t0 = tic ;

STOPPING_TIME = -2;
STOPPING_GROUND_TRUTH = -1;
STOPPING_DUALITY_GAP = 1;
STOPPING_SPARSE_SUPPORT = 2;
STOPPING_OBJECTIVE_VALUE = 3;
STOPPING_SUBGRADIENT = 4;
STOPPING_DEFAULT = STOPPING_SUBGRADIENT;

stoppingCriterion = STOPPING_DEFAULT;

[m,n] = size(A) ;
xG = [];
QF = 1;
% Initializing optimization variables
t_k = 1 ; 
t_km1 = 1 ;
L0 = .01 ;
%L0 = 1;%%% default
G = A'*A ;
nIter = 0 ;
c = A'*b ;
lambda0 = QF*0.5*L0*norm(c,inf) ;
eta = 0.75 ;
% lambda_bar = 1e-10*lambda0 ;
lambda_bar = QF*2;
% xk = zeros(n,1) ;
%xk = A\b ;
xk = inv(A'*A)*A'*b;
%xk(Index_dct_2)=0;
lambda = lambda0 ;
L = L0 ;
beta = 1.5 ; %%% default

timeSteps = nan(1,maxIter) ;
errorSteps = nan(1,maxIter) ;

keep_going = 1 ;
nz_x = (abs(xk)> eps*10);
f = 0.5*norm(b-A*xk)^2 + lambda_bar * norm(xk,1);
xkm1 = xk;
while keep_going && (nIter < maxIter)
    nIter = nIter + 1 ;
    
    yk = xk + ((t_km1-1)/t_k)*(xk-xkm1) ;
    
    stop_backtrack = 0 ;
    
    temp = G*yk - c ; % gradient of f at yk
    
    while ~stop_backtrack
        
        gk = yk - (1/L)*temp ;
        
        xkp1 = soft(gk,lambda/L) ;
        
%         xkp1 = soft(gk,1/L) ;
        
        temp1 = 0.5*norm(b-A*xkp1)^2 ;
        temp2 = 0.5*norm(b-A*yk)^2 + (xkp1-yk)'*temp + (L/2)*norm(xkp1-yk)^2 ;
        
        if temp1 <= temp2
            stop_backtrack = 1 ;
        else
            L = L*beta ;
        end
        
    end
    
    % disp(['Iter ' num2str(nIter) ' ||x||_0 ' num2str(sum(abs(xkp1) > 0)) ' err ' num2str(norm(xG-xkp1))]) ;
    timeSteps(nIter) = toc(t0) ;

    
    switch stoppingCriterion
        case STOPPING_GROUND_TRUTH
            keep_going = norm(xG-xkp1)>tolerance;
        case STOPPING_SUBGRADIENT
            sk = L*(yk-xkp1) + G*(xkp1-yk) ;
            keep_going = norm(sk) > tolerance*L*max(1,norm(xkp1));
        case STOPPING_SPARSE_SUPPORT
            % compute the stopping criterion based on the change
            % of the number of non-zero components of the estimate
            nz_x_prev = nz_x;
            nz_x = (abs(xkp1)>eps*10);
            num_nz_x = sum(nz_x(:));
            num_changes_active = (sum(nz_x(:)~=nz_x_prev(:)));
            if num_nz_x >= 1
                criterionActiveSet = num_changes_active / num_nz_x;
                keep_going = (criterionActiveSet > tolerance);
            end
        case STOPPING_OBJECTIVE_VALUE
            % compute the stopping criterion based on the relative
            % variation of the objective function.
            prev_f = f;
            f = 0.5*norm(b-A*xkp1)^2 + lambda_bar * norm(xk,1);
            criterionObjective = abs(f-prev_f)/(prev_f);
            keep_going =  (criterionObjective > tolerance);
        case STOPPING_DUALITY_GAP
            error('Duality gap is not a valid stopping criterion for PGBP.');
        case STOPPING_TIME
            keep_going = timeSteps(nIter) < maxTime ;
        otherwise
            error('Undefined stopping criterion.');
    end
    
    lambda = max(eta*lambda,lambda_bar) ;
    
    t_kp1 = 0.5*(1+sqrt(1+4*t_k*t_k)) ;
    
    t_km1 = t_k ;
    t_k = t_kp1 ;
    
    xkm1 = xk ;
    xk = xkp1 ;
    
   % xk(Index_dct_2)=0;
    
end

x_hat = xk ;

function y = soft(x,T)
if sum(abs(T(:)))==0
    y = x;
else
    y = max(abs(x) - T, 0);
    y = sign(x).*y;
end