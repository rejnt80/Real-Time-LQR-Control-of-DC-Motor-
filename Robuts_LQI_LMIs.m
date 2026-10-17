
%% Robust discrete-time LQI using MATLAB LMI Lab
% Five first-order DC motor models
% Common quadratic stability + guaranteed LQR cost bound
% Decision variables: X, Y, Z
% Controller: u(k) = -F*[x(k); eta(k)]

clear; clc;

%% 1. Motor data
OP  = [60; 100; 140; 180; 220];
Kdc = [28.50; 10.21; 4.84; 3.14; 2.09];
tau = [0.02582; 0.02246; 0.02304; 0.02045; 0.02333];

Ts = 0.01;                 % Sampling time (s)
N  = length(Kdc);

%% 2. Exact zero-order-hold discrete models
% x(k+1) = Ad*x(k) + Bd*u(k)
% y(k)   = x(k)

Ad = zeros(1,1,N);
Bd = zeros(1,1,N);

for i = 1:N
    a = exp(-Ts/tau(i));
    b = Kdc(i)*(1-a);

    Ad(:,:,i) = a;
    Bd(:,:,i) = b;
end

%% 3. Integral augmentation
% eta(k+1) = eta(k) + Ts*(r(k)-y(k))

n = 2;                      % Augmented state dimension
m = 1;                      % Input dimension

A = cell(N,1);
B = cell(N,1);

for i = 1:N
    A{i} = [Ad(:,:,i), 0;
            -Ts,       1];

    B{i} = [Bd(:,:,i);
             0];
end

%% 4. Discrete LQR cost
% J = sum [z'*Q*z + u'*R*u]
% Tune these weights for your motor and PWM limits.

Q = diag([0.000037, 0.00128]);
R = 0.01;

% Qh*Qh' = Q
Qh = chol(Q,'lower');
Rh = sqrt(R);

%% 5. Define LMI decision variables
% X = X' > 0, size n-by-n
% Y = F*X, size m-by-n
% Z = Z' > X^(-1)

setlmis([]);

X = lmivar(1,[n 1]);
Y = lmivar(2,[m n]);
Z = lmivar(1,[n 1]);

%% 6. Guaranteed-cost LMI for each of the five models
%
% H_i = [ X,    W_i',  X*Qh,   Y'*Rh;
%         W_i,  X,     0,      0;
%         Qh'*X,0,     I,      0;
%         Rh*Y, 0,     0,      I ] > 0
%
% W_i = A_i*X - B_i*Y
%
% LMI Lab uses L(x) < R(x), so encode -H_i < 0.

for i = 1:N

    ell = i;

    % Block (1,1): -X
    lmiterm([ell 1 1 X],-1,1);

    % Block (2,1): -W_i = -A_i*X + B_i*Y
    lmiterm([ell 2 1 X],-A{i},1);
    lmiterm([ell 2 1 Y], B{i},1);

    % Block (2,2): -X
    lmiterm([ell 2 2 X],-1,1);

    % Block (1,3): -X*Qh
    lmiterm([ell 1 3 X],-1,Qh);

    % Block (3,3): -I
    lmiterm([ell 3 3 0],-eye(n));

    % Block (4,1): -Rh*Y
    lmiterm([ell 4 1 Y],-Rh,1);

    % Block (4,4): -I
    lmiterm([ell 4 4 0],-eye(m));

end

%% 7. Positivity constraint: X > 0
ellX = N+1;

lmiterm([ellX 1 1 X],-1,1);
lmiterm([ellX 1 1 0],1e-8*eye(n));

%% 8. Cost-bound LMI:
% [ Z  I ] > 0
% [ I  X ]
%
% This implies Z > inv(X) = P.
% Encode the negative of this matrix as an LMI.

ellZ = N+2;

lmiterm([ellZ 1 1 Z],-1,1);
lmiterm([ellZ 1 2 0],-eye(n));
lmiterm([ellZ 2 2 X],-1,1);

%% 9. Minimize trace(Z) using mincx
LMIs = getlmis;

% Variables were declared in the order X, Y, Z.
% Select trace(Z) as the linear objective.
c = mat2dec(LMIs, zeros(n), zeros(m,n), eye(n));

options = [1e-7, 200, 1e8, 0, 0];

[costBound,xopt] = mincx(LMIs,c,options);

if isempty(xopt) || ~isfinite(costBound)
    error(['LMI optimization failed. Check feasibility, ', ...
           'weights, scaling, and solver diagnostics.']);
end

%% 10. Recover the optimal matrices
Xopt = dec2mat(LMIs,xopt,X);
Yopt = dec2mat(LMIs,xopt,Y);
Zopt = dec2mat(LMIs,xopt,Z);

% Symmetrize to remove numerical roundoff
Xopt = (Xopt+Xopt')/2;
Zopt = (Zopt+Zopt')/2;

% Recover F = Y/X without explicitly computing inv(X)
F = Yopt/Xopt;

Kx = F(1)
Ki = F(2)

%% 11. Display the controller and optimization results
fprintf('\n========================================\n');
fprintf('ROBUST DISCRETE-TIME LQI DESIGN\n');
fprintf('========================================\n');

fprintf('Sampling time Ts = %.6f s\n',Ts);
fprintf('Kx = %.10g\n',Kx);
fprintf('Ki = %.10g\n',Ki);
fprintf('Guaranteed cost bound trace(Z) = %.10g\n', ...
        costBound);

disp('X ='); disp(Xopt);
disp('Y ='); disp(Yopt);
disp('Z ='); disp(Zopt);
disp('Feedback gain F = [Kx Ki] ='); disp(F);

%% 12. Verify stability for all five models
fprintf('\nClosed-loop eigenvalues:\n');

stableAll = true;

for i = 1:N

    Acl = A{i} - B{i}*F;
    poles = eig(Acl);

    fprintf('\nOperating point %g:\n',OP(i));
    disp(poles.');

    fprintf('Spectral radius = %.8f\n',max(abs(poles)));

    if max(abs(poles)) >= 1
        stableAll = false;
        warning('Closed-loop stability failed at OP = %g.',OP(i));
    end

end

if stableAll
    fprintf('\nAll five discrete models are stable.\n');
else
    fprintf('\nThe common stability requirement is NOT verified.\n');
end

%% 13. Optional: print all discrete plant coefficients
fprintf('\nDiscrete plant coefficients:\n');
fprintf(' OP       a_i              b_i\n');

for i = 1:N
    fprintf('%3g   %12.8f   %12.8f\n', ...
        OP(i),Ad(:,:,i),Bd(:,:,i));
end
