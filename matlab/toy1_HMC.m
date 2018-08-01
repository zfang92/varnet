
clear, clc;

%--------------------------------------------------------------------------
% (Relatively) fixed global parameters

mu1 = [0.2 0.6];
mu2 = [0.6 0.2];
sig = 0.04;
S0 = [0.7 0.1];
Rf = 1;

%--------------------------------------------------------------------------
% Tunable global parameters

eps = 0.01;
L = 100;
niter = 1e4;
m = [1 1];

% Switches

SA = 1;
CreateGIF = 0;

%--------------------------------------------------------------------------
% Optimization usting Hamiltonian Monte Carlo

S = zeros(niter+1,2);
S(1,:) = S0;

if SA == 1
%     T = 0.9.^(linspace(0,niter-1,niter));
    T = exp(-1e0*linspace(0,niter-1,niter));
else
    T = ones(1,niter);
end

Acceptance = 0;
Downhill = 0;

Action = zeros(niter+1,1);
Action(1) = A(S(1,1),S(1,2),Rf,mu1,mu2,sig);

S_min = S(1,:);
Action_min = Action(1);

tic;
fprintf('Running... ');
for n = 2:niter+1
    [S(n,:),Action(n),flag] = HMC(Rf,mu1,mu2,sig,S(n-1,:),Action(n-1),eps,L,m,T(n-1));
    
    Acceptance = Acceptance + flag;
    
    if Action(n) < Action_min
        Action_min = Action(n);
        S_min = S(n,:);
        Downhill = Downhill+1;
    end
end
fprintf('done. ');
toc;

Acceptance = Acceptance/niter;
Downhill = Downhill/niter;

figure;
PlotA(Rf,mu1,mu2,sig);

if SA == 0
    plot(S(1:end,1),S(1:end,2),'r.');
    hold off;
else
    plot(S(1:end,1),S(1:end,2),'rp','MarkerSize',6,'MarkerFaceColor','r');
    hold off;
end

subplot(1,2,1); xlim([0 niter+1]);
loglog(Action); grid on;
ylabel('Action'); xlabel('Iteration');

%--------------------------------------------------------------------------
% Plot and save to make animated GIF

if CreateGIF == 1
    nframes = 20;
    Rf_temp = linspace(0,1,nframes);
    
    figure;
    for i = 1:nframes
        PlotA(Rf_temp(i),mu1,mu2,sig);
        saveas(gcf,['C:\Users\zfang\Downloads\' num2str(i) '.png']);
    end
end

%--------------------------------------------------------------------------
% Functions to evaluate the Action and its derivatives

function Action = A(X,Y,Rf,mu1,mu2,sig)

Action = 1 + (X-mu1(1)).^2 + (Y-mu1(2)).^2 ...
    - Rf*(exp(-((X-mu1(1)).^2+(Y-mu1(2)).^2)/(2*sig^2)) ...
    + exp(-((X-mu2(1)).^2+(Y-mu2(2)).^2)/(2*sig^2)));

end

function x_deriv = dxA(x,y,Rf,mu1,mu2,sig)

x_deriv = Rf*(exp(-((x-mu1(1))^2+(y-mu1(2))^2)/(2*sig^2))*(x-mu1(1))/sig^2 ...
    + exp(-((x-mu2(1))^2+(y-mu2(2))^2)/(2*sig^2))*(x-mu2(1))/sig^2) ...
    + 2*(x-mu1(1));

end

function y_deriv = dyA(x,y,Rf,mu1,mu2,sig)

y_deriv = Rf*(exp(-((x-mu1(1))^2+(y-mu1(2))^2)/(2*sig^2))*(y-mu1(2))/sig^2 ...
    + exp(-((x-mu2(1))^2+(y-mu2(2))^2)/(2*sig^2))*(y-mu2(2))/sig^2) ...
    + 2*(y-mu1(2));

end

%--------------------------------------------------------------------------
% Function to propose a state by the moethod of Hamiltonian Monte Carlo

function [q_new,Action_new,flag] = HMC(Rf,mu1,mu2,sig,q0,Action_current,eps,L,m,T)

q = q0;
p0 = [normrnd(0,m(1)) normrnd(0,m(2))];

p = p0 - eps*[dxA(q(1),q(2),Rf,mu1,mu2,sig) dyA(q(1),q(2),Rf,mu1,mu2,sig)]/2;

for i = 1:L
    q = q + eps*[p(1)/m(1) p(2)/m(2)];

    if i~=L
        p = p - eps*[dxA(q(1),q(2),Rf,mu1,mu2,sig) dyA(q(1),q(2),Rf,mu1,mu2,sig)];
    end
end

p = p - eps*[dxA(q(1),q(2),Rf,mu1,mu2,sig) dyA(q(1),q(2),Rf,mu1,mu2,sig)]/2;

Action_candidate = A(q(1),q(2),Rf,mu1,mu2,sig);

if rand <= exp((Action_current+sum(p0.^2./(2*m)) - Action_candidate-sum(p.^2./(2*m)))/T)
    q_new = q;
    Action_new = Action_candidate;
    flag = 1;
else
    q_new = q0;
    Action_new = Action_current;
    flag = 0;
end

end

%--------------------------------------------------------------------------
% Function to plot Action and exp(-Action)

function PlotA(Rf,mu1,mu2,sig)

%                   ------------Action------------
[X,Y] = meshgrid(linspace(0,1,500),linspace(0,1,500));
Action = A(X,Y,Rf,mu1,mu2,sig);
    
subplot(1,2,1);
        
s1 = surf(X,Y,Action);
xlim([0 1]), ylim([0 1]); zlim([0 max(max(Action))]);

shading interp;
light;
lighting gouraud;
    
s1.CData = hypot(X,Y);            % set color data
s1.AlphaData = gradient(Action);  % set vertex transparencies
s1.FaceAlpha = 'interp';          % interpolate to get face transparencies

xlabel('x'); ylabel('y'); zlabel('Action');
%                   ---------exp(-Action)---------
[X,Y] = meshgrid(linspace(-2,2,500),linspace(-2,2,500));
phi = exp(-A(X,Y,Rf,mu1,mu2,sig));
    
subplot(1,2,2);

% The hold below needs to be turned off after subplot(1,2,2) is used twice.
s2 = surf(X,Y,phi); hold on;
xlim([-2 2]), ylim([-2 2]); zlim([0 1]);
    
shading interp;
light;
lighting gouraud;
alpha(1);
    
s2.CData = hypot(X,Y);            % set color data
s2.AlphaData = gradient(phi);     % set vertex transparencies
s2.FaceAlpha = 'interp';          % interpolate to get face transparencies

xlabel('x'); ylabel('y'); zlabel('exp(- Action)');
%                   ------------------------------
set(gcf,'position',[300 300 1200 500]);

end
















