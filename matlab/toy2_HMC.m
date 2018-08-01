%%

% This is a 2D toy model for doing logistic regression with the method of
% Hamiltonian Monte Carlo. It contains only one 
% state variable and one model parameter. In addition, 
% Simulated Annealing algorithm can be turned on or off.

clear, clc;

Rf = 1e0; %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
y = 0.5; %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

A = @(X,W) (X-y).^2 + Rf*(X-1./(1+exp(-W*y))).^2;

dwA = @(x,w) -2*Rf*y*exp(w*y)*((x-1)*exp(w*y)+x)/(1+exp(w*y))^3;
dxA = @(x,w) 2*Rf*(x-1/(1+exp(-w*y)))+2*(x-y);

plt = 1;

if plt == 1
    f = figure;
    %-----------------------------Action-----------------------------------
    [X,W] = meshgrid(linspace(y-5,y+5,500),linspace(-5,5,500));
    Action = A(X,W);
    
    subplot(1,2,1);
    
    s1 = surf(X,W,Action); zlim([0 max(max(Action))]);
    
    s1.CData = hypot(X,W);            % set color data
    s1.AlphaData = gradient(Action);  % set vertex transparencies
    s1.FaceAlpha = 'interp';          % interpolate to get face transparencies

    shading interp;
    light;
    lighting gouraud;

    xlabel('x'); ylabel('w'); zlabel('Action');
    %--------------------------exp(-Action)--------------------------------
    [X,W] = meshgrid(linspace(y-5,y+5,500),linspace(-5,5,500));
    phi = exp(-A(X,W));
    
    subplot(1,2,2);
    
    s2 = surf(X,W,phi); zlim([0 max(max(phi))]); hold on;

    shading interp;
    light;
    lighting gouraud;
    alpha(1);
    
    s2.CData = hypot(X,W);         % set color data
    s2.AlphaData = gradient(phi);  % set vertex transparencies
    s2.FaceAlpha = 'interp';       % interpolate to get face transparencies   

    xlabel('x'); ylabel('w'); zlabel('exp(- Action)');
    %----------------------------------------------------------------------
    set(gcf,'position',[300 300 1200 500]);
end

% Plots show that the Action, as well as exp(-Action), is very insensitive
% to the change of \omaga.


%% 

niter = 1e3; %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
% nburn = floor(0.5*niter);

eps = 0.001; %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
L = 100; %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
m = [1 1]; %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

SA = 1; %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

%------------------------Tunable parameters above--------------------------

S = zeros(niter+1,2);
S(1,:) = [0 0.5]; %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

if SA == 1
%     T = 0.9.^(linspace(0,niter-1,niter));
    T = exp(-1e0*linspace(0,niter-1,niter)); %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
else
    T = ones(1,niter);
end

Acceptance = 0;
Downhill = 0;

Action = zeros(niter+1,1);
Action(1) = A(S(1,1),S(1,2));

S_min = S(1,:);
Action_min = Action(1);

tic;
fprintf('Running... ');
for n = 2:niter+1

    q = S(n-1,:);
    p0 = [normrnd(0,m(1)) normrnd(0,m(2))];
    
    p = p0 - eps*[dxA(q(1),q(2)) dwA(q(1),q(2))]/2;
    
    for i = 1:L
        q = q + eps*[p(1)/m(1) p(2)/m(2)];
        
        if i~=L
            p = p - eps*[dxA(q(1),q(2)) dwA(q(1),q(2))];
        end
    end
    
    p = p - eps*[dxA(q(1),q(2)) dwA(q(1),q(2))]/2;
    
    Action_candidate = A(q(1),q(2));
    
    if rand <= exp((Action(n-1)+sum(p0.^2./(2*m)) - Action_candidate-sum(p.^2./(2*m)))/T(n-1))
        if Action_candidate < Action_min
            Action_min = Action_candidate;
            S_min = q;
            
            Downhill = Downhill+1;
        end
        
        S(n,:) = q;
        Action(n) = Action_candidate;
        
        Acceptance = Acceptance + 1;
    else
        S(n,:) = S(n-1,:);
        Action(n) = Action(n-1);
    end
end
fprintf('done. ');
toc;

Acceptance = Acceptance/niter;
Downhill = Downhill/niter;

% plot(S(:,1),S(:,2),'r.'); hold off; %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

if SA == 0
    plot(S(1:end,1),S(1:end,2),'r.'); hold off; %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
else
    plot(S(1:end,1),S(1:end,2),'rp','MarkerSize',6,'MarkerFaceColor','r'); hold off; %<<<<<<<<<
end

subplot(1,2,1);
loglog(Action); xlim([0 niter+1]);
ylabel('Action'); xlabel('Iteration');
















