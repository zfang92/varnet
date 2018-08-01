
clear, clc;

Rf = 1;

load('toy1_HMC_SA.mat');

[r,idx,ic] = unique(S,'stable','rows');

figure;
for i = 1:length(idx)
    PlotA(r,idx,Action,Rf,i,mu1,mu2,sig,niter);
end

%--------------------------------------------------------------------------
% Functions to evaluate the Action

function Action = A(X,Y,Rf,mu1,mu2,sig)

Action = 1 + (X-mu1(1)).^2 + (Y-mu1(2)).^2 ...
    - Rf*(exp(-((X-mu1(1)).^2+(Y-mu1(2)).^2)/(2*sig^2)) ...
    + exp(-((X-mu2(1)).^2+(Y-mu2(2)).^2)/(2*sig^2)));

end

%--------------------------------------------------------------------------
% Function to plot Action and exp(-Action)

function PlotA(r,idx,Avalue,Rf,order,mu1,mu2,sig,niter)

%                   ------------Action------------
[X,Y] = meshgrid(linspace(0,1,500),linspace(0,1,500));
Action = A(X,Y,Rf,mu1,mu2,sig);
    
subplot(2,2,1);
        
s1 = surf(X,Y,Action); hold on;
plot(r(order,1),r(order,2),'rp','MarkerSize',8,'MarkerFaceColor','r');
hold off;
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
    
subplot(2,2,2);
    
s2 = surf(X,Y,phi); hold on;
plot(r(order,1),r(order,2),'rp','MarkerSize',8,'MarkerFaceColor','r');
hold off;
xlim([-2 2]), ylim([-2 2]); zlim([0 1]);
    
shading interp;
light;
lighting gouraud;
alpha(1);
    
s2.CData = hypot(X,Y);         % set color data
s2.AlphaData = gradient(phi);  % set vertex transparencies
s2.FaceAlpha = 'interp';       % interpolate to get face transparencies

xlabel('x'); ylabel('y'); zlabel('exp(- Action)');
%                   -------A vs. interation-------
subplot(2,2,[3 4]);
loglog(Avalue(1:idx(order)),'r-','LineWidth',1.5); grid on;
xlim([0 niter+1]); ylim([1e-8 1e2]);
ylabel('Action'); xlabel('Iteration');
%                   ------------------------------
set(gcf,'position',[500 50 1100 950]);
saveas(gcf,['C:\Users\zfang\Downloads\' num2str(order) '.png']);

end









