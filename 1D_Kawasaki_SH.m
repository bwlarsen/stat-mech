% Kinetic 1D Ising Model Toy Model
% Kawasaki dynamics

% Generate array of spins

clear all; close all; clc

% clear all; clc

N = 200;
tau = 1;
spins = round(rand(1,N));
invertspins = (spins == 0);
sigma = spins - invertspins;
% sigma = ones(1,N);


K = 0.9;
gamma = tanh(2*K);
% gamma = 1;

tmax = 1e3;
trials = 5000;
m = zeros(trials,tmax);
% s1si = zeros(tmax,N,trials);
s0s1 = zeros(trials,tmax);

%figure; 
%plot(sigma)

 h = waitbar(0,'Please wait...');

for j = 1:trials

    sigma = ones(1,N);
    sigma = sigma - 2*(randi(2,1,N) - 1);
    sigma_i = sigma;
%     sigma(1) = 1; % set initial condition
    
for t = 1:tmax
    pick = randi(N); % pick random spin
    
    % account for cyclic boundary conditions
    if pick == N
        r = (1/(2*tau))*(1 - (gamma/2)*(sigma(pick-1)*sigma(pick) + sigma(1)*sigma(2)) )*(1/2)*(1 - sigma(pick)*sigma(1));
    elseif pick == 1
        r = (1/(2*tau))*(1 - (gamma/2)*(sigma(N)*sigma(pick) + sigma(pick+1)*sigma(pick+2)) )*(1/2)*(1 - sigma(pick)*sigma(pick+1));
    elseif pick == N-1
        r = (1/(2*tau))*(1 - (gamma/2)*(sigma(pick-1)*sigma(pick) + sigma(pick+1)*sigma(1)) )*(1/2)*(1 - sigma(pick)*sigma(pick+1));
    else
        r = (1/(2*tau))*(1 - (gamma/2)*(sigma(pick-1)*sigma(pick) + sigma(pick+1)*sigma(pick+2)) )*(1/2)*(1 - sigma(pick)*sigma(pick+1));
    end
    
    flip = rand;
    if rand > r
        % do nothing
    elseif rand <= r && pick == N
        sigma(pick) = (-1)*sigma(pick);
        sigma(pick+1) = (-1)*sigma(1);
    elseif rand <= r && pick ~= N
        sigma(pick) = (-1)*sigma(pick);
        sigma(pick+1) = (-1)*sigma(pick+1);
    end

    
    m(j,t) = (1/N)*sum(sigma);
    
%     for i = 1:N
%         s1si(t,i,j) = sigma(1)*sigma(i);
%     end
    
       
%     s0s1(j,t) = sigma(1)*sigma(2);
% 
%     skskp1 = 0;
%     
%     for k = 1:N
%         if k == N
%             skskp1 = skskp1 + sigma(k)*sigma(1);
%         else
%             skskp1 = skskp1 + sigma(k)*sigma(k+1);
%         end
%     end
    
    rho = 0;
    for k = 1:N
        if k == N
            rho = rho + (1/2)*(1 - sigma(k)*sigma(1));
        else
            rho = rho + (1/2)*(1 - sigma(k)*sigma(k+1));
        end
    end
    
    rho_sum(j,t) = (1/N)*rho;
    
%     g1(j,t) = (1/N)*skskp1;
    

    
    %plot(sigma)
    %pause(0.05)
            
end
 waitbar(j/trials,h)
end

close(h)

mean_m = mean(m);

time = linspace(0,(1/N)*tmax,tmax);

%     
%    ind = 1:100:tmax;
%    mean_m_sub = mean_m(1,ind);
%     
%    figure;
%    plot(m(ind))

% 
% model = 1*exp(-(1-gamma)*(time));
% 
% figure; 
% plot(time,mean_m)
% hold on
% plot(time,model)
% 
% 
% % corrs = mean(s1si,3);
% 
% % s0s1 = corrs(:,2);
% s0s1_mean = mean(s0s1);
% 
% 
% corrmodel = 1 - 2*((4*pi*time).^(-1/2));
% 
% figure; 
% plot(time,s0s1_mean)
% hold on
% plot(time, corrmodel)
% 
% % plot domain wall density
% 
% rho = (1/2)*(1 - s0s1_mean); % rho vs. time
% 
% rho_model = (4*pi*time).^(-1/2);
% 
% rho_model_2 = exp(-2*time).*besseli(0,2*time);
% 
% figure; 
% plot(time, rho)
% hold on; 
% plot(time, rho_model)
% hold on
% plot(time, rho_model_2)
% 
% % plot mean domain wall separation
% 
% mean_sep = 2./(1 - s0s1_mean); % 1./(rho);
% 
% mean_sep_model = (4*pi*time).^(1/2);
% 
% figure; 
% plot(time, mean_sep)
% hold on; 
% plot(time, mean_sep_model)
% 
% g1_mean = mean(g1);
% 
% 
% figure; 
% plot(time, g1_mean)

rho_mean = mean(rho_sum);

% rho_test = (1/2)*(1 - g1_mean); % rho vs. time


figure; 
plot(time, rho_mean)


mean_sep =  1./(rho_mean);


%% Fit a t^(1/3) function to the mean separation data

t = 1:1:1000;
g = fittype( @(a, b, c, x) a*(x).^b + c );
[fit1,gof,fitinfo] = fit(time(10:end)',mean_sep(10:end)',g,'StartPoint',[0.87 1/3 2]);%, 'Lower',[0 0.2 0],'Upper',[100 0.4 100]);
% [fit1,gof,fitinfo] = fit(t',mean_sep(1:end)',g,'StartPoint',[0.87 1/3 2], 'Lower',[0 0.2 0],'Upper',[100 0.4 100]);
fit1

figure; 
plot(time, mean_sep)
hold on
plot(time, fit1(time))

%% 

% %% Fit a t^(-1/3) function to the density
% 
% t = 1:1:1000;
% g = fittype( @(a, b, c, d, x) a*(x+b).^c + d );
% [fit1,gof,fitinfo] = fit(time(2:end)',rho_mean(2:end)',g,'StartPoint',[0.87 0 -1/3 2 ]);%, 'Lower',[0 0.2 0],'Upper',[100 0.4 100]);
% % [fit1,gof,fitinfo] = fit(t',mean_sep(1:end)',g,'StartPoint',[0.87 1/3 2], 'Lower',[0 0.2 0],'Upper',[100 0.4 100]);
%     
% 
% figure; 
% plot(time, rho_mean)
% hold on
% plot(time, fit1(time))


