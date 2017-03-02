% Kinetic 1D Ising Model Toy Model
% Glauber dynamics

% Generate array of spins

clear all; close all; clc

N = 20;
tau = 1;
spins = round(rand(1,N));
invertspins = (spins == 0);
sigma = spins - invertspins;
% sigma = ones(1,N);


K = 1;
gamma = tanh(2*K);
% gamma = 1;

tmax = 2e2;
trials = 5000;
m = zeros(trials,tmax);
% s1si = zeros(tmax,N,trials);
s0s1 = zeros(trials,tmax);
g1 = zeros(trials, tmax);

%figure; 
%plot(sigma)

% create paramagnetic IC

for i = 1:N
    if mod(i,2) == 0
        sigma_p(i) = 1;
    else
        sigma_p(i) = -1;
    end
end

        

 h = waitbar(0,'Please wait...');
 
for j = 1:trials

%       sigma = sigma_p;  %paramagnetic i.c.
    sigma = ones(1,N);
    sigma = sigma - 2*(randi(2,1,N) - 1);
    
for t = 1:tmax
    pick = randi(N); % pick random spin
    
    % account for cyclic boundary conditions
    if pick == N
        r = (1/(2*tau))*(1 - (gamma/2)*sigma(pick)*(sigma(1) + sigma(pick-1)) );
    elseif pick == 1
        r = (1/(2*tau))*(1 - (gamma/2)*sigma(pick)*(sigma(pick+1) + sigma(N)) );
    else
        r = (1/(2*tau))*(1 - (gamma/2)*sigma(pick)*(sigma(pick+1) + sigma(pick-1)) );
    end
    
    flip = rand;
    if rand > r
        % do nothing
    elseif rand <= r
        sigma(pick) = (-1)*sigma(pick);
    end

    
    m(j,t) = (1/N)*sum(sigma);   
       
    s0s1(j,t) = sigma(1)*sigma(2);

    skskp1 = 0;
    
    for k = 1:N
        if k == N
            skskp1 = skskp1 + sigma(k)*sigma(1);
        else
            skskp1 = skskp1 + sigma(k)*sigma(k+1);
        end
    end
    
    g1(j,t) = (1/N)*skskp1;
    

    
    %plot(sigma)
    %pause(0.05)
            
end
 waitbar(j/trials,h)
end

close(h)

mean_m = mean(m);

time = linspace(0,(1/N)*tmax,tmax);

% Plot m vs time

% model = 1*exp(-(1-gamma)*(time));
% 
% figure; 
% plot(time,mean_m)
% hold on
% plot(time,model)


g1_mean = mean(g1);

rho = (1/2)*(1 - g1_mean); % rho vs. time

rho_model = (4*pi*time).^(-1/2);

% figure; 
% plot(time, rho)
% hold on; 
% plot(time, rho_model)

mean_sep =  1./(rho);

mean_sep_model = (4*pi*time).^(1/2);

figure; 
plot(time, mean_sep)
hold on; 
plot(time, mean_sep_model)

%% Fit a sqrt function to the mean separation data


g = fittype( @(a, b, c, x) a*x.^b + c );
[fit1,gof,fitinfo] = fit(time(1:end)',mean_sep(1:end)',g,'StartPoint',[(4*pi)^(1/2) 1/2 0],'Algorithm','Levenberg-Marquardt');%'Lower',[0 0 -1],'Upper',[0 4*pi 0]);
fit1 % display fit coefficients

figure; 
plot(time, mean_sep)
hold on
plot(time, fit1(time))
hold on
plot(time, 1./((1/2)*exp(-2*time).*(besseli(0,2*time) + besseli(1,2*time))))

