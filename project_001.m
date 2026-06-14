function cr_final_subplots()

clc; clear; close all;

%% ================= PARAMETERS =================
N = 1024;
NumTimeSlots = 200;
Fs = 1e6;

NoisePower = 1;
SNR_dB = -10;
SignalPower = 10^(SNR_dB/10)*NoisePower;

P01 = 0.1;
P10 = 0.3;

rho = 1.02;
numBands = 4;
interference_radius = 30;

%% ================= STORAGE =================
PU_state = zeros(1,NumTimeSlots);
SU_tx = zeros(1,NumTimeSlots);

energy_vec = zeros(1,NumTimeSlots);
threshold_vec = zeros(1,NumTimeSlots);

PU_band = zeros(numBands, NumTimeSlots);
SU_band_matrix = zeros(numBands, NumTimeSlots);

noise_history = [];
PU_prev = 0;

%% ================= MAIN LOOP =================
for t = 1:NumTimeSlots

    %% --- MARKOV PU ---
    if PU_prev == 0
        PU_state(t) = rand < P01;
    else
        PU_state(t) = ~(rand < P10);
    end
    PU_prev = PU_state(t);

    %% --- SIGNAL ---
    time = (0:N-1)/Fs;

    if PU_state(t)==1
        pu_base = cos(2*pi*20e3*time) + cos(2*pi*80e3*time);
        pu_signal = sqrt(SignalPower/mean(pu_base.^2))*pu_base;
    else
        pu_signal = zeros(1,N);
    end

    %% --- CHANNEL ---
    h = (randn + 1j*randn)/sqrt(2);
    noise_pow = NoisePower*(1 + (rho-1)*(2*rand-1));
    noise = sqrt(noise_pow/2)*(randn(1,N)+1j*randn(1,N));

    rx = h*pu_signal + noise;

    %% --- ENERGY DETECTION ---
    energy = mean(abs(rx).^2);
    energy_vec(t) = energy;

    %% --- CFAR THRESHOLD ---
    if PU_state(t)==0
        noise_history = [noise_history energy];
    end

    if length(noise_history)>20
        mu = mean(noise_history);
        sigma = std(noise_history);
        Threshold = mu + 2.5*sigma;
    else
        Threshold = 1.5*NoisePower;
    end

    threshold_vec(t) = Threshold;

    %% --- MULTI-BAND FFT ---
    Y = fft(rx);
    PSD = abs(Y).^2/N;

    band_size = floor(N/numBands);
    band_energy = zeros(1,numBands);

    for b=1:numBands
        idx1=(b-1)*band_size+1;
        idx2=b*band_size;
        band_energy(b)=mean(PSD(idx1:idx2));
    end

    band_free = band_energy < Threshold;

    %% --- PU PER BAND ---
    for b = 1:numBands
        if PU_state(t)==1 && band_energy(b) > Threshold
            PU_band(b,t) = 1;
        else
            PU_band(b,t) = 0;
        end
    end

    %% --- SU DECISION + BAND SELECTION ---
    if any(band_free)
        selected_band = find(band_free,1);
        detected_free = 1;
    else
        detected_free = 0;
        selected_band = 0;
    end

    %% --- DISTANCE CONDITION ---
    distance = rand*100;

    if detected_free==1 && distance>interference_radius
        SU_tx(t)=1;
        SU_band_matrix(selected_band,t)=1;
    else
        SU_tx(t)=0;
    end

end

%% ================= 1️⃣ SUBPLOTS (PU vs SU PER BAND) =================
figure;

for b = 1:numBands
    subplot(2,2,b);

    stairs(PU_band(b,:), 'r', 'LineWidth', 1.5); hold on;
    stairs(SU_band_matrix(b,:), 'b', 'LineWidth', 1.5);

    ylim([-0.2 1.2]);
    grid on;

    title(['Band ', num2str(b), ' Activity']);
    xlabel('Time Slot');
    if b == 1
        legend('PU','SU');
    end
end

xlabel('Time Slot');

%% ================= 2️⃣ ENERGY vs THRESHOLD =================
figure;
plot(energy_vec,'b','LineWidth',1.5); hold on;
plot(threshold_vec,'r','LineWidth',1.5);
legend('Energy','Threshold');
title('Energy vs Adaptive Threshold');
xlabel('Time Slot'); grid on;

%% ================= 3️⃣ ROC =================
thresholds = linspace(min(energy_vec), max(energy_vec), 40);

Pd = zeros(size(thresholds));
Pfa = zeros(size(thresholds));

for i=1:length(thresholds)
    th = thresholds(i);
    decision = energy_vec > th;

    H1 = PU_state==1;
    H0 = PU_state==0;

    Pd(i) = sum(decision(H1)==1)/sum(H1);
    Pfa(i) = sum(decision(H0)==1)/sum(H0);
end

figure;
plot(Pfa,Pd,'LineWidth',2);
xlabel('Pfa'); ylabel('Pd');
title('ROC Curve');
grid on;

%% ================= 4️⃣ BAND ALLOCATION =================
figure;
stairs(sum(SU_band_matrix.*(1:numBands)',1),'LineWidth',1.5);
xlabel('Time Slot');
ylabel('Band Index');
title('Selected Band by SU');
grid on;

end
