%% EXPERIMENT 1 - Learn a 128-tap echo path using LMS

%% Step 1 - Set up the experiment and load the echo path
clearvars;
rng(2101, 'twister');  % Repeat the same random experiment on every run.

scriptFolder = fileparts(mfilename('fullpath'));

candidates = {fullfile(scriptFolder, 'h128.mat'), ...
    fullfile(scriptFolder, 'h128(1).mat'), ...
    fullfile(scriptFolder, 'data', 'h128.mat'), ...
    fullfile(scriptFolder, 'data', 'h128(1).mat')};
echoPathFile = '';
for k = 1:numel(candidates)
    if isfile(candidates{k})
        echoPathFile = candidates{k};
        break;
    end
end
assert(~isempty(echoPathFile), ...
    'Put h128.mat or h128(1).mat in the same folder as this script.');

data = load(echoPathFile, 'h');
h = data.h(:);  % Make the 128 coefficients a column.
L = 128;
assert(numel(h) == L && isreal(h) && all(isfinite(h)) && any(h ~= 0), ...
    'The echo path must contain 128 finite real coefficients with nonzero energy.');

%% Step 2 - Generate 30,000 input samples
N = 30000;
x = randn(N, 1);
% randn has theoretical mean 0 and variance 1. A finite recording's
% measured mean will be close to 0 rather than necessarily exactly 0.

%% Step 3 - Simulate the clean echo from the room
cleanEcho = filter(h, 1, x);
% This calculates h(1)*x(n) + h(2)*x(n-1) + ... + h(128)*x(n-127).
% Input samples before the recording begins are assumed to be zero.

%% Step 4 - Add a separate background noise sequence at 30 dB SNR
targetSNR = 30;
backgroundNoise = randn(N, 1);  % A new random sequence, separate from x.
backgroundNoise = backgroundNoise - mean(backgroundNoise);

echoPower = mean(cleanEcho.^2);
requiredNoisePower = echoPower / 10^(targetSNR / 10);
backgroundNoise = backgroundNoise * ...
    sqrt(requiredNoisePower / mean(backgroundNoise.^2));

d = cleanEcho + backgroundNoise;
measuredSNR = 10 * log10(echoPower / mean(backgroundNoise.^2));

fprintf('\nExperiment 1 - LMS echo-path identification\n');
fprintf('Samples = %d, filter length = %d\n', N, L);
fprintf('Measured input mean = %.5f\n', mean(x));
fprintf('Measured microphone SNR = %.2f dB\n', measuredSNR);

%% Step 5 - Choose three fixed LMS step sizes
% Project Eq. (3): mu < 2 / sum of squares of the current input vector.
% Find the largest 128-sample input energy in this particular recording.
tapInputEnergy = filter(ones(L, 1), 1, x.^2);
muUpperBound = 2 / max(tapInputEnergy);
muValues = muUpperBound * [0.04, 0.20, 0.70];
% These fractions are chosen experimental settings. All three satisfy
% Eq. (3) for every observed input window. General mean-square stability
% also depends on the input statistics; check the resulting curves.
fprintf('Step sizes, small to large: %.6f, %.6f, %.6f\n', muValues);

%% Step 6 - Run LMS three times and measure the coefficient error
% Each trial uses exactly the same x and d, starting from zero weights.
errorSignals = zeros(N, 3);
finalWeights = zeros(L, 3);
weightErrorRatio = zeros(N + 1, 3);
labels = cell(1, 3);

for trial = 1:3
    mu = muValues(trial);
    [e, wFinal, weightHistory] = lms_filter(x, d, mu, L);

    errorSignals(:, trial) = e;
    finalWeights(:, trial) = wFinal;

    % Project Eq. (4): compare learned coefficients with the true path.
    % Column 1 of weightHistory is the initial zero vector.
    coefficientError = weightHistory - h;
    weightErrorRatio(:, trial) = sum(coefficientError.^2, 1).' / sum(h.^2);
    labels{trial} = sprintf('mu = %.6f', mu);
end
wevnDB = 10 * log10(max(weightErrorRatio, realmin));
assert(all(isfinite(wevnDB(:))), 'The LMS filter diverged.');
clear weightHistory coefficientError;  % Release the temporary history arrays.
samplesProcessed = (0:N).';

%% Step 7 - Plot the learning curves
% A downward curve indicates learning. A more negative WEVN is better.
% Use plot for a dB value, and semilogy for a positive linear error ratio.
fig1 = figure('Color', 'w', 'Name', 'Experiment 1 - Learning curves');
set(gcf, 'Position', [80 80 1050 760], ...
    'DefaultAxesFontSize', 11, 'DefaultTextFontSize', 11, ...
    'DefaultAxesColorOrder', [0 0.35 0.65; 0.85 0.25 0.08; 0.45 0.15 0.65]);
subplot(2, 1, 1);
plot(samplesProcessed, wevnDB, 'LineWidth', 1.2);
grid on;
xlabel('Input samples processed');
ylabel('WEVN (dB)');
title('LMS learning speed and final coefficient error');
legend(labels, 'Location', 'best');

subplot(2, 1, 2);
semilogy(samplesProcessed, max(weightErrorRatio, realmin), 'LineWidth', 1.2);
grid on;
xlabel('Input samples processed');
ylabel('Normalized squared weight error');
title('The same learning curves on a logarithmic scale');
legend(labels, 'Location', 'best');

%% Step 8 - Plot the residual error for each step size
% e(n) contains the remaining echo and background noise.
% The noise remains even when the echo path has been learned accurately.
fig2 = figure('Color', 'w', 'Name', 'Experiment 1 - Residual error');
set(gcf, 'Position', [80 80 1050 760], ...
    'DefaultAxesFontSize', 11, 'DefaultTextFontSize', 11, ...
    'DefaultAxesColorOrder', [0 0.35 0.65; 0.85 0.25 0.08; 0.45 0.15 0.65]);
errorLimit = 1.05 * max(abs(errorSignals(:)));
for trial = 1:3
    subplot(3, 1, trial);
    plot(1:N, errorSignals(:, trial));
    grid on;
    ylim([-errorLimit, errorLimit]);
    xlabel('Sample index');
    ylabel('e(n)');
    title(labels{trial});
end

% Also display squared residual error on a logarithmic axis.
% Squaring handles negative residuals; realmin protects any exact zeros.
fig4 = figure('Color', 'w', 'Name', 'Experiment 1 - Squared residual error');
set(gcf, 'Position', [80 80 1050 760], ...
    'DefaultAxesFontSize', 11, 'DefaultTextFontSize', 11, ...
    'DefaultAxesColorOrder', [0 0.35 0.65; 0.85 0.25 0.08; 0.45 0.15 0.65]);
for trial = 1:3
    subplot(3, 1, trial);
    semilogy(1:N, max(errorSignals(:, trial).^2, realmin));
    grid on;
    xlabel('Sample index');
    ylabel('Squared residual e(n)^2');
    title(labels{trial});
end

%% Step 9 - Compare learned coefficients with the true echo path
fig3 = figure('Color', 'w', 'Name', 'Experiment 1 - Coefficients');
set(gcf, 'Position', [80 80 1050 760], ...
    'DefaultAxesFontSize', 11, 'DefaultTextFontSize', 11, ...
    'DefaultAxesColorOrder', [0 0.35 0.65; 0.85 0.25 0.08; 0.45 0.15 0.65]);
for trial = 1:3
    subplot(3, 1, trial);
    plot(0:L-1, h, 'k', 'LineWidth', 1.2);
    hold on;
    plot(0:L-1, finalWeights(:, trial), 'r--', 'LineWidth', 1);
    grid on;
    xlabel('Tap index');
    ylabel('Coefficient');
    title(labels{trial});
    legend('True echo path h', 'LMS estimate w', 'Location', 'best');
end

%% Step 10 - Print a comparison table and save the results
% Compare the final 3,000 updates (last 10% of the recording).
% The tail mean below averages WEVN values in dB.
% Call it steady state only if the corresponding curve has settled.
tailStartRow = N - 3000 + 2;  % Row n+1 contains weights after update n.
meanTailWEVN = mean(wevnDB(tailStartRow:end, :), 1).';
finalWEVN = wevnDB(end, :).';
lowestWEVN = min(wevnDB(2:end, :), [], 1).';
% Report the first time each filter reaches a common accuracy threshold.
% This is a threshold-crossing measure, not a settling-time estimate.
thresholdDB = -20;
samplesToFirstReach = nan(3, 1);
for trial = 1:3
    firstRow = find(wevnDB(:, trial) <= thresholdDB, 1, 'first');
    if ~isempty(firstRow)
        % Row 1 corresponds to zero samples processed.
        samplesToFirstReach(trial) = samplesProcessed(firstRow);
    end
end
comparison = table(muValues(:), samplesToFirstReach, lowestWEVN, finalWEVN, meanTailWEVN, ...
    'VariableNames', {'StepSize', 'SamplesToFirstReachMinus20dB', ...
    'LowestWEVN_dB', 'FinalWEVN_dB', 'MeanLast3000WEVN_dB'});
fprintf('NaN in the threshold column means -20 dB was not reached.\n');
disp(comparison);

resultsFolder = fullfile(scriptFolder, 'results', 'experiment_1');
if ~isfolder(resultsFolder)
    mkdir(resultsFolder);
end
writetable(comparison, fullfile(resultsFolder, 'comparison.csv'));
print(fig1, fullfile(resultsFolder, 'learning_curves.png'), '-dpng', '-r200');
savefig(fig1, fullfile(resultsFolder, 'learning_curves.fig'));
print(fig2, fullfile(resultsFolder, 'residual_error.png'), '-dpng', '-r200');
savefig(fig2, fullfile(resultsFolder, 'residual_error.fig'));
print(fig3, fullfile(resultsFolder, 'coefficients.png'), '-dpng', '-r200');
savefig(fig3, fullfile(resultsFolder, 'coefficients.fig'));
print(fig4, fullfile(resultsFolder, 'residual_error_semilogy.png'), '-dpng', '-r200');
savefig(fig4, fullfile(resultsFolder, 'residual_error_semilogy.fig'));
save(fullfile(resultsFolder, 'experiment_1_results.mat'), ...
    'x', 'h', 'd', 'cleanEcho', 'backgroundNoise', 'measuredSNR', ...
    'muValues', 'muUpperBound', 'wevnDB', 'weightErrorRatio', ...
    'errorSignals', 'finalWeights', 'thresholdDB', 'samplesToFirstReach', 'comparison');
fprintf('Figures and numerical results saved in %s\n', resultsFolder);

%% LMS function - Experiment 1(a)
function [e, wFinal, weightHistory] = lms_filter(x, d, mu, L)
% Required inputs: input x, microphone signal d, step size mu, length L.
% Required outputs: error e and final coefficient vector wFinal.
% Extra output: coefficient history for evaluating learning outside LMS.
x = x(:);
d = d(:);
assert(numel(x) == numel(d), 'x and d must have the same length.');
N = numel(x);
e = zeros(N, 1);
wFinal = zeros(L, 1);
inputVector = zeros(L, 1);
weightHistory = zeros(L, N + 1);

for n = 1:N
    % Put the newest input first, followed by L-1 previous samples.
    inputVector = [x(n); inputVector(1:end-1)];

    % Predict the echo with the current coefficients.
    predictedEcho = wFinal.' * inputVector;

    % Project Eq. (1): microphone minus predicted echo.
    e(n) = d(n) - predictedEcho;

    % Project Eq. (2): learn from the current error.
    wFinal = wFinal + (mu * e(n)) * inputVector;
    weightHistory(:, n+1) = wFinal;
end
end
