% Experiment 4: NLMS versus block FFT Fast-NLMS.


clearvars;
root = fileparts(mfilename('fullpath'));
resultsDir = fullfile(root, 'results', 'experiment_4');
if ~isfolder(resultsDir)
    mkdir(resultsDir);
end

fprintf('\nExperiment 4: NLMS versus Fast-NLMS\n');
rng(2104, 'twister');
h = load_echo_path(root, 'h1024.mat', 1024);
[x, fs] = audioread(data_path(root, 'long_speech.wav'));
assert(size(x, 2) == 1, 'Expected mono speech.');
L = numel(h);

% Adapt once per complete L-sample block
% Process any remaining audio with the last learned Fast-NLMS filter.
% Both algorithms receive the entire recording, without repeated speech.
nSamples = numel(x);
nCompleteBlocks = floor(nSamples / L);
assert(nCompleteBlocks > 0, 'Experiment 4 needs at least one complete block.');
fprintf('%d full blocks and %d trailing samples at %d Hz\n', ...
    nCompleteBlocks, mod(nSamples, L), fs);
% The brief leaves the Experiment 4 noise level open.
snrDb = 30;
[d, actualSnr] = noisy_microphone(x, h, snrDb);

muNlms = 0.60;
muFast = 0.50;
lambdaFast = 0.70;  % Selected setting within the lecture range 0 < lambdaF < 1.
epsilonNlms = 1e-3 * mean(filter(ones(L, 1), 1, x.^2));
initialBinPower = 2 * L * mean(x.^2);
epsilonFast = 1e-3 * initialBinPower;
fprintf('Measured input SNR %.2f dB; NLMS epsilon %.5g\n', actualSnr, epsilonNlms);
fprintf('Fast-NLMS lambda %.2f; epsilon %.5g; initial bin power %.5g\n', ...
    lambdaFast, epsilonFast, initialBinPower);

[eNlms, wNlms, wevnNlms] = run_nlms(x, d, muNlms, epsilonNlms, h);
[eFast, wFast, wevnFast] = run_fnlms(x, d, muFast, lambdaFast, ...
    epsilonFast, initialBinPower, h);
blockEnds = (L:L:nSamples).';
% For a fair block comparison, summarize only common update positions.
fastPlotSamples = [0; blockEnds];
fastPlotWevn = [0; wevnFast];
if blockEnds(end) < nSamples
    fastPlotSamples(end+1, 1) = nSamples;
    fastPlotWevn(end+1, 1) = wevnFast(end);
end
nlmsLabel = sprintf('NLMS, mu = %.2f', muNlms);
fastLabel = sprintf('Fast-NLMS, mu = %.2f', muFast);
% Compare learning speed only at common block-end sample indices.
% The -5 dB target measures first crossing, not convergence to steady state.
thresholdDb = -5;
blockCurves = [wevnNlms(blockEnds), wevnFast];
algorithmNames = {'NLMS'; 'Fast-NLMS'};
samplesToFirstReach = nan(2, 1);
for k = 1:2
    firstBlock = find(blockCurves(:, k) <= thresholdDb, 1, 'first');
    if isempty(firstBlock)
        fprintf('  %s: %.1f dB not reached at a block endpoint\n', ...
            algorithmNames{k}, thresholdDb);
    else
        samplesToFirstReach(k) = blockEnds(firstBlock);
        fprintf('  %s first reaches %.1f dB at block %d, after %d samples\n', ...
            algorithmNames{k}, thresholdDb, firstBlock, blockEnds(firstBlock));
    end
end

% Distinguish the last common block boundary from the actual end of the file.
% NLMS continues to adapt on the final partial block.
% Fast-NLMS holds its learned weights while processing that final tail.
endOfFileWevnDb = [wevnNlms(end); wevnFast(end)];
print_summary(nlmsLabel, blockCurves(:, 1), endOfFileWevnDb(1));
print_summary(fastLabel, blockCurves(:, 2), endOfFileWevnDb(2));
tailBlockCount = max(1, ceil(0.10 * numel(blockEnds)));
lowestBlockWevnDb = min(blockCurves, [], 1).';
lastBlockWevnDb = blockCurves(end, :).';
meanLastBlockWevnDb = mean(blockCurves(end-tailBlockCount+1:end, :), 1).';
comparison = table(algorithmNames, samplesToFirstReach, ...
    lowestBlockWevnDb, lastBlockWevnDb, endOfFileWevnDb, meanLastBlockWevnDb, ...
    'VariableNames', {'Algorithm', 'SamplesToFirstReachMinus5dB', ...
    'LowestBlockWEVN_dB', 'LastBlockWEVN_dB', ...
    'EndOfFileWEVN_dB', 'MeanLast10PercentBlockWEVN_dB'});
disp(comparison);
writetable(comparison, fullfile(resultsDir, 'comparison.csv'));
fprintf('NaN means the -5 dB target was not reached at a block endpoint.\n');
fprintf('The final-window mean is not assumed to represent steady state.\n');
fprintf('Lecture operation-count model (real multiplications per sample):\n');
fprintf('  NLMS %d; Fast-NLMS %d\n', 2 * L + 3, 10 * log2(L) + 36);

fig = wevn_figure(4, 'NLMS versus Fast-NLMS on the 1024-tap echo path');
plot((0:nSamples).', [0; wevnNlms], 'LineWidth', 1.2);
hold on;
plot(fastPlotSamples, fastPlotWevn, '-o', 'LineWidth', 1.2, 'MarkerSize', 3);
legend({nlmsLabel, fastLabel}, 'Location', 'best');
save_figure(fig, resultsDir, 'exp4_nlms_vs_fast_nlms.png');
% Supplementary figure using only the lecture operation-count model.
% This compares multiplication counts, not measured execution times.
complexityCounts = [2 * L + 3; 10 * log2(L) + 36];
figComplexity = figure('Color', 'w', 'Position', [100 100 850 550]);
bar(complexityCounts, 0.55);
set(gca, 'XTick', 1:2, 'XTickLabel', {'NLMS', 'Fast-NLMS'}, 'FontSize', 11);
ylabel('Real multiplications per input sample');
title('Experiment 4 - Lecture operation-count model');
grid on;
ylim([0, 1.15 * max(complexityCounts)]);
for k = 1:2
    text(k, complexityCounts(k) + 0.025 * max(complexityCounts), ...
        sprintf('%d', complexityCounts(k)), 'HorizontalAlignment', 'center');
end
save_figure(figComplexity, resultsDir, 'exp4_complexity.png');



% The lecture complexity excludes WEVN evaluation and MATLAB plotting.
% The last partial block has no Fast-NLMS update and is excluded from that count.
save(fullfile(resultsDir, 'exp4_results.mat'), ...
    'x', 'd', 'h', 'fs', 'actualSnr', 'snrDb', 'muNlms', 'muFast', ...
    'lambdaFast', 'epsilonNlms', 'epsilonFast', 'initialBinPower', ...
    'blockEnds', 'fastPlotSamples', 'fastPlotWevn', ...
    'thresholdDb', 'samplesToFirstReach', 'comparison', ...
    'eNlms', 'wNlms', 'wevnNlms', 'eFast', 'wFast', 'wevnFast');

function [e, w, wevn] = run_nlms(x, d, mu, epsilon, h)
x = x(:);
d = d(:);
assert(numel(x) == numel(d), 'Input and microphone lengths must match.');
L = numel(h);
w = zeros(L, 1);
e = zeros(numel(x), 1);
wevn = zeros(numel(x), 1);
trueEnergy = sum(h.^2);
xPad = [zeros(L - 1, 1); x(:)];
inputEnergy = 0;
oldSample = 0;

for n = 1:numel(x)
    newest = n + L - 1;
    u = xPad(newest:-1:n);
    e(n) = d(n) - w.' * u;
    % Add the newest sample's energy and remove the outgoing sample's.
    inputEnergy = max(0, inputEnergy + xPad(newest)^2 - oldSample^2);
    oldSample = xPad(n);
    gain = (mu * e(n)) / (epsilon + inputEnergy);
    w = w + gain * u;
    wevn(n) = weight_error_db(h, w, trueEnergy);
end
end

function [e, w, wevn] = run_fnlms(x, d, mu, lambda, epsilon, initialPower, h)
x = x(:);
d = d(:);
assert(numel(x) == numel(d), 'Input and microphone lengths must match.');
% M for the tap count; this code uses L = M.
% Eqs. (30)-(35): overlap-save input, output, and zero-prefixed error.
% Eqs. (39)-(42): per-bin normalization followed by a causal constraint.
% epsilon is an explicit numerical regularizer for low-energy bins.
% Constrained frequency-domain NLMS. Use an overlap-save FFT of size 2L
% for L new samples and update the L causal FIR coefficients once per block.
L = numel(h);
nBlocks = floor(numel(x) / L);
W = zeros(2 * L, 1);
w = zeros(L, 1);
previousInput = zeros(L, 1);
zeroBlock = zeros(L, 1);
binPower = initialPower * ones(2 * L, 1);
e = zeros(numel(x), 1);
wevn = zeros(nBlocks, 1);
trueEnergy = sum(h.^2);
for block = 1:nBlocks
    index = (block - 1) * L + (1:L);
    currentInput = x(index);
    X = fft([previousInput; currentInput]);
    predicted = real(ifft(X .* W));
    residual = d(index) - predicted(L+1:end);
    e(index) = residual;

    E = fft([zeroBlock; residual]);
    % Eq. (39): exponentially smoothed energy in each frequency bin.
    binPower = lambda * binPower + (1 - lambda) * abs(X).^2;
    % Eq. (41): normalize BEFORE IFFT and the gradient constraint.
    normalizedGradient = (conj(X) .* E) ./ (binPower + epsilon);
    gradient = real(ifft(normalizedGradient));
    causalGradient = gradient(1:L);
    % Eq. (42): zero-pad the constrained gradient and update.
    W = W + mu * fft([causalGradient; zeroBlock]);
    w = w + mu * causalGradient;  % same update, retained for WEVN

    wevn(block) = weight_error_db(h, w, trueEnergy);
    previousInput = currentInput;
end
% Evaluate the remaining actual samples without a partial-block update.
firstTailSample = nBlocks * L + 1;
if firstTailSample <= numel(x)
    tailIndex = firstTailSample:numel(x);
    tailOutput = filter(w, 1, [previousInput; x(tailIndex)]);
    e(tailIndex) = d(tailIndex) - tailOutput(L+1:end);
end
end

function [d, actualSnr] = noisy_microphone(x, h, snrDb)
% system identification: microphone = echo + independent noise.
cleanEcho = filter(h, 1, x);
noise = randn(size(cleanEcho));
noise = noise - mean(noise);
signalPower = mean(cleanEcho.^2);
noise = noise * sqrt(signalPower / (10^(snrDb / 10) * mean(noise.^2)));
d = cleanEcho + noise;
actualSnr = 10 * log10(signalPower / mean(noise.^2));
end

function h = load_echo_path(root, filename, L)
data = load(data_path(root, filename), 'h');
h = data.h(:);
assert(numel(h) == L && isreal(h) && all(isfinite(h)) && any(h ~= 0), ...
    'Invalid echo path in %s.', filename);
end

function path = data_path(root, filename)
path = fullfile(root, filename);
if isfile(path)
    return;
end
path = fullfile(root, 'data', filename);
if isfile(path)
    return;
end
% Also support the layout used for the original attachments.
path = fullfile(root, 'upload', filename);
assert(isfile(path), 'Put %s next to this experiment script or in the data subfolder.', filename);
end

function value = weight_error_db(h, w, trueEnergy)
difference = h - w;
ratio = (difference.' * difference) / trueEnergy;
assert(isfinite(ratio), 'Filter diverged. Reduce the step size or check the update.');
value = 10 * log10(max(ratio, realmin));
end

function fig = wevn_figure(number, description)
fig = figure('Color', 'w', 'Name', sprintf('Experiment %d', number));
set(fig, 'Position', [80 80 1050 650], ...
    'DefaultAxesFontSize', 11, 'DefaultTextFontSize', 11, ...
    'DefaultAxesColorOrder', [0 0.35 0.65; 0.85 0.25 0.08; 0.45 0.15 0.65]);
grid on;
hold on;
xlabel('Input samples processed');
ylabel('WEVN (dB)');
title(sprintf('Experiment %d - %s', number, description));
end

function save_figure(fig, resultsDir, filename)
print(fig, fullfile(resultsDir, filename), '-dpng', '-r200');
[~, stem] = fileparts(filename);
savefig(fig, fullfile(resultsDir, [stem '.fig']));
end

function print_summary(label, blockCurve, endOfFileWevn)
tailCount = max(1, ceil(0.10 * numel(blockCurve)));
tail = blockCurve(end-tailCount+1:end);
fprintf(['  %s: lowest block-end WEVN %.2f dB; last block-end WEVN %.2f dB; ' ...
    'WEVN at end of file %.2f dB; mean of last %d block-end values %.2f dB\n'], ...
    label, min(blockCurve), blockCurve(end), endOfFileWevn, tailCount, mean(tail));
end
