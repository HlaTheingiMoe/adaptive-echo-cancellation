% Experiment 2: LMS versus NLMS on speech.


clearvars;
root = fileparts(mfilename('fullpath'));
resultsDir = fullfile(root, 'results', 'experiment_2');
if ~isfolder(resultsDir)
    mkdir(resultsDir);
end

% Experiments 2 and 3 use the same seed and microphone signal.
rng(2102, 'twister');
h = load_echo_path(root, 'h128.mat', 128);
[x, fs] = audioread(data_path(root, 'female1_8k.wav'));
assert(size(x, 2) == 1, 'Expected mono speech.');
L = numel(h);
numSamples = numel(x);
[d, actualSnr] = noisy_microphone(x, h, 30);
fprintf('\nSpeech input: %d samples at %d Hz, SNR %.2f dB\n', ...
    numel(x), fs, actualSnr);


time = (0:numSamples-1).' / fs;
fig = figure('Color', 'w', 'Name', 'Experiment 2 - input signals');
subplot(2, 1, 1);
plot(time, x);
grid on;
ylabel('x(n)');
title('Experiment 2 - far-end speech x(n) (female1\_8k.wav)');
subplot(2, 1, 2);
plot(time, d);
grid on;
xlabel('Time (s)');
ylabel('d(n)');
title('Microphone signal d(n) = echo + background noise (30 dB SNR)');
save_figure(fig, resultsDir, 'exp2_input_signals.png');



sample = (0:numel(x)).';
energy = filter(ones(L, 1), 1, x.^2);
muNlms = 0.45;  % Handout 9 Eq. (29), normalized step size
% Regularization tuned for the low-energy speech tail.
epsilon = 1e-2 * mean(energy);
[eNlms, wNlms, wevnNlms] = run_nlms(x, d, muNlms, epsilon, h);
nlmsLabel = sprintf('NLMS, mu = %.2f', muNlms);

fprintf('\nExperiment 2: LMS versus NLMS (NLMS epsilon = %.5g)\n', epsilon);
muLms = 0.70 * 2 / max(energy);
[eLms, wLms, wevnLms] = run_lms(x, d, muLms, L, h);
lmsLabel = sprintf('LMS, mu = %.5g', muLms);
print_summary(lmsLabel, wevnLms);
print_summary(nlmsLabel, wevnNlms);

fig = wevn_figure(2, 'LMS versus NLMS on speech');
plot(sample, [0, 0; wevnLms, wevnNlms], 'LineWidth', 1.2);
legend({lmsLabel, nlmsLabel}, 'Location', 'best');
save_figure(fig, resultsDir, 'exp2_lms_vs_nlms.png');

% All statistics refer to the same complete speech recording.
% The tail is near silence, so the final-window mean is not labelled steady state.
curves = [wevnLms, wevnNlms];
algorithmNames = {'LMS'; 'NLMS'};
tailCount = ceil(0.10 * numel(x));
lowestWevnDb = min(curves, [], 1).';
finalWevnDb = curves(end, :).';
meanTailWevnDb = mean(curves(end-tailCount+1:end, :), 1).';
comparison = table(algorithmNames, lowestWevnDb, finalWevnDb, meanTailWevnDb, ...
    'VariableNames', {'Algorithm', 'LowestWEVN_dB', 'FinalWEVN_dB', ...
    'MeanLast10PercentWEVN_dB'});
disp(comparison);
writetable(comparison, fullfile(resultsDir, 'comparison.csv'));



save(fullfile(resultsDir, 'exp2_results.mat'), ...
    'x', 'd', 'h', 'fs', 'actualSnr', 'muLms', 'muNlms', ...
    'epsilon', 'sample', 'eLms', 'wLms', 'wevnLms', ...
    'eNlms', 'wNlms', 'wevnNlms', 'comparison');

function [e, w, wevn] = run_lms(x, d, mu, L, h)
x = x(:);
d = d(:);
assert(numel(x) == numel(d), 'Input and microphone lengths must match.');
% The optional echo path h is used only to evaluate the learned weights.
w = zeros(L, 1);
e = zeros(numel(x), 1);
wevn = [];
measure = nargin >= 5;
if measure
    wevn = zeros(numel(x), 1);
    trueEnergy = sum(h.^2);
end
xPad = [zeros(L - 1, 1); x(:)];

for n = 1:numel(x)
    u = xPad(n+L-1:-1:n);
    e(n) = d(n) - w.' * u;
    w = w + (mu * e(n)) * u;
    if measure
        wevn(n) = weight_error_db(h, w, trueEnergy);
    end
end
end

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

function print_summary(label, curve)
tailCount = max(1, ceil(0.10 * numel(curve)));
tail = curve(end-tailCount+1:end);
fprintf(['  %s: lowest WEVN %.2f dB; final WEVN %.2f dB; ' ...
    'mean of last %d values %.2f dB\n'], ...
    label, min(curve), curve(end), tailCount, mean(tail));
end
