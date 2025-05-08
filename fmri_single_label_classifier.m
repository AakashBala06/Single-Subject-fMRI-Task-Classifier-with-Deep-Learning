%% STEP 1: Load fMRI Data and Labels
fmri = niftiread('sub-1_task-objectviewing_run-01_bold.nii.gz');
events = readtable('sub-1_task-objectviewing_run-01_events.tsv', 'FileType', 'text');

%% STEP 2: Preprocess Label Table
TR = 2.0;
events.time_index = floor(events.onset / TR) + 1;
events = events(events.time_index <= size(fmri, 4), :);

% Label as 1 if "scissors", 0 otherwise
labels = strcmp(events.trial_type, 'scissors');

%% STEP 3: Extract Slice at z = 20 from Each Labeled Time
z = 20;
n = height(events);
imgs = zeros(40, 64, n);  % 40x64 slices

for i = 1:n
    t = events.time_index(i);
    slice = fmri(:, :, z, t);
    imgs(:, :, i) = rescale(slice);  % normalize
end

%% STEP 4: Prepare Data for CNN
X = reshape(imgs, [40, 64, 1, n]);  % H x W x 1 x N
Y = categorical(labels);  % binary classification: scissors vs not

% Split
idx = randperm(n);
split = round(0.8 * n);
XTrain = X(:, :, :, idx(1:split));
YTrain = Y(idx(1:split));
XTest = X(:, :, :, idx(split+1:end));
YTest = Y(idx(split+1:end));

%% STEP 5: Define Simple CNN
layers = [
    imageInputLayer([40 64 1])
    convolution2dLayer(3, 8, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
];

%% STEP 6: Train Network
options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 8, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

net = trainNetwork(XTrain, YTrain, layers, options);

%% STEP 7: Evaluate
YPred = classify(net, XTest);
acc = sum(YPred == YTest) / numel(YTest);
fprintf("Test Accuracy: %.2f%%\n", acc * 100);
