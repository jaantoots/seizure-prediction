local nn = require "nn"
require "rnn"
require "cunn"
local cudnn = require "cudnn"

local dropout = 0.5

-- Convolutional layers
-- (240000/2^7+5)/2^3
local cnn = nn.Sequential()

-- Switch batch to first dim
cnn:add(nn.Transpose({1, 2}))

function ConvolutionLayer (inputFrameSize, outputFrameSize)
   -- Pad for nice convolution (dim, pad, nInputDim)
   cnn:add(nn.Padding(1, 3, 2))
   -- TemporalConvolution layer with 4 kernel, stride 1
   cnn:add(nn.TemporalConvolution(inputFrameSize, outputFrameSize, 4, 1))
   -- ReLU activation
   cnn:add(nn.ReLU(true))
end

function DownConvolutionLayer (inputFrameSize, outputFrameSize)
   -- Pad for nice convolution (dim, pad, nInputDim)
   cnn:add(nn.Padding(1, 2, 2))
   -- TemporalConvolution layer with 4 kernel, stride 2
   cnn:add(nn.TemporalConvolution(inputFrameSize, outputFrameSize, 4, 2))
   -- ReLU activation
   cnn:add(nn.ReLU(true))
end

-- 240000x16
ConvolutionLayer(16, 32)
ConvolutionLayer(32, 32)
cnn:add(nn.TemporalMaxPooling(2, 2))
-- 120000x64
ConvolutionLayer(32, 32)
ConvolutionLayer(32, 32)
cnn:add(nn.TemporalMaxPooling(2, 2))
-- 60000x256
ConvolutionLayer(32, 64)
ConvolutionLayer(64, 64)
cnn:add(nn.TemporalMaxPooling(2, 2))
-- 30000x256
ConvolutionLayer(64, 64)
ConvolutionLayer(64, 64)
cnn:add(nn.TemporalMaxPooling(2, 2))
-- 15000x256
ConvolutionLayer(64, 128)
ConvolutionLayer(128, 128)
cnn:add(nn.TemporalMaxPooling(2, 2))
-- 7500x256
ConvolutionLayer(128, 128)
ConvolutionLayer(128, 128)
cnn:add(nn.TemporalMaxPooling(2, 2))
-- 3750x256
ConvolutionLayer(128, 256)
ConvolutionLayer(256, 256)
cnn:add(nn.TemporalMaxPooling(2, 2))
-- 1875x256
cnn:add(nn.Padding(1, 5, 2))
-- 1880x256
ConvolutionLayer(256, 256)
ConvolutionLayer(256, 256)
cnn:add(nn.TemporalMaxPooling(2, 2))
-- 940x256
ConvolutionLayer(256, 256)
ConvolutionLayer(256, 256)
cnn:add(nn.TemporalMaxPooling(2, 2))
-- 470x256
ConvolutionLayer(256, 256)
ConvolutionLayer(256, 256)
cnn:add(nn.TemporalMaxPooling(2, 2))
-- 235x256

-- Switch batch to second dim
cnn:add(nn.Transpose({1, 2}))

-- Weights initialization for convolutional layers
for _, module in pairs(cnn:findModules("nn.TemporalConvolution")) do
   -- see arXiv:1502.01852 [cs.CV]
   local n = module.kW * module.outputFrameSize
   module.weight:normal(0, math.sqrt(2/n))
   module.bias:zero()
end

-- Convert to cudnn
cudnn.convert(cnn, cudnn)

-- Recurrent layers
local rnn = nn.Sequential()
nn.FastLSTM.bn = true

-- LSTM
rnn:add(nn.FastLSTM(256, 256))
rnn:add(nn.Dropout(dropout))
rnn:add(nn.FastLSTM(256, 128))
rnn:add(nn.Dropout(dropout))
rnn:add(nn.FastLSTM(128, 128))
rnn:add(nn.Dropout(dropout))
rnn:add(nn.FastLSTM(128, 64))
rnn:add(nn.Dropout(dropout))
rnn:add(nn.FastLSTM(64, 64))
rnn:add(nn.Dropout(dropout))
-- FC linear layer for classification
rnn:add(nn.Linear(64, 1))
rnn:add(nn.Sigmoid())

rnn = nn.Sequencer(rnn)

-- Combine the networks
local net = nn.Sequential()
net:add(cnn)
net:add(rnn)

-- Move to GPU
net = net:cuda()

return net
