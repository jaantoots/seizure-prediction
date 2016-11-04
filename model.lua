local nn = require "nn"
require "rnn"
require "cunn"
local cudnn = require "cudnn"

local dropout = 0.5

-- Convolutional layers
-- (240000/2^7+5)/2^3
local net = nn.Sequential()

-- Switch batch to first dim
net:add(nn.Transpose({1, 2}))

function net:ConvolutionLayer (inputFrameSize, outputFrameSize)
   -- Pad for nice convolution (dim, pad, nInputDim)
   self:add(nn.Padding(1, 3, 2))
   -- TemporalConvolution layer with 4 kernel, stride 1
   self:add(nn.TemporalConvolution(inputFrameSize, outputFrameSize, 4, 1))
   -- Batch normalization with default arguments
   self:add(nn.BatchNormalization(outputFrameSize))
   -- ReLU activation
   self:add(nn.ReLU(true))
end

function net:DownConvolutionLayer (inputFrameSize, outputFrameSize)
   -- Pad for nice convolution (dim, pad, nInputDim)
   self:add(nn.Padding(1, 2, 2))
   -- TemporalConvolution layer with 4 kernel, stride 2
   self:add(nn.TemporalConvolution(inputFrameSize, outputFrameSize, 4, 2))
   -- Batch normalization with default arguments
   self:add(nn.BatchNormalization(outputFrameSize))
   -- ReLU activation
   self:add(nn.ReLU(true))
end

-- 240000x16
net:ConvolutionLayer(16, 64)
net:ConvolutionLayer(64, 64)
net:add(nn.TemporalMaxPooling(2, 2))
-- 120000x64
net:ConvolutionLayer(64, 256)
net:ConvolutionLayer(256, 256)
net:add(nn.TemporalMaxPooling(2, 2))
-- 60000x256
net:DownConvolutionLayer(256, 256)
net:DownConvolutionLayer(256, 256)
net:add(nn.Dropout(dropout))
net:DownConvolutionLayer(256, 256)
net:DownConvolutionLayer(256, 256)
net:add(nn.Dropout(dropout))
net:DownConvolutionLayer(256, 256)
-- 1875x256
net:add(nn.Padding(1, 5, 2))
-- 1880x256
net:DownConvolutionLayer(256, 256)
net:add(nn.Dropout(dropout))
net:DownConvolutionLayer(256, 256)
net:DownConvolutionLayer(256, 256)
net:add(nn.Dropout(dropout))
-- 235x256

-- Switch batch to second dim
net:add(nn.Transpose({1, 2}))

-- Weights initialization for convolutional layers
for _, module in pairs(net:findModules("nn.TemporalConvolution")) do
   -- see arXiv:1502.01852 [cs.CV]
   local n = module.kW * module.outputFrameSize
   module.weight:normal(0, math.sqrt(2/n))
   module.bias:zero()
end

-- Move to GPU
cudnn.convert(net, cudnn)
net = net:cuda()

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

-- Move to GPU
rnn = rnn:cuda()

return net, rnn
