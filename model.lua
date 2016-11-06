local nn = require "nn"
require "cunn"
local cudnn = require "cudnn"

local dropout = 0.5

-- Convolutional layers
local cnn = nn.Sequential()

function ConvolutionLayer (inputFrameSize, outputFrameSize)
   -- Pad for nice convolution (dim, pad, nInputDim)
   cnn:add(nn.Padding(1, 3, 2))
   -- TemporalConvolution layer with 4 kernel, stride 1
   cnn:add(nn.TemporalConvolution(inputFrameSize, outputFrameSize, 4, 1))
   -- ReLU activation
   cnn:add(nn.ReLU(true))
end

-- 240000x16
ConvolutionLayer(16, 32)
ConvolutionLayer(32, 32)
cnn:add(nn.TemporalMaxPooling(4, 4))
-- 60000
ConvolutionLayer(32, 64)
ConvolutionLayer(64, 64)
cnn:add(nn.TemporalMaxPooling(4, 4))
-- 15000
ConvolutionLayer(64, 128)
ConvolutionLayer(128, 128)
cnn:add(nn.TemporalMaxPooling(4, 4))
-- 3750
cnn:add(nn.Padding(1, 10, 2))
-- 3760
ConvolutionLayer(128, 256)
ConvolutionLayer(256, 256)
cnn:add(nn.TemporalMaxPooling(4, 4))
-- 940
ConvolutionLayer(256, 512)
ConvolutionLayer(512, 512)
cnn:add(nn.TemporalMaxPooling(4, 4))
-- 235
cnn:add(nn.Padding(1, 5, 2))
-- 240
ConvolutionLayer(512, 512)
cnn:add(nn.Dropout(dropout))
ConvolutionLayer(512, 512)
cnn:add(nn.TemporalMaxPooling(4, 4))
-- 60
ConvolutionLayer(512, 512)
cnn:add(nn.Dropout(dropout))
ConvolutionLayer(512, 512)
cnn:add(nn.TemporalMaxPooling(4, 4))
-- 15
cnn:add(nn.Padding(1, 1, 2))
-- 16
ConvolutionLayer(512, 512)
cnn:add(nn.Dropout(dropout))
ConvolutionLayer(512, 512)
cnn:add(nn.TemporalMaxPooling(4, 4))
-- 4
ConvolutionLayer(512, 512)
cnn:add(nn.Dropout(dropout))
ConvolutionLayer(512, 512)
cnn:add(nn.TemporalMaxPooling(4, 4))
-- 1
cnn:add(nn.Squeeze(1, 2))
cnn:add(nn.Linear(512, 128))
cnn:add(nn.Linear(128, 32))
cnn:add(nn.Linear(32, 1))
cnn:add(nn.Sigmoid())

-- Weights initialization for convolutional layers
for _, module in pairs(cnn:findModules("nn.TemporalConvolution")) do
   -- see arXiv:1502.01852 [cs.CV]
   local n = module.kW * module.outputFrameSize
   module.weight:normal(0, math.sqrt(2/n))
   module.bias:zero()
end

-- Convert to cudnn
cudnn.convert(cnn, cudnn)
-- Move to GPU
net = cnn:cuda()

return net
