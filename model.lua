local nn = require "nn"
require "rnn"
require "cunn"
local cudnn = require "cudnn"

local inputSize = 16
local hiddenSize = 128
local outputSize = 1
local nLayers = 8

local net = nn.Sequential()

-- First LSTM layer
net:add(nn.FastLSTM(inputSize, hiddenSize))
-- Other LSTM layers
for i = 0, (nLayers - 1) do
   net:add(nn.FastLSTM(hiddenSize, hiddenSize))
end
-- FC linear layer for classification
net:add(nn.Linear(hiddenSize, outputSize))

net = nn.Sequencer(net)

-- Weights initialization for convolutional layers
local function InitializeWeights (name)
  -- Initialize layer of type `name`
  for _, module in pairs(net:findModules(name)) do
    -- see arXiv:1502.01852 [cs.CV]
    local n = module.inputSize * module.outputSize
    module.weight:normal(0, math.sqrt(2/n))
    module.bias:zero()
  end
end
-- Initialize used types
InitializeWeights("nn.FastLSTM")

-- Move to GPU
cudnn.convert(net, cudnn)
net = net:cuda()

return net
