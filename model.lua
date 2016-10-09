local nn = require "nn"
require "rnn"
require "cunn"
local cudnn = require "cudnn"

local inputSize = 16
local hiddenSize = 128
local outputSize = 1
local nLayers = 8
local dropout = 0.5
local rho = 240000
nn.FastLSTM.bn = true

local net = nn.Sequential()

-- First LSTM layer
net:add(nn.FastLSTM(inputSize, hiddenSize, rho))
-- Hidden LSTM layers
for i = 0, (nLayers - 1) do
   net:add(nn.Dropout(dropout))
   net:add(nn.FastLSTM(hiddenSize, hiddenSize, rho))
end
-- FC linear layer for classification
net:add(nn.Dropout(dropout))
net:add(nn.Linear(hiddenSize, outputSize))
net:add(nn.Sigmoid())

-- Move to GPU
net = net:cuda()

return net
