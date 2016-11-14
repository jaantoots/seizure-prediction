local torch = require 'torch'
require 'cutorch'
local nn = require 'nn'
require 'cunn'
local cudnn = require 'cudnn'
local optim = require 'optim'
local paths = require 'paths'
local json = require 'json'

local helpers = require 'helpers'
local data = require 'data'

-- Enable these for final training
cudnn.benchmark = true
cudnn.fastest = true

-- Parse arguments & load configuration
local parser = helpers.parser()
local args = parser:parse()
local opts = helpers.opts(args)
paths.mkdir(opts.output)

-- Initialize and normalize training data
print('==> Load data')
local testData = data.new(args.dir, false)
trainData:printStats()
opts.mean, opts.std = trainData:normalize(opts.mean, opts.std)
opts.mean = opts.mean:totable()
opts.std = opts.std:totable()

-- Network and loss function
print('==> Initialise/load model')
local net = require 'model'
-- Load network from file if provided
local startIteration = 0
if args.model then
   net = torch.load(args.model)
end

-- Predictions
local predictions = optim.Logger(opts.output .. '/prediction.log')
predictions:setNames{'Name', 'Prediction'}

-- Validate the network
net:evaluate()
print('==> Start validation')
local predValues = torch.Tensor(math.ceil(testData.data/opts.batchSize)*opts.batchSize, 2)
for i = 1, math.ceil(testData.data/opts.batchSize) do
   -- Get the sequence
   local batch, names = testData:nextTest(opts.batchSize)
   local inputs = batch.inputs:cuda()

   -- Forward through the network
   local outputs = net:forward(inputs)
   -- Log predictions
   for j = 1, opts.batchSize do
      predictions:add{names[j], outputs[j]:squeeze()}
      print(names[j], outputs[j]:squeeze())
   end
end
