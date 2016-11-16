local torch = require 'torch'
require 'cutorch'
local nn = require 'nn'
require 'cunn'
local cudnn = require 'cudnn'
local optim = require 'optim'
local paths = require 'paths'
local json = require 'json'

local metrics = require 'metrics'
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
local trainData = data.new(args.dir, true)
trainData:splitValidate(opts.output .. '/split.dat', 0.1)
trainData:printStats()
opts.mean, opts.std = trainData:normalize(opts.mean, opts.std)
opts.mean = opts.mean:totable()
opts.std = opts.std:totable()

-- Network and loss function
local net = require 'model'
local criterion = nn.BCECriterion()
criterion = criterion:cuda()

-- Load network from file
print('==> Start validation')
for _, model in pairs(args.validate) do
   print(model)
   net = torch.load(model):cuda()

   -- Validate the network
   net:evaluate()
   local lossValues = torch.Tensor(
      math.ceil(#trainData.validate/opts.batchSize))
   local predValues = torch.Tensor(
      math.ceil(#trainData.validate/opts.batchSize)*opts.batchSize, 2)
   for i = 1, math.ceil(#trainData.validate/opts.batchSize) do
      -- Get the sequence
      local batch = trainData:nextValidate(opts.batchSize)
      local inputs = batch.inputs:cuda()
      local labels = batch.labels:cuda()

      -- Forward through the network
      local outputs = net:forward(inputs)
      local loss = criterion:forward(outputs, labels)
      lossValues[i] = loss
      predValues[{ {(i - 1)*opts.batchSize + 1, i*opts.batchSize}, 1}] =
         outputs:squeeze():double()
      predValues[{ {(i - 1)*opts.batchSize + 1, i*opts.batchSize}, 2}] =
         labels:squeeze():double()
   end
   print(lossValues:mean())
   rocPoints = metrics.roc.points(predValues[{ {}, 1 }], predValues[{ {}, 2 }], 0, 1)
   area = metrics.roc.area(rocPoints)
   print(area)
end
