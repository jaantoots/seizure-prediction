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
print('==> Initialise/load model')
local net = require 'model'
local criterion = nn.BCECriterion()
criterion = criterion:cuda()
-- Load network from file if provided
local startIteration = 0
if args.model then
   net = torch.load(args.model)
   startIteration = string.match(args.model, '_(%d+)%.t7$') or startIteration
end

-- Prepare output
opts.maxIterations = args.iter and (startIteration + args.iter) or
   opts.maxIterations or (startIteration + 10000)
json.save(opts.output .. '/conf.json', opts)
local logger = optim.Logger(opts.output .. '/loss.log')
logger:setNames{'Iteration', 'Loss'}
local lossWindow = torch.Tensor(10):zero()

-- Debugging info
local predictions = optim.Logger(opts.output .. '/prediction.log')
predictions:setNames{'Label', 'Prediction'}

-- Train the network
net:training()
-- optim requires 1D tensors
local params, gradParams = net:getParameters()
print('==> Start training: ' .. params:nElement() .. ' parameters')
for i = (startIteration + 1), opts.maxIterations do
   -- Get the sequence
   local batch = trainData:nextTrain(opts.batchSize)
   local inputs = batch.inputs:cuda()
   local labels = batch.labels:cuda()

   local function feval (_)
      -- For optim, outputs f(X): loss and df/dx: gradients
      gradParams:zero()
      -- Forward pass
      local outputs = net:forward(inputs)
      local loss = criterion:forward(outputs, labels)
      -- Backpropagation
      local gradLoss = criterion:backward(outputs, labels)
      net:backward(inputs, gradLoss)
      -- Log predictions
      for j = 1, opts.batchSize do
         predictions:add{labels[j]:squeeze(), outputs[j]:squeeze()}
      end
      -- Statistics
      return loss, gradParams
   end
   local _, fs = optim.adam(feval, params, opts.config)

   -- Log loss
   lossWindow[math.fmod(i, 10) + 1] = fs[1]
   if i >= 10 then
      print(i, lossWindow:mean())
      logger:add{i, lossWindow:mean()}
   end
   -- Save model
   if math.fmod(i, 100) == 0 then
      net:clearState()
      torch.save(opts.output .. '/model_' .. i .. '.t7', net)
   end
end

-- Validate the network
net:evaluate()
print('==> Start validation')
local lossValues = torch.Tensor(math.ceil(trainData.data/opts.batchSize))
local predValues = torch.Tensor(math.ceil(trainData.data/opts.batchSize)*opts.batchSize, 2)
for i = 1, math.ceil(trainData.data/opts.batchSize) do
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
