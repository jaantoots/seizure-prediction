local torch = require 'torch'
local paths = require 'paths'
local json = require 'json'
local argparse = require 'argparse'

local helpers = {}

function helpers.parser ()
   -- Return argparse object
   local parser = argparse('train.lua',
                           'Train a LSTM RNN for seizure prediction.')
   parser:option('-c --conf', 'Configuration file.', 'conf.json')
   parser:option('-d --dir', 'Data directory.')
   parser:option('-o --output', 'Output directory.')
   parser:option('-b --batch', 'Batch size.')
   parser:option('-i --iter', 'Number of iterations to train.')
   parser:option('-m --model', 'Saved model, if continuing training.')
   return parser
end

function helpers.opts (args)
  -- Return opts for training
  local opts
  if paths.filep(args.conf) then
    opts = json.load(args.conf)
  else
    opts = {}
  end
  opts.output = args.output or opts.output or
    'out/' .. os.date('%Y-%m-%d-%H-%M-%S')
  opts.batchSize = args.batch or opts.batchSize or 8
  opts.config = opts.config or {
    learningRate = 1e-2,
    alpha = 0.99,
    epsilon = 1e-6
  }
  return opts
end

return helpers
