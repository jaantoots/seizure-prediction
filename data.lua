local torch = require 'torch'
local paths = require 'paths'
local matio = require 'matio'

local Data = torch.class('Data')

function Data:__init (dir, isTrain)
   -- Load all mat files from dir
   self.inputs = {}
   self.names = {}
   self.isTrain = isTrain
   for file in paths.files(dir, function (file)
                              return string.match(file, '%.mat$') end) do
      self.names[#self.names + 1] = string.match(file, '^(.*)%.mat$')
      local content = matio.load(dir .. '/' .. file)
      self.inputs[#self.inputs + 1] = content['dataStruct']
   end
   -- Variables for sizes
   self.data = #self.inputs
   self.samples = self.inputs[1]['data'].size[1]
   self.electrodes = self.inputs[1]['data'].size[2]
   -- Labels only when training
   if self.isTrain then
      self._getLabels()
   end
   -- Initialise stuff for returning data
   self.identity = torch.range(1, self.data)
   self.shuffle = nil
   self.iteration = self.data
end

function Data:_getLabels ()
   -- Extract label from name for training data
   self.labels = {}
   for i, name in pairs(self.names) do
      self.labels[i] = tonumber(string.match(name, '%d+_%d+_(%d+)'))
   end
end

function Data:splitValidate (splitFile, splitRatio)
   -- Split the data into training and validation sets
   if paths.filep(splitFile) then
      local split = torch.load(splitFile, 'ascii')
      self.validate = split.validate
      self.train = split.train
   else
      self.validate = {}
      self.train = {}
      splitRatio = splitRatio or 0.1
      for i, _ in pairs(self.name) do
         if (torch.bernoulli(splitRatio) == 1) then
            self.validate[#self.validate + 1] = i
         else
            self.train[#self.train + 1] = i
         end
      end
      local split = {validate = self.validate, train = self.train}
      torch.save(splitFile, split, 'ascii')
   end
end

function Data:printStats ()
   -- Print data statistics
   print("Data", "Samples", "Electrodes")
   print(self.data, self.samples, self.electrodes)
   if self.isTrain then
      print("Train", "Validate")
      print(#self.train, #self.validate)
   end
end

function Data:normalize (mean, std)
   -- Normalize each electrode channel based on the training data
   if not mean or not std then
      local means = torch.Tensor(self.data, self.electrodes)
      local stds = torch.Tensor(self.data, self.electrodes)
      for i, input in pairs(self.inputs) do
         means[i] = input['data']:mean(1)
         stds[i] = input['data']:std(1)
      end
      self.mean = means:mean(1)
      self.std = stds:std(1)
   else
      self.mean = mean
      self.std = std
   end
   return self.mean, self.std
end

-- TODO: use batches!
function Data:nextTrain (noShuffle)
   return self:_nextFromMap(self.train, noShuffle)
end

function Data:nextValidate ()
   return self:_nextFromMap(self.validate, true)
end

function Data:nextTest ()
   return self._nextFromMap(self.identity, true)
end

function Data:_nextFromMap (map, noShuffle)
   if self.iteration >= #map then
      if noShuffle then
         local shuffle = torch.range(1, #map)
      else
         local shuffle = torch.randperm(#map)
      end
      self.shuffle = torch.Tensor(#map)
      for i, seq in pairs(shuffle) do
         self.shuffle[i] = map[seq]
      end
      self.iteration = 0
   end
   return self:_getSequence()
end   

function Data:_getSequence ()
   self.iteration = self.iteration + 1
   local n = self.shuffle[self.iteration]
   local input = self.inputs[n]['data']:
      add(-self.mean:view(1, self.electrodes):
             expand(self.samples, self.electrodes)):
      cdiv(self.std:view(1, self.electrodes):
              expand(self.samples, self.electrodes))
   if self.labels then
      local label = self.labels[n]
   else
      local label = nil
   end
   return {input = input, label = label}, self.names[n]
end

return Data
