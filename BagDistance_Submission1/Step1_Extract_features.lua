-------------------------------------------------------------------------------------------------------------------------------------------
--- To use:
--- (I) Install torch and donwload the resnet model https://github.com/facebook/fb.resnet.torch
--- (II) Place the 3 files in the same folder: 
-------(1) the model file 'resnet-101.t7'  
-------(2) the list of images 'image_list.txt'
-------(3) this file 'Step1_Extract_features.lua'
--- (III)Then run $th Step1_Extract_features.lua 
--- The image features will be save in 'image_features.h5'

--- Remark: This file is modified from https://github.com/facebook/fb.resnet.torch/blob/master/pretrained/extract-features.lua
-------------------------------------------------------------------------------------------------------------------------------------------


--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  extracts features from an image using a trained model
--

require 'torch'
require 'paths'


require 'cudnn'
require 'cunn'
require 'image'
local t = require '../datasets/transforms'

-- Load the model
--local model = torch.load(arg[1])
local model = torch.load('resnet-101.t7')

-- Remove the fully connected layer
assert(torch.type(model:get(#model.modules)) == 'nn.Linear')
model:remove(#model.modules)

-- The model was trained with this input normalization
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}

local transform = t.Compose{
   t.Scale(256),
   t.ColorNormalize(meanstd),
   t.CenterCrop(224),
}

local features

local ctr = 0
for _ in io.lines'image_list.txt' do
  ctr = ctr + 1
end

local i = 1
for line in io.lines'image_list.txt' do

   local img = image.load(line, 3, 'float')

   -- Scale, normalize, and crop the image
   img = transform(img)

   -- View as mini-batch of size 1
   img = img:view(1, table.unpack(img:size():totable()))

   -- Get the output of the layer before the (removed) fully connected layer
   local output = model:forward(img:cuda()):squeeze(1)

   if not features then
      features = torch.FloatTensor(ctr, output:size(1)):zero()
   end

   features[i]:copy(output)
   i = i+1
end

require 'hdf5'
local myFile = hdf5.open('image_features.h5', 'w')
myFile:write('feature',features)
myFile:close()






