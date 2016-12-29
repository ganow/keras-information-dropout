require 'os'
require 'torch'
require 'paths'

local mnist_cluttered = require 'mnist_cluttered'
local idx = torch.range(0, 9)

local TRAINSIZE = 50000
local VALIDSIZE = 10000
local TESTSIZE = 10000

local TRAINPATH = 'data/train.t7'
local VALIDPATH = 'data/valid.t7'
local TESTPATH = 'data/test.t7'

local TRAINOUTPATH = 'data/cluttered_train.t7'
local VALIDOUTPATH = 'data/cluttered_valid.t7'
local TESTOUTPATH = 'data/cluttered_test.t7'

local function processData(n, inpath, outpath)

  print(inpath .. "->" .. outpath)

  local dataConfig = {datasetPath = inpath, megapatch_w=96, num_dist=21}
  local dataInfo = mnist_cluttered.createData(dataConfig)

  local set = { data = torch.ByteTensor(n, 96, 96),
                labels = torch.ByteTensor(n, 1) }

  for i = 1, n do
    if i % 1000 == 0 then
      print(i .. "/" .. n .. " done.")
    end

    local observation, target = unpack(dataInfo.nextExample())

    -- read the image
    set.data[i] = (observation*255):floor()
    -- read label
    set.labels[i][1] = idx * target
  end

  torch.save(outpath, set)
end

processData(TRAINSIZE, TRAINPATH, TRAINOUTPATH)
processData(VALIDSIZE, VALIDPATH, VALIDOUTPATH)
processData(TESTSIZE,  TESTPATH,  TESTOUTPATH)
