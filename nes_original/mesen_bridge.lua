-- Trusted local MesenCE bridge: controller inputs and read-only instrumentation.
-- The Python-generated entry script defines MESEN_BRIDGE_ROOT/TIMEOUT.
-- No game RAM writes, ROM patches, save-state restores, or frame skipping.
local root = assert(MESEN_BRIDGE_ROOT)
local timeout = MESEN_BRIDGE_TIMEOUT or 30
local sequence = 0
local stopped = false
local lastCommand = -1
local pipeIn, pipeOut = nil, nil
if MESEN_BRIDGE_PIPE_IN then
  pipeIn = assert(io.open(MESEN_BRIDGE_PIPE_IN, "rb"))
  pipeOut = assert(io.open(MESEN_BRIDGE_PIPE_OUT, "wb"))
end

local function publish(name, data)
  local temporary = root .. name .. ".tmp"
  local f = assert(io.open(temporary, "wb"))
  f:write(data); f:close()
  -- Windows readers can briefly hold a non-delete-sharing file handle.
  local deadline = os.time() + timeout
  repeat
    os.remove(root .. name)
    if os.rename(temporary, root .. name) then return end
  until os.time() > deadline
  error("atomic publication timed out: " .. name)
end

local function fail(reason)
  if stopped then return end
  stopped = true
  publish("error.json", string.format('{"sequence":%d,"reason":"%s"}', sequence, reason))
  emu.stop(2)
end

local function capture(command)
  -- Requested explicitly, never transferred on every frame.
  local pixels = emu.getScreenBuffer()
  local chunks = {}
  for i = 1, #pixels do chunks[i] = string.pack("<I4", pixels[i]) end
  publish("screen-" .. command .. ".argb", table.concat(chunks))
  publish("screen-" .. command .. ".png", emu.takeScreenshot())
  publish("capture.json", string.format('{"sequence":%d,"command":%d,"pixels":%d}', sequence, command, #pixels))
end

emu.addEventCallback(function()
  if stopped then return end
  sequence = sequence + 1
  local values = {}
  for address = 0, 2047 do
    values[#values + 1] = string.format("%02x", emu.read(address, emu.memType.nesInternalRam))
  end
  local s = emu.getState()
  local request = string.format(
    '{"sequence":%d,"ram_hex":"%s","pc":%d,"cpu_cycle":%d,"video_frame":%d,"ppu_scanline":%d,"ppu_cycle":%d}',
    sequence, table.concat(values), s["cpu.pc"], s["cpu.cycleCount"], s["frameCount"], s["ppu.scanline"], s["ppu.cycle"])
  if pipeOut then pipeOut:write(request .. "\n"); pipeOut:flush()
  else publish("request.json", request) end
  local deadline = os.time() + timeout
  while not stopped do
    local data = nil
    if pipeIn then
      -- C read yields the CPU; the Python host's independent idle watchdog
      -- terminates ONLY its own child if actions stop arriving. Host death
      -- closes the pipe, unblocking this read without an orphaned emulator.
      data = pipeIn:read("*l")
      if not data then fail("host_pipe_closed"); return end
    else
      local f = io.open(root .. "response.txt", "rb")
      if f then data = f:read("*a"); f:close() end
    end
    if data then
      if data:match("^quit") then stopped = true; emu.stop(0); return end
      local seq, kind, value = data:match("^(%d+) (%a+) (%d+)$")
      seq, value = tonumber(seq), tonumber(value)
      if seq == sequence and kind == "capture" and value ~= lastCommand then
        capture(value); lastCommand = value
      elseif seq == sequence and kind == "action" then
        if not value or value > 255 then fail("invalid_controller_byte"); return end
        emu.setInput({a=value%2>=1,b=math.floor(value/2)%2>=1,
          select=math.floor(value/4)%2>=1,start=math.floor(value/8)%2>=1,
          up=math.floor(value/16)%2>=1,down=math.floor(value/32)%2>=1,
          left=math.floor(value/64)%2>=1,right=math.floor(value/128)%2>=1}, 0, 0)
        return
      end
    end
    if os.time() > deadline then fail("host_action_timeout"); return end
  end
end, emu.eventType.inputPolled)
