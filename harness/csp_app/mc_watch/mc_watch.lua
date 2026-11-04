-- mapcreator watch: development-only Custom Shaders Patch app. Logs every car in the session, for tests with AI
-- opponents (M4): grid spawns, the start, the run to the finish and what happens in the run-off.
--
-- It does nothing unless Documents/Assetto Corsa/mapcreator_probe/watch.on exists when AC starts (the harness's
-- `race` command writes it, and the app deletes it). It then samples every car 4 times a second for the first
-- 15 s and once a second after that, and rewrites watch.json every 5 s. Fields some CSP versions may lack are read
-- through pcall.

local dir = ac.getFolder(ac.FolderID.ACDocuments) .. '/mapcreator_probe'
local active = io.load(dir .. '/watch.on') ~= nil
if active then
  -- One session per switch, as the probe deletes its points file: later launches play normally.
  io.deleteFile(dir .. '/watch.on')
end
local samples, t, next_sample, next_save = {}, 0, 0, 5

local function try(f)
  local ok, v = pcall(f)
  if ok then return v end
  return nil
end

local function sample()
  local sim = ac.getSim()
  local cars = {}
  for i = 0, sim.carsCount - 1 do
    local c = ac.getCar(i)
    if c then
      local wheels = {}
      for w = 0, 3 do
        local wh = c.wheels[w]
        wheels[#wheels + 1] = {
          y = wh.contactPoint.y,
          valid = wh.surfaceValidTrack,
          load = try(function() return wh.load end),
        }
      end
      cars[#cars + 1] = {
        i = i,
        pos = { c.position.x, c.position.y, c.position.z },
        look = try(function() return { c.look.x, c.look.y, c.look.z } end),
        kmh = c.speedKmh,
        spline = try(function() return c.splinePosition end),
        laps = try(function() return c.lapCount end),
        ai = try(function() return c.isAIControlled end),
        out = c.wheelsOutside,
        wheels = wheels,
        dmg = try(function() return { c.damage[0], c.damage[1], c.damage[2], c.damage[3], c.damage[4] } end),
      }
    end
  end
  samples[#samples + 1] = {
    t = t,
    fps = try(function() return sim.fps end),
    menu = sim.isInMainMenu,
    cars = cars,
  }
end

function script.update(dt)
  if not active then return end
  t = t + dt
  if t >= next_sample then
    sample()
    next_sample = t + ((t < 15) and 0.25 or 1.0)
  end
  if t >= next_save then
    io.save(dir .. '/watch.json', JSON.stringify({ samples = samples, csp = ac.getPatchVersionCode() }))
    next_save = t + 5
  end
end

function script.windowMain(dt)
  ui.text('mapcreator watch: ' .. (active and (#samples .. ' samples') or 'off'))
end
