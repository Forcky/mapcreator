-- mapcreator probe: development-only Custom Shaders Patch app. First run at M3, on CSP 0.2.11.
--
-- Reads Documents/Assetto Corsa/mapcreator_probe/points.json, written by
-- `ac_harness.py probe-points`. Then:
--   1. presses Drive
--   2. teleports the player car to each point
--   3. waits a few frames
--   4. records the car position, each wheel's contact point and surface
--      validity, the lateral offset along the AI spline, and a downward
--      raycast that finds holes in the collision mesh
--   5. writes results.json and, if asked, quits AC
--
-- Written from the CSP Lua SDK docs. The teleport-and-check approach follows
-- stormbuster/assetto-corsa-mcp's MIT-licensed bridge; see THIRD_PARTY_NOTICES.md.

local dir = ac.getFolder(ac.FolderID.ACDocuments) .. '/mapcreator_probe'
-- Wait this long in sim time after each teleport: the car drops ~0.5 m and must settle before it's read.
local SETTLE_S = 1.2

local state = 'loading'
local points, results, index, wait = nil, {}, 1, nil
local options = { quit = false, screenshots_every = 0 }

local function save(name, value)
  io.save(dir .. '/' .. name, JSON.stringify(value))
end

local function vec(t)
  return vec3(t[1], t[2], t[3])
end

local function load_points()
  local raw = io.load(dir .. '/points.json')
  if not raw then return false end
  local data = JSON.parse(raw)
  points = data.points
  options.quit = data.quit or false
  options.screenshots_every = data.screenshots_every or 0
  return points ~= nil and #points > 0
end

local function record(p)
  local car = ac.getCar(0)
  local wheels = {}
  for w = 0, 3 do
    local wheel = car.wheels[w]
    wheels[#wheels + 1] = {
      contact = { wheel.contactPoint.x, wheel.contactPoint.y, wheel.contactPoint.z },
      valid_track = wheel.surfaceValidTrack,
      surface = wheel.surfaceSectionIndex,
    }
  end
  local above = car.position + vec3(0, 20, 0)
  local hit = physics.raycastTrack(above, vec3(0, -1, 0), 60)
  -- Lateral offset and progress along the AI spline; not every CSP version has this call (0.2.11 does).
  local track = ac.worldCoordinateToTrack and ac.worldCoordinateToTrack(car.position) or nil
  -- Frame rate at this point, as CSP reports it; a missing field on the sim struct would throw, hence pcall.
  local ok, fps = pcall(function() return ac.getSim().fps end)
  results[#results + 1] = {
    fps = ok and fps or nil,
    i = index,
    requested = p.pos,
    position = { car.position.x, car.position.y, car.position.z },
    wheels = wheels,
    wheels_outside = car.wheelsOutside,
    raycast_m = hit,
    lateral = track and track.x or nil,
    progress = track and track.z or nil,
  }
end

function script.update(dt)
  if state == 'loading' then
    if load_points() then
      state = 'starting'
      save('status.json', { state = state, points = #points, csp = ac.getPatchVersionCode() })
    end
  elseif state == 'starting' then
    -- CSP's own start button keys on isInMainMenu (the pits setup screen); this works on 0.2.11.
    local sim = ac.getSim()
    if sim.isInMainMenu then
      ac.tryToStart(true)
    else
      state = 'probing'
    end
  elseif state == 'probing' then
    if wait == nil then
      local p = points[index]
      physics.setCarPosition(0, vec(p.pos), vec(p.dir))
      wait = SETTLE_S
    else
      wait = wait - dt
      if wait <= 0 then
        wait = nil
        record(points[index])
        if options.screenshots_every > 0 and (index - 1) % options.screenshots_every == 0 then
          ac.makeScreenshot(dir .. '/shot_' .. index .. '.png', ac.ScreenshotFormat.PNG)
        end
        index = index + 1
        if index > #points then
          state = 'done'
          save('results.json', { results = results, csp = ac.getPatchVersionCode() })
          save('status.json', { state = state, points = #points })
          -- os.remove is a stub inside CSP's sandbox; io.deleteFile is the real one. Without this the
          -- probe would run again (and teleport the car) on every later launch.
          io.deleteFile(dir .. '/points.json')
          if options.quit then ac.shutdownAssettoCorsa() end
        end
      end
    end
  end
end

function script.windowMain(dt)
  ui.text('mapcreator probe: ' .. state)
  if points then ui.text(string.format('point %d of %d', math.min(index, #points), #points)) end
end
