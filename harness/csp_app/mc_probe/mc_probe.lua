-- mapcreator probe: development-only Custom Shaders Patch app. UNTESTED until M3.
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
local SETTLE_FRAMES = 20

local state = 'loading'
local points, results, index, wait = nil, {}, 1, 0
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
  local track = ac.worldCoordinateToTrack(car.position)
  results[#results + 1] = {
    i = index,
    requested = p.pos,
    position = { car.position.x, car.position.y, car.position.z },
    wheels = wheels,
    wheels_outside = car.wheelsOutside,
    raycast_m = hit,
    lateral = track.x,
    progress = track.z,
  }
end

function script.update(dt)
  if state == 'loading' then
    if load_points() then
      state = 'starting'
      save('status.json', { state = state, points = #points, csp = ac.getPatchVersionCode() })
    end
  elseif state == 'starting' then
    local sim = ac.getSim()
    if not sim.isSessionStarted then
      ac.tryToStart(true)
    else
      state = 'probing'
    end
  elseif state == 'probing' then
    if wait == 0 then
      local p = points[index]
      physics.setCarPosition(0, vec(p.pos), vec(p.dir))
      wait = SETTLE_FRAMES
    else
      wait = wait - 1
      if wait == 0 then
        record(points[index])
        if options.screenshots_every > 0 and (index - 1) % options.screenshots_every == 0 then
          ac.makeScreenshot(dir .. '/shot_' .. index .. '.png', ac.ScreenshotFormat.PNG)
        end
        index = index + 1
        if index > #points then
          state = 'done'
          save('results.json', { results = results, csp = ac.getPatchVersionCode() })
          save('status.json', { state = state, points = #points })
          os.remove(dir .. '/points.json')
          if options.quit then ac.shutdownAssettoCorsa() end
        end
      end
    end
  end
end
