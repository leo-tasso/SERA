const { app, BrowserWindow, ipcMain } = require('electron')
const fs = require('fs')
const path = require('path')
const { spawn } = require('child_process')

// In a packaged build the Python backend, trained twin, data and GeoJSON are
// shipped under <resources>/payload (see "extraResources" in package.json); in
// development they live in the repository tree. The payload mirrors the repo
// layout so backend_bridge.py (parents[1]/src) and sera.config (PROJECT_ROOT =
// three levels up from config.py) resolve their paths unchanged either way.
// REPO_ROOT is both the spawned Python process's cwd and the model-path base.
const PAYLOAD_ROOT = app.isPackaged
  ? path.join(process.resourcesPath, 'payload')
  : path.resolve(__dirname, '..')
const BACKEND_DIR = app.isPackaged ? path.join(PAYLOAD_ROOT, 'ui') : __dirname
const REPO_ROOT = PAYLOAD_ROOT
const BRIDGE_PATH = path.join(BACKEND_DIR, 'backend_bridge.py')
const PROGRESS_PREFIX = '@@PROGRESS@@'
const DEFAULT_MAP_PATHS = [
  path.join(BACKEND_DIR, 'province_provinces.geojson'),
  path.join(__dirname, 'province_provinces.geojson'),
  path.join(process.env.USERPROFILE || 'C:\\Users\\Leonardo', 'Downloads', 'SERA', 'ui', 'province_provinces.geojson'),
]

function getPythonExecutable() {
  if (process.platform === 'win32') {
    const venvPython = path.join(REPO_ROOT, '.venv', 'Scripts', 'python.exe')
    if (fs.existsSync(venvPython)) {
      return venvPython
    }
    return 'python'
  }

  const unixVenvPython = path.join(REPO_ROOT, '.venv', 'bin', 'python')
  if (fs.existsSync(unixVenvPython)) {
    return unixVenvPython
  }
  return 'python3'
}

// How to launch the Python backend bridge.
//   * Packaged build: a self-contained PyInstaller executable shipped under
//     <resources>/payload/backend, so the end user needs no Python install.
//   * Development (`npm start`): the system / .venv Python running the bridge
//     script directly from the repo.
// Both speak the same protocol — argv[1] is the command, the JSON payload comes
// in on stdin — so the rest of the code is identical for either launcher.
function getBackendLauncher() {
  if (app.isPackaged) {
    const exeName = process.platform === 'win32' ? 'backend_bridge.exe' : 'backend_bridge'
    return { command: path.join(PAYLOAD_ROOT, 'backend', exeName), prefixArgs: [] }
  }
  return { command: getPythonExecutable(), prefixArgs: [BRIDGE_PATH] }
}

function resolveMapPath() {
  for (const candidate of DEFAULT_MAP_PATHS) {
    if (candidate && fs.existsSync(candidate)) {
      return candidate
    }
  }

  throw new Error(
    `Province GeoJSON not found. Expected it at ${DEFAULT_MAP_PATHS[0]}.`,
  )
}

function parseBridgeOutput(stdout, stderr, status) {
  if (status !== 0) {
    throw new Error(stderr || stdout || `Bridge exited with code ${status}`)
  }

  try {
    return JSON.parse(stdout)
  } catch (error) {
    throw new Error(`Failed to parse bridge output: ${stderr || stdout || error.message}`)
  }
}

function runBridgeAsync(command, payload, webContents) {
  return new Promise((resolve, reject) => {
    const { command: launcher, prefixArgs } = getBackendLauncher()
    const proc = spawn(launcher, [...prefixArgs, command], {
      cwd: REPO_ROOT,
      // The bundled data/ and twin model live next to the executable in the
      // payload, not at sera.config's __file__-derived path; point the backend
      // there. Harmless in dev (it already resolves to the same repo root).
      env: { ...process.env, SERA_PROJECT_ROOT: REPO_ROOT },
      stdio: ['pipe', 'pipe', 'pipe'],
    })

    let stdout = ''
    let stderr = ''
    let stderrBuffer = ''

    proc.stdout.on('data', (chunk) => {
      stdout += chunk.toString()
    })

    const sendToRenderer = (channel, payload) => {
      if (webContents && !webContents.isDestroyed()) {
        webContents.send(channel, payload)
      }
    }

    const consumeStderrLine = (line) => {
      if (line.startsWith(PROGRESS_PREFIX)) {
        try {
          sendToRenderer('ui:simulation-progress', JSON.parse(line.slice(PROGRESS_PREFIX.length)))
        } catch (_error) {
          // Malformed progress line: ignore rather than pollute the log.
        }
        return
      }
      stderr += `${line}\n`
      sendToRenderer('ui:simulation-log', `${line}\n`)
    }

    proc.stderr.on('data', (chunk) => {
      stderrBuffer += chunk.toString()
      const lines = stderrBuffer.split(/\r?\n/)
      stderrBuffer = lines.pop()
      lines.forEach(consumeStderrLine)
    })

    proc.on('error', (error) => reject(error))
    proc.on('close', (code) => {
      if (stderrBuffer) {
        consumeStderrLine(stderrBuffer)
        stderrBuffer = ''
      }
      try {
        resolve(parseBridgeOutput(stdout, stderr, code))
      } catch (error) {
        reject(error)
      }
    })

    proc.stdin.write(JSON.stringify(payload || {}))
    proc.stdin.end()
  })
}

function createWindow() {
  const window = new BrowserWindow({
    width: 1500,
    height: 980,
    minWidth: 1180,
    minHeight: 760,
    backgroundColor: '#edf3fb',
    autoHideMenuBar: true,
    icon: path.join(__dirname, 'assets', 'icon.png'),
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  })

  window.loadFile(path.join(__dirname, 'index.html'))
  return window
}

app.whenReady().then(() => {
  createWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow()
    }
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

ipcMain.handle('ui:load-province-map', async () => {
  const mapPath = resolveMapPath()
  const raw = await fs.promises.readFile(mapPath, 'utf8')
  return { mapPath, raw }
})

ipcMain.handle('ui:bootstrap', async () => {
  return runBridgeAsync('bootstrap', {
    baselineYear: 2025,
    modelPath: path.join(REPO_ROOT, 'twin_models.joblib'),
  })
})

ipcMain.handle('ui:load-province-trends', async (_event, payload) => {
  return runBridgeAsync('province-trends', payload)
})

ipcMain.handle('ui:simulate-next-year', async (event, payload) => {
  event.sender.send('ui:simulation-log', 'Running one-year simulation...\n')
  const response = await runBridgeAsync(
    'simulate-next-year',
    {
      ...payload,
      modelPath: path.join(REPO_ROOT, 'twin_models.joblib'),
    },
    event.sender,
  )
  event.sender.send('ui:simulation-log', `Simulation completed for ${response.nextYear}.\n`)
  return response
})

ipcMain.handle('ui:compare-objectives', async (event, payload) => {
  event.sender.send('ui:simulation-log', 'Training the policy model once per ethical objective...\n')
  const response = await runBridgeAsync(
    'compare-objectives',
    {
      ...payload,
      modelPath: path.join(REPO_ROOT, 'twin_models.joblib'),
    },
    event.sender,
  )
  event.sender.send('ui:simulation-log', `Ethics comparison completed through ${response.finalYear}.\n`)
  return response
})

ipcMain.handle('ui:pareto-front', async (event, payload) => {
  event.sender.send('ui:simulation-log', 'Mapping the efficiency-equity frontier with NSGA-II...\n')
  const response = await runBridgeAsync(
    'pareto-front',
    {
      ...payload,
      modelPath: path.join(REPO_ROOT, 'twin_models.joblib'),
    },
    event.sender,
  )
  event.sender.send('ui:simulation-log', `Pareto frontier mapped through ${response.finalYear} (${response.evaluations} rollouts).\n`)
  return response
})

ipcMain.handle('ui:optimize-policy', async (event, payload) => {
  event.sender.send('ui:simulation-log', 'Running policy model over the horizon...\n')
  const response = await runBridgeAsync(
    'optimize-policy',
    {
      ...payload,
      modelPath: path.join(REPO_ROOT, 'twin_models.joblib'),
    },
    event.sender,
  )
  event.sender.send('ui:simulation-log', `Policy run completed through ${response.finalYear}.\n`)
  return response
})