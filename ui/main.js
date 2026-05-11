const { app, BrowserWindow, ipcMain } = require('electron')
const fs = require('fs')
const path = require('path')
const { spawn, spawnSync } = require('child_process')

const REPO_ROOT = path.resolve(__dirname, '..')
const BRIDGE_PATH = path.join(__dirname, 'backend_bridge.py')
const DEFAULT_MAP_PATHS = [
  path.join(process.env.USERPROFILE || 'C:\\Users\\Leonardo', 'Downloads', 'SERA', 'ui', 'province_provinces.geojson'),
  'C:\\Users\\Leonardo\\Downloads\\SERA\\ui\\province_provinces.geojson',
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

function resolveMapPath() {
  for (const candidate of DEFAULT_MAP_PATHS) {
    if (candidate && fs.existsSync(candidate)) {
      return candidate
    }
  }

  throw new Error(
    'Province GeoJSON not found. Expected it at C:\\Users\\Leonardo\\Downloads\\SERA\\ui\\province_provinces.geojson.',
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

function runBridgeSync(command, payload) {
  const result = spawnSync(getPythonExecutable(), [BRIDGE_PATH, command], {
    cwd: REPO_ROOT,
    encoding: 'utf8',
    input: JSON.stringify(payload || {}),
    maxBuffer: 64 * 1024 * 1024,
  })

  if (result.error) {
    throw result.error
  }

  return parseBridgeOutput(result.stdout, result.stderr, result.status)
}

function runBridgeAsync(command, payload, webContents) {
  return new Promise((resolve, reject) => {
    const proc = spawn(getPythonExecutable(), [BRIDGE_PATH, command], {
      cwd: REPO_ROOT,
      stdio: ['pipe', 'pipe', 'pipe'],
    })

    let stdout = ''
    let stderr = ''

    proc.stdout.on('data', (chunk) => {
      stdout += chunk.toString()
    })

    proc.stderr.on('data', (chunk) => {
      const text = chunk.toString()
      stderr += text
      if (webContents && !webContents.isDestroyed()) {
        webContents.send('ui:simulation-log', text)
      }
    })

    proc.on('error', (error) => reject(error))
    proc.on('close', (code) => {
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
  return runBridgeSync('bootstrap', {
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