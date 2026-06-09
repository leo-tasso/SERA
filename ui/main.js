const { app, BrowserWindow, ipcMain } = require('electron')
const fs = require('fs')
const path = require('path')
const { spawn } = require('child_process')

const REPO_ROOT = path.resolve(__dirname, '..')
const BRIDGE_PATH = path.join(__dirname, 'backend_bridge.py')
const PROGRESS_PREFIX = '@@PROGRESS@@'
const DEFAULT_MAP_PATHS = [
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
    const proc = spawn(getPythonExecutable(), [BRIDGE_PATH, command], {
      cwd: REPO_ROOT,
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