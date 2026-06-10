// Renders assets/icon.svg to assets/icon.png (256x256, transparent background).
// Run from ui/: npx electron tools/render-icon.js
const { app, BrowserWindow } = require('electron')
const fs = require('fs')
const path = require('path')

const SIZE = 256
const svgPath = path.join(__dirname, '..', 'assets', 'icon.svg')
const pngPath = path.join(__dirname, '..', 'assets', 'icon.png')

app.whenReady().then(async () => {
  const svg = fs.readFileSync(svgPath, 'utf8')
  const html = `<!doctype html><html><body style="margin:0;overflow:hidden;background:transparent">${svg.replace('<svg ', '<svg style="display:block" ')}</body></html>`

  const win = new BrowserWindow({
    width: SIZE,
    height: SIZE,
    show: false,
    frame: false,
    transparent: true,
    webPreferences: { offscreen: true },
  })

  await win.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(html)}`)
  // Give the offscreen compositor a frame to paint before capturing.
  await new Promise((resolve) => setTimeout(resolve, 300))
  const image = await win.webContents.capturePage({ x: 0, y: 0, width: SIZE, height: SIZE })
  fs.writeFileSync(pngPath, image.toPNG())
  console.log(`Wrote ${pngPath}`)
  app.quit()
})
