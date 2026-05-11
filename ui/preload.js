const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('seraApi', {
  loadProvinceMap: () => ipcRenderer.invoke('ui:load-province-map'),
  bootstrap: () => ipcRenderer.invoke('ui:bootstrap'),
  loadProvinceTrends: (payload) => ipcRenderer.invoke('ui:load-province-trends', payload),
  simulateNextYear: (payload) => ipcRenderer.invoke('ui:simulate-next-year', payload),
  onSimulationLog: (callback) => {
    const listener = (_event, message) => callback(message)
    ipcRenderer.on('ui:simulation-log', listener)
    return () => ipcRenderer.removeListener('ui:simulation-log', listener)
  },
})