const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('seraApi', {
  loadProvinceMap: () => ipcRenderer.invoke('ui:load-province-map'),
  bootstrap: () => ipcRenderer.invoke('ui:bootstrap'),
  loadProvinceTrends: (payload) => ipcRenderer.invoke('ui:load-province-trends', payload),
  simulateNextYear: (payload) => ipcRenderer.invoke('ui:simulate-next-year', payload),
  optimizePolicy: (payload) => ipcRenderer.invoke('ui:optimize-policy', payload),
  compareObjectives: (payload) => ipcRenderer.invoke('ui:compare-objectives', payload),
  onSimulationLog: (callback) => {
    const listener = (_event, message) => callback(message)
    ipcRenderer.on('ui:simulation-log', listener)
    return () => ipcRenderer.removeListener('ui:simulation-log', listener)
  },
  onSimulationProgress: (callback) => {
    const listener = (_event, progress) => callback(progress)
    ipcRenderer.on('ui:simulation-progress', listener)
    return () => ipcRenderer.removeListener('ui:simulation-progress', listener)
  },
})