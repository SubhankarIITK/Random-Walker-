$ErrorActionPreference = 'Stop'
$projectPath = $PSScriptRoot
$frontendPath = Join-Path $projectPath 'frontend'
$logPath = Join-Path $projectPath '.showcase-logs'
New-Item -ItemType Directory -Force -Path $logPath | Out-Null

function PortReady([int]$port) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $connection = $client.ConnectAsync('127.0.0.1', $port)
        if (-not $connection.Wait(200)) { return $false }
        return $client.Connected
    } catch { return $false }
    finally { $client.Dispose() }
}

if (-not (PortReady 8000)) {
    $venvPython = Join-Path $projectPath '.venv\Scripts\python.exe'
    $pythonPath = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { (Get-Command python -ErrorAction Stop).Source }
    Start-Process -FilePath $pythonPath -ArgumentList '-m uvicorn backend.main:app --host 127.0.0.1 --port 8000' -WorkingDirectory $projectPath -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logPath 'backend.log') -RedirectStandardError (Join-Path $logPath 'backend-errors.log')
}
if (-not (PortReady 5173)) {
    Get-Command npm.cmd -ErrorAction Stop | Out-Null
    Start-Process -FilePath $env:ComSpec -ArgumentList '/c npm.cmd run dev -- --host 127.0.0.1 --port 5173 --strictPort' -WorkingDirectory $frontendPath -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logPath 'frontend.log') -RedirectStandardError (Join-Path $logPath 'frontend-errors.log')
}

$deadline = (Get-Date).AddSeconds(30)
while ((Get-Date) -lt $deadline) {
    if ((PortReady 8000) -and (PortReady 5173)) {
        Start-Process 'http://localhost:5173'
        Write-Host 'Website ready: http://localhost:5173'
        exit 0
    }
    Start-Sleep -Milliseconds 500
}
Write-Host 'Startup did not finish. Check .showcase-logs in the project folder.'
exit 1
