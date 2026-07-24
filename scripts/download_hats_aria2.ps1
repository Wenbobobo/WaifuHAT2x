$ErrorActionPreference = 'Stop'

$Project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$WslDistro = if ($env:WAIFUHAT_WSL_DISTRO) { $env:WAIFUHAT_WSL_DISTRO } else { 'Ubuntu' }
$Aria = Get-Command aria2c.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source
if (-not $Aria) {
    $WingetRoot = Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Packages'
    $Aria = Get-ChildItem -Path $WingetRoot -Filter aria2c.exe -Recurse -ErrorAction SilentlyContinue |
        Where-Object FullName -Like '*aria2.aria2_*' |
        Select-Object -Last 1 -ExpandProperty FullName
}
if (-not $Aria) {
    Write-Warning 'aria2c is unavailable; WSL will fall back to gdown.'
    exit 0
}

Write-Host "Using aria2: $Aria"
$TicketJson = & wsl.exe -d $WslDistro --cd $Project -- bash scripts/project_python.sh scripts/gdrive_tickets.py
if ($LASTEXITCODE -ne 0) { throw 'Unable to obtain Google Drive download tickets.' }
$Tickets = $TicketJson | ConvertFrom-Json

foreach ($Ticket in $Tickets) {
    $Target = Join-Path (Join-Path $Project 'models') ($Ticket.filename -replace '/', '\')
    if (Test-Path -LiteralPath $Target -PathType Leaf) {
        Write-Host "Present: $($Ticket.filename)"
        continue
    }
    $Directory = Split-Path -Parent $Target
    New-Item -ItemType Directory -Force -Path $Directory | Out-Null
    $Partial = "$Target.part"
    if (-not (Test-Path -LiteralPath $Partial)) {
        $Legacy = Get-ChildItem -Path "$Partial*.part" -File -ErrorAction SilentlyContinue |
            Sort-Object Length -Descending | Select-Object -First 1
        if ($Legacy) { Move-Item -LiteralPath $Legacy.FullName -Destination $Partial }
    }
    Write-Host "Downloading $($Ticket.name): $([math]::Round([double]$Ticket.size / 1MB, 1)) MiB"
    & $Aria `
        '--continue=true' `
        '--max-connection-per-server=8' `
        '--split=8' `
        '--min-split-size=1M' `
        '--file-allocation=none' `
        '--auto-file-renaming=false' `
        '--allow-overwrite=true' `
        '--summary-interval=5' `
        "--dir=$Directory" `
        "--out=$([IO.Path]::GetFileName($Partial))" `
        "--header=Cookie: $($Ticket.cookies)" `
        '--user-agent=WaifuHAT2x/1.0' `
        $Ticket.url
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "aria2 failed TLS/download validation for $($Ticket.name); WSL will resume with gdown."
        exit 0
    }
    $Actual = (Get-Item -LiteralPath $Partial).Length
    if ([long]$Ticket.size -gt 0 -and $Actual -ne [long]$Ticket.size) {
        throw "Size mismatch for $($Ticket.name): $Actual/$($Ticket.size)"
    }
    Move-Item -LiteralPath $Partial -Destination $Target -Force
}
