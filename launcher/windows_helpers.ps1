#Requires -Version 5.1
<#
  Small helpers for the desktop launcher.

  Commands:
    parent-pid       Print the parent process id (the cmd window that called us).
    port-state       Write FREE, "OURS <pid>", "OTHER <pid>", or UNKNOWN.
    wait-and-open    Open the app URL once Streamlit's health check answers.
    resolve-python   Write .venv\Scripts\python.exe, else venv\Scripts\python.exe, else MISSING.
    append-log       Append DFS_LOG_MSG to -LogPath. Retries briefly, then gives up quietly.
    self-test        Check path matching. Does not start a server or create shortcuts.

  port-state never stops a process. UNKNOWN and OTHER both mean "not confirmed
  as this app", so the caller must not kill the listener and must not treat it
  as permission to start a second server.
#>
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('parent-pid', 'port-state', 'wait-and-open', 'resolve-python', 'append-log', 'self-test')]
    [string]$Command,

    [string]$ProjectRoot,
    [int]$Port = 8501,
    [string]$Url = 'http://localhost:8501/',
    [int]$TimeoutSeconds = 90,
    [string]$OutFile,
    [string]$LogPath
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Add-LogLine([string]$Path, [string]$Message) {
    # Share the file for concurrent appends. A few short retries cover a
    # transient OneDrive lock. A long-lived holder (Streamlit's launch log)
    # is not waited out: logging gives up and the caller still prints status.
    if ([string]::IsNullOrWhiteSpace($Path)) { return }
    $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $line = '[' + $stamp + '] ' + $Message + [Environment]::NewLine
    $enc = New-Object System.Text.UTF8Encoding $false
    $bytes = $enc.GetBytes($line)
    $maxAttempts = 4
    for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
        $stream = $null
        try {
            $parentDir = Split-Path -Parent $Path
            if ($parentDir -and -not (Test-Path -LiteralPath $parentDir)) {
                New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
            }
            $stream = New-Object System.IO.FileStream(
                $Path,
                [System.IO.FileMode]::Append,
                [System.IO.FileAccess]::Write,
                [System.IO.FileShare]::ReadWrite)
            $stream.Write($bytes, 0, $bytes.Length)
            $stream.Flush()
            return
        } catch {
            if ($attempt -ge $maxAttempts) { return }
            Start-Sleep -Milliseconds 200
        } finally {
            if ($null -ne $stream) { $stream.Dispose() }
        }
    }
}

function Write-LaunchLog([string]$Message) {
    Add-LogLine -Path $LogPath -Message $Message
}

function Get-NormalizedRoot([string]$PathText) {
    if ([string]::IsNullOrWhiteSpace($PathText)) { return '' }
    return $PathText.Trim().TrimEnd('\').TrimEnd('/').Replace('/', '\').ToLowerInvariant()
}

function Test-PathHasRootBoundary([string]$Haystack, [string]$RootLower) {
    if ([string]::IsNullOrEmpty($Haystack) -or [string]::IsNullOrEmpty($RootLower)) {
        return $false
    }
    $start = 0
    while ($start -lt $Haystack.Length) {
        $index = $Haystack.IndexOf($RootLower, $start, [System.StringComparison]::Ordinal)
        if ($index -lt 0) { return $false }
        $after = $index + $RootLower.Length
        if ($after -ge $Haystack.Length) { return $true }
        $ch = $Haystack[$after]
        if ($ch -eq '\' -or $ch -eq '"' -or $ch -eq "'" -or [char]::IsWhiteSpace($ch)) {
            return $true
        }
        $start = $after
    }
    return $false
}

function Test-CommandIsOurApp {
    param(
        [string]$CommandLine,
        [string]$ExecutablePath,
        [string]$AppRoot
    )
    if ([string]::IsNullOrWhiteSpace($CommandLine) -or [string]::IsNullOrWhiteSpace($AppRoot)) {
        return $false
    }
    $cmd = $CommandLine.ToLowerInvariant()
    if ($cmd.IndexOf('streamlit') -lt 0) { return $false }
    if ($cmd.IndexOf('app.py') -lt 0) { return $false }

    $rootLower = Get-NormalizedRoot $AppRoot
    if ([string]::IsNullOrEmpty($rootLower)) { return $false }
    $cmdNorm = $CommandLine.Replace('/', '\').ToLowerInvariant()
    $exeNorm = ''
    if (-not [string]::IsNullOrWhiteSpace($ExecutablePath)) {
        $exeNorm = $ExecutablePath.Replace('/', '\').ToLowerInvariant()
    }
    $dotVenv = $rootLower + '\.venv\scripts\python.exe'
    $plainVenv = $rootLower + '\venv\scripts\python.exe'
    if ($exeNorm -eq $dotVenv -or $exeNorm -eq $plainVenv) { return $true }
    if (Test-PathHasRootBoundary -Haystack $cmdNorm -RootLower $rootLower) { return $true }
    if (Test-PathHasRootBoundary -Haystack $exeNorm -RootLower $rootLower) { return $true }
    return $false
}

function Resolve-LauncherPython([string]$AppRoot) {
    if ([string]::IsNullOrWhiteSpace($AppRoot)) { return $null }
    $root = $AppRoot.Trim().TrimEnd('\').TrimEnd('/')
    # .venv wins when both folders exist. venv is the fallback.
    foreach ($folder in @('.venv', 'venv')) {
        $candidate = Join-Path (Join-Path (Join-Path $root $folder) 'Scripts') 'python.exe'
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    return $null
}

function Write-AsciiFile([string]$Path, [string]$Text) {
    $line = $Text.Trim()
    if ([string]::IsNullOrWhiteSpace($Path)) {
        [Console]::Out.WriteLine($line)
        return
    }
    $parentDir = Split-Path -Parent $Path
    if ($parentDir -and -not (Test-Path -LiteralPath $parentDir)) {
        New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
    }
    $enc = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($Path, $line + "`r`n", $enc)
}

function Write-PortState([string]$Text) {
    $line = $Text.Trim()
    if (-not [string]::IsNullOrWhiteSpace($OutFile)) {
        $parent = Split-Path -Parent $OutFile
        if ($parent -and -not (Test-Path -LiteralPath $parent)) {
            New-Item -ItemType Directory -Path $parent -Force | Out-Null
        }
        $enc = New-Object System.Text.UTF8Encoding $false
        [System.IO.File]::WriteAllText($OutFile, $line + "`r`n", $enc)
    } else {
        [Console]::Out.WriteLine($line)
    }
    Write-LaunchLog ("port-state: " + $line)
}

function Get-ListenerPids([int]$ListenPort) {
    $found = @{
        Status = 'unknown'
        Pids   = @()
    }
    $haveCmdlet = $null -ne (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue)
    if ($haveCmdlet) {
        try {
            $conns = @(Get-NetTCPConnection -LocalPort $ListenPort -State Listen -ErrorAction Stop)
            $ids = @()
            foreach ($conn in $conns) {
                $ids += [int]$conn.OwningProcess
            }
            $found.Status = 'ok'
            $found.Pids = @($ids | Select-Object -Unique)
            return $found
        } catch {
            $category = ''
            if ($_.CategoryInfo) { $category = [string]$_.CategoryInfo.Category }
            $message = ''
            if ($_.Exception) { $message = [string]$_.Exception.Message }
            # An empty result is an ObjectNotFound / "No matching" error on Windows PowerShell.
            if ($category -eq 'ObjectNotFound' -or $message -match 'No matching') {
                $found.Status = 'ok'
                $found.Pids = @()
                return $found
            }
            Write-LaunchLog ("WARN: Get-NetTCPConnection failed: " + $message)
        }
    }

    $netstat = $null
    try {
        $netstat = @(& netstat.exe -ano -p TCP)
    } catch {
        Write-LaunchLog ("WARN: netstat could not be run: " + $_)
        return $found
    }
    if (-not $netstat -or $netstat.Count -eq 0) {
        Write-LaunchLog 'WARN: netstat returned no output. Port owner is unknown.'
        return $found
    }

    $ids = @()
    foreach ($raw in $netstat) {
        $trim = ([string]$raw).Trim()
        if ($trim.Length -eq 0) { continue }
        if ($trim.StartsWith('Proto') -or $trim.StartsWith('Active')) { continue }
        $parts = @($trim -split '\s+')
        if ($parts.Count -lt 4) { continue }
        # TCP lines: Proto Local Foreign State PID. LISTENING foreign address ends in :0.
        if ($parts[0] -ne 'TCP') { continue }
        $local = [string]$parts[1]
        $foreign = [string]$parts[2]
        $pidToken = [string]$parts[$parts.Count - 1]
        if ($local -notmatch (':{0}$' -f $ListenPort)) { continue }
        if ($foreign -notmatch ':(0|\*)$') { continue }
        if ($pidToken -notmatch '^\d+$') { continue }
        $ids += [int]$pidToken
    }
    $found.Status = 'ok'
    $found.Pids = @($ids | Select-Object -Unique)
    return $found
}

function Invoke-PortState {
    if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
        Write-PortState 'UNKNOWN'
        Write-LaunchLog 'WARN: port check was called without a project root.'
        return
    }
    $listen = Get-ListenerPids -ListenPort $Port
    if ($listen.Status -ne 'ok') {
        Write-PortState 'UNKNOWN'
        return
    }
    if (-not $listen.Pids -or $listen.Pids.Count -eq 0) {
        Write-PortState 'FREE'
        return
    }

    $sawOurs = $false
    $sawOther = $false
    $oursPid = 0
    $otherPid = 0
    foreach ($procId in $listen.Pids) {
        if ([int]$procId -le 0) {
            $sawOther = $true
            $otherPid = [int]$procId
            continue
        }
        $proc = $null
        try {
            $proc = Get-CimInstance Win32_Process -Filter ("ProcessId=" + [int]$procId) -ErrorAction Stop
        } catch {
            $proc = $null
            Write-LaunchLog ("WARN: could not read process " + $procId + ": " + $_)
        }
        if (-not $proc) {
            # Missing command line is not proof the port is free or that the
            # listener is this app. Fail closed.
            $sawOther = $true
            $otherPid = [int]$procId
            continue
        }
        $isOurs = Test-CommandIsOurApp -CommandLine ([string]$proc.CommandLine) -ExecutablePath ([string]$proc.ExecutablePath) -AppRoot $ProjectRoot
        if ($isOurs) {
            $sawOurs = $true
            $oursPid = [int]$procId
        } else {
            $sawOther = $true
            $otherPid = [int]$procId
            Write-LaunchLog ("Port listener is not this app. PID " + $procId)
        }
    }

    if ($sawOther -and $sawOurs) {
        Write-PortState 'UNKNOWN'
        return
    }
    if ($sawOther) {
        Write-PortState ("OTHER " + $otherPid)
        return
    }
    if ($sawOurs) {
        Write-PortState ("OURS " + $oursPid)
        return
    }
    Write-PortState 'FREE'
}

function Invoke-WaitAndOpen {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $healthUrls = @(
        ('http://127.0.0.1:{0}/_stcore/health' -f $Port),
        ('http://127.0.0.1:{0}/health' -f $Port),
        ('http://localhost:{0}/_stcore/health' -f $Port)
    )
    while ((Get-Date) -lt $deadline) {
        foreach ($health in $healthUrls) {
            try {
                $resp = Invoke-WebRequest -Uri $health -UseBasicParsing -TimeoutSec 2
                $body = ''
                if ($resp.Content) { $body = [string]$resp.Content }
                if ($resp.StatusCode -eq 200 -and $body.Trim().ToLowerInvariant() -eq 'ok') {
                    Start-Process $Url
                    Write-LaunchLog ("Opened browser at " + $Url)
                    return
                }
            } catch {
            }
        }
        Start-Sleep -Seconds 1
    }
    Write-LaunchLog ("WARN: Streamlit did not answer on port " + $Port + " within " + $TimeoutSeconds + " seconds. Browser was not opened.")
}

function Invoke-SelfTest {
    $root = 'C:\OneDrive\Python Scripts\dfs-lineup-generator'
    $venv = $root + '\.venv\Scripts\python.exe'
    $cases = @(
        @{
            Name = 'dot venv streamlit'
            Expect = $true
            CommandLine = ($venv + ' -m streamlit run app.py --server.port=8501')
            ExecutablePath = $venv
        },
        @{
            Name = 'plain venv streamlit'
            Expect = $true
            CommandLine = (($root + '\venv\Scripts\python.exe') + ' -m streamlit run app.py')
            ExecutablePath = ($root + '\venv\Scripts\python.exe')
        },
        @{
            Name = 'sibling plain venv'
            Expect = $false
            CommandLine = 'C:\OneDrive\Python Scripts\dfs-lineup-generator-other\venv\Scripts\python.exe -m streamlit run app.py'
            ExecutablePath = 'C:\OneDrive\Python Scripts\dfs-lineup-generator-other\venv\Scripts\python.exe'
        },
        @{
            Name = 'quoted path with spaces'
            Expect = $true
            CommandLine = ('"' + $venv + '" -m streamlit run app.py')
            ExecutablePath = $venv
        },
        @{
            Name = 'longer sibling folder'
            Expect = $false
            CommandLine = 'C:\OneDrive\Python Scripts\dfs-lineup-generator-other\.venv\Scripts\python.exe -m streamlit run app.py'
            ExecutablePath = 'C:\OneDrive\Python Scripts\dfs-lineup-generator-other\.venv\Scripts\python.exe'
        },
        @{
            Name = 'unrelated port listener'
            Expect = $false
            CommandLine = 'C:\Python\python.exe -m http.server 8501'
            ExecutablePath = 'C:\Python\python.exe'
        },
        @{
            Name = 'other streamlit app'
            Expect = $false
            CommandLine = 'C:\other\python.exe -m streamlit run app.py'
            ExecutablePath = 'C:\other\python.exe'
        },
        @{
            Name = 'relative app.py without project path'
            Expect = $false
            CommandLine = 'C:\Python\python.exe -m streamlit run app.py'
            ExecutablePath = 'C:\Python\python.exe'
        },
        @{
            Name = 'missing app.py'
            Expect = $false
            CommandLine = ($venv + ' -m streamlit --version')
            ExecutablePath = $venv
        }
    )
    foreach ($case in $cases) {
        $actual = Test-CommandIsOurApp -CommandLine $case.CommandLine -ExecutablePath $case.ExecutablePath -AppRoot $root
        if ($actual -ne $case.Expect) {
            throw ("self-test failed: " + $case.Name)
        }
    }
    [Console]::Out.WriteLine('self-test ok')
}

switch ($Command) {
    'parent-pid' {
        $self = Get-CimInstance Win32_Process -Filter ("ProcessId=" + $PID)
        if (-not $self -or -not $self.ParentProcessId) { exit 1 }
        $text = [string][int]$self.ParentProcessId
        # Write a file when asked. cmd.exe misreads UTF-16 pipeline output.
        if (-not [string]::IsNullOrWhiteSpace($OutFile)) {
            $parentDir = Split-Path -Parent $OutFile
            if ($parentDir -and -not (Test-Path -LiteralPath $parentDir)) {
                New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
            }
            $enc = New-Object System.Text.UTF8Encoding $false
            [System.IO.File]::WriteAllText($OutFile, $text + "`r`n", $enc)
        } else {
            [Console]::Out.WriteLine($text)
        }
        exit 0
    }
    'port-state' {
        Invoke-PortState
        exit 0
    }
    'wait-and-open' {
        Invoke-WaitAndOpen
        exit 0
    }
    'append-log' {
        $text = [string]$env:DFS_LOG_MSG
        Add-LogLine -Path $LogPath -Message $text
        exit 0
    }
    'resolve-python' {
        $chosen = Resolve-LauncherPython $ProjectRoot
        if ([string]::IsNullOrWhiteSpace($chosen)) {
            Write-AsciiFile $OutFile 'MISSING'
            Write-LaunchLog 'No .venv\Scripts\python.exe or venv\Scripts\python.exe was found.'
        } else {
            Write-AsciiFile $OutFile $chosen
            Write-LaunchLog ("Selected Python: " + $chosen)
        }
        exit 0
    }
    'self-test' {
        Invoke-SelfTest
        exit 0
    }
}
