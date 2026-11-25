# Point the four venvs at the Python copied next to the app.
param(
    [Parameter(Mandatory = $true)][string]$AppDir
)
$ErrorActionPreference = "Stop"
$pyHome = Join-Path $AppDir "runtime\python312"
$python = Join-Path $pyHome "python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "runtime Python is missing: $python"
}
$venvs = @(".venv", ".venvs\hyworld", ".venvs\seedvr2", ".venvs\difix")
foreach ($rel in $venvs) {
    $dir = Join-Path $AppDir $rel
    $cfg = Join-Path $dir "pyvenv.cfg"
    if (-not (Test-Path -LiteralPath $cfg)) {
        throw "pyvenv.cfg is missing: $cfg"
    }
    $values = @{
        "home"       = $pyHome
        "executable" = $python
        "command"    = "$python -m venv $dir"
    }
    $seen = @{}
    $out = New-Object System.Collections.Generic.List[string]
    foreach ($line in [IO.File]::ReadAllLines($cfg)) {
        $key = ($line -split "=", 2)[0].Trim()
        if ($values.ContainsKey($key)) {
            $out.Add("$key = $($values[$key])")
            $seen[$key] = $true
        } else {
            $out.Add($line)
        }
    }
    foreach ($key in $values.Keys) {
        if (-not $seen.ContainsKey($key)) {
            $out.Add("$key = $($values[$key])")
        }
    }
    [IO.File]::WriteAllText($cfg, (($out -join "`r`n") + "`r`n"), [Text.Encoding]::ASCII)

    Get-ChildItem -LiteralPath $dir -Recurse -Filter *.pth -File -ErrorAction SilentlyContinue | ForEach-Object {
        $text = [IO.File]::ReadAllText($_.FullName)
        if ($text.Contains("D:\Votion3DGS")) {
            $text = $text.Replace("D:\Votion3DGS", $AppDir)
            [IO.File]::WriteAllText($_.FullName, $text, [Text.Encoding]::ASCII)
            "rewrote $($_.FullName)"
        }
    }
}
"venvs point at $python"
