param(
    [string]$HostName,

    [int]$Port = 0,

    [string]$UserName = "root",
    [string]$IdentityFile,
    [string]$RemoteDataRoot = "/workspace/heretic-data/qwen36-v3",
    [string]$ManifestPath,
    [string]$LocalDataRoot,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
$bundle = Split-Path -Parent $MyInvocation.MyCommand.Path
$manifestFile = if ($ManifestPath) {
    $ManifestPath
} else {
    Join-Path $bundle "data_manifest.json"
}
$manifest = Get-Content -LiteralPath $manifestFile -Raw | ConvertFrom-Json

if ($RemoteDataRoot -notmatch '^/[A-Za-z0-9._/-]+$') {
    throw "RemoteDataRoot contains unsupported characters: $RemoteDataRoot"
}

$sources = @{}
if ($LocalDataRoot) {
    foreach ($record in $manifest.files) {
        $sources[$record.name] = Join-Path $LocalDataRoot $record.name
    }
} else {
    $sources = @{
        "direction_safe.jsonl" = "F:\AI\hf_originals\heretic_out\research\runs\adaptive_search_v2\data\direction_safe.jsonl"
        "direction_unsafe.jsonl" = "F:\AI\hf_originals\heretic_out\research\runs\adaptive_search_v2\data\direction_unsafe.jsonl"
        "search_unsafe.jsonl" = "F:\AI\hf_originals\heretic_out\research\runs\adaptive_search_v2\data\search_unsafe.jsonl"
        "prototypes.jsonl" = "F:\AI\hf_originals\heretic_out\research\results\refusal_classifier_eval\sparse_geometry_bank_v1\prototypes.jsonl"
    }
}

$validatedBytes = 0L
foreach ($record in $manifest.files) {
    $source = $sources[$record.name]
    if (-not $source -or -not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Missing local source for $($record.name): $source"
    }
    $item = Get-Item -LiteralPath $source
    if ($item.Length -ne [long]$record.bytes) {
        throw "Size mismatch for $source"
    }
    $digest = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($digest -ne $record.sha256) {
        throw "SHA-256 mismatch for $source"
    }
    $validatedBytes += $item.Length
}

if ($ValidateOnly) {
    [ordered]@{
        status = "PASS"
        files = @($manifest.files).Count
        bytes = $validatedBytes
    } | ConvertTo-Json -Compress
    return
}

if ([string]::IsNullOrWhiteSpace($HostName) -or $Port -le 0) {
    throw "HostName and a positive Port are required unless -ValidateOnly is used"
}

$destination = "$UserName@$HostName"
$sshArgs = @("-p", $Port)
$scpArgs = @("-P", $Port)
if ($IdentityFile) {
    $sshArgs += @("-i", $IdentityFile)
    $scpArgs += @("-i", $IdentityFile)
}

& ssh @sshArgs $destination "mkdir -p -- '$RemoteDataRoot'"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create remote data directory"
}

foreach ($record in $manifest.files) {
    $source = $sources[$record.name]
    & scp @scpArgs $source "${destination}:$RemoteDataRoot/$($record.name)"
    if ($LASTEXITCODE -ne 0) {
        throw "Upload failed for $($record.name)"
    }
    Write-Host "Uploaded $($record.name) ($($record.bytes) bytes)"
}
