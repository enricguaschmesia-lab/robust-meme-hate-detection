param(
    [ValidateSet("cs5242", "emily49_metadata")]
    [string]$Source = "cs5242",
    [string]$OutputDir = "data/raw/hateful_memes",
    [int]$MaxImages = 0,
    [int]$Retries = 3
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$sources = @{
    cs5242 = @{
        repo = "cs5242-hateful-memes/hateful-memes-data"
        jsonl = @("train.jsonl", "dev_seen.jsonl", "dev_unseen.jsonl", "test_seen.jsonl", "test_unseen.jsonl")
    }
    emily49_metadata = @{
        repo = "emily49/hateful-memes"
        jsonl = @("train.jsonl", "dev.jsonl", "test.jsonl")
    }
}

function Get-ResolveUrl {
    param([string]$Repo, [string]$Path)
    return "https://huggingface.co/datasets/$Repo/resolve/main/$Path"
}

function Save-Url {
    param([string]$Url, [string]$Destination, [int]$Retries, [switch]$IgnoreErrors)
    if ((Test-Path -LiteralPath $Destination) -and ((Get-Item -LiteralPath $Destination).Length -gt 0)) {
        return $true
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
    $tmp = "$Destination.tmp"
    for ($attempt = 1; $attempt -le $Retries; $attempt++) {
        try {
            Invoke-WebRequest -Uri $Url -UseBasicParsing -OutFile $tmp
            Move-Item -LiteralPath $tmp -Destination $Destination -Force
            return $true
        } catch {
            if (Test-Path -LiteralPath $tmp) {
                Remove-Item -LiteralPath $tmp -Force
            }
            if ($attempt -eq $Retries) {
                if ($IgnoreErrors) {
                    return $false
                }
                throw
            }
            Start-Sleep -Seconds (2 * $attempt)
        }
    }
    return $false
}

$selected = $sources[$Source]
$repo = $selected.repo
$jsonlFiles = $selected.jsonl

Write-Host "Selected source: $Source ($repo)"
Write-Host "Output directory: $OutputDir"
if ($Source -eq "emily49_metadata") {
    Write-Warning "emily49/hateful-memes appears to expose metadata only. Use cs5242 for images."
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

@("README.md", "LICENSE.txt") + $jsonlFiles | ForEach-Object {
    Write-Host "Downloading $_"
    [void](Save-Url -Url (Get-ResolveUrl -Repo $repo -Path $_) -Destination (Join-Path $OutputDir $_) -Retries $Retries)
}

$imageRefs = New-Object System.Collections.Generic.HashSet[string]
foreach ($jsonl in $jsonlFiles) {
    $path = Join-Path $OutputDir $jsonl
    if (-not (Test-Path -LiteralPath $path)) {
        continue
    }
    Get-Content -LiteralPath $path | ForEach-Object {
        if ([string]::IsNullOrWhiteSpace($_)) {
            return
        }
        $item = $_ | ConvertFrom-Json
        $img = $item.img
        if (-not $img) {
            $img = $item.image
        }
        if ($img) {
            [void]$imageRefs.Add(($img -replace "\\", "/"))
        }
    }
}

$images = @($imageRefs | Sort-Object)
if ($MaxImages -gt 0) {
    $images = @($images | Select-Object -First $MaxImages)
}

Write-Host "Downloading $($images.Count) image files"
for ($i = 0; $i -lt $images.Count; $i++) {
    $imageRef = $images[$i]
    $n = $i + 1
    if ($n -eq 1 -or $n % 100 -eq 0 -or $n -eq $images.Count) {
        Write-Host "  [$n/$($images.Count)] $imageRef"
    }
    $ok = Save-Url -Url (Get-ResolveUrl -Repo $repo -Path $imageRef) -Destination (Join-Path $OutputDir $imageRef) -Retries $Retries -IgnoreErrors
    if (-not $ok) {
        Write-Warning "failed or missing after retries, skipped for this pass: $imageRef"
    }
}

Write-Host "Download complete."
