#paste into powershell to use
#sources:
#https://ffmpeg.org/download.html
#https://ffmpeg.org/ffmpeg.html
#https://www.gyan.dev/ffmpeg/builds/

param([string]$InputFolder, [string[]]$InputFiles, [string]$FfmpegPath)

$ErrorActionPreference = 'Stop'

function Get-Converter {
    if ($FfmpegPath) {
        if (-not (Test-Path -LiteralPath $FfmpegPath -PathType Leaf)) { throw 'Nie znaleziono wskazanego ffmpeg.exe.' }
        return $FfmpegPath
    }
    $localExe = Join-Path $PSScriptRoot 'narzedzia\ffmpeg.exe'
    if (Test-Path -LiteralPath $localExe) { return $localExe }
    $existing = Get-Command ffmpeg.exe -ErrorAction SilentlyContinue
    if ($existing) { return $existing.Source }

    Write-Host 'Pierwsze uruchomienie: pobieranie FFmpeg z gyan.dev. To moze potrwac kilka minut.'
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $toolsDir = Join-Path $PSScriptRoot 'narzedzia'
    New-Item -ItemType Directory -Path $toolsDir -Force | Out-Null
    $zipPath = Join-Path $toolsDir ('pobieranie-' + [guid]::NewGuid().ToString('N') + '.zip')
    $tempExe = Join-Path $toolsDir ('ffmpeg-' + [guid]::NewGuid().ToString('N') + '.tmp')
    try {
        $oldProgress = $ProgressPreference
        $ProgressPreference = 'SilentlyContinue'
        try {
            Invoke-WebRequest -UseBasicParsing -Uri 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' -OutFile $zipPath
        } finally { $ProgressPreference = $oldProgress }
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $archive = [IO.Compression.ZipFile]::OpenRead($zipPath)
        try {
            $entry = $archive.Entries | Where-Object { $_.FullName -match '/bin/ffmpeg\.exe$' } | Select-Object -First 1
            if (-not $entry) { throw 'Archiwum nie zawiera ffmpeg.exe.' }
            [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $tempExe)
        } finally { $archive.Dispose() }
        Move-Item -LiteralPath $tempExe -Destination $localExe
    } finally {
        if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath }
        if (Test-Path -LiteralPath $tempExe) { Remove-Item -LiteralPath $tempExe }
    }
    return $localExe
}

try {
    if ($InputFolder -and $InputFiles) { throw 'Podaj folder albo liste plikow.' }
    if (-not $InputFolder -and (-not $InputFiles -or $InputFiles.Count -eq 0)) {
        Add-Type -AssemblyName System.Windows.Forms
        $picker = New-Object System.Windows.Forms.FolderBrowserDialog
        $picker.Description = 'Wybierz folder audio. Zostana przeszukane wszystkie podfoldery.'
        $picker.ShowNewFolderButton = $false
        try {
            if ($picker.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { exit 0 }
            $InputFolder = $picker.SelectedPath
        } finally { $picker.Dispose() }
    }

    if ($InputFolder) {
        $root = Get-Item -LiteralPath $InputFolder
        if (-not $root.PSIsContainer) { throw 'Podana sciezka nie jest folderem.' }
        $rootPrefix = $root.FullName.TrimEnd('\') + '\'
        $outputRoot = Join-Path $root.FullName 'WAV'
        $extensions = @('.mp3','.m4a','.aac','.flac','.ogg','.opus','.wma','.aiff','.aif','.amr','.ape','.alac','.ac3','.m4b','.mka','.mp2','.m4r','.oga','.aifc','.caf','.au','.dts','.mpa','.wv','.w64','.dsf','.dff','.mp4','.webm','.3gp','.mov','.mkv')
        Write-Host ('Przeszukiwanie: ' + $root.FullName)
        $pending = New-Object 'System.Collections.Generic.Stack[string]'
        $found = New-Object 'System.Collections.Generic.List[string]'
        $pending.Push($root.FullName)
        while ($pending.Count -gt 0) {
            foreach ($item in (Get-ChildItem -LiteralPath $pending.Pop() -Force)) {
                # Nie podazaj za dowiazaniami / junctionami ani katalogami wynikow.
                if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { continue }
                if ($item.PSIsContainer) {
                    if ($item.Name -ine 'WAV') { $pending.Push($item.FullName) }
                } elseif ($item.Extension -in $extensions) {
                    $found.Add($item.FullName)
                }
            }
        }
        $InputFiles = @($found.ToArray() | Sort-Object)
        Write-Host ('Znaleziono plikow audio: ' + $InputFiles.Count)
        if ($InputFiles.Count -eq 0) { Write-Host 'Brak plikow do konwersji.'; exit 0 }
    }

    $converter = Get-Converter
    $ok = 0
    $failed = 0
    foreach ($inputFile in $InputFiles) {
        $tempOutput = $null
        try {
            $source = Get-Item -LiteralPath $inputFile
            if ($source.PSIsContainer) { throw 'Wybierz pliki, a nie folder.' }
            if ($InputFolder) {
                $relative = $source.FullName.Substring($rootPrefix.Length)
                $relativeDir = [IO.Path]::GetDirectoryName($relative)
                $targetDir = $outputRoot
                if ($relativeDir) { $targetDir = Join-Path $outputRoot $relativeDir }
            } else {
                $targetDir = Join-Path $source.DirectoryName 'WAV'
            }
            New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
            $base = $source.BaseName
            $target = Join-Path $targetDir ($base + '.wav')
            $suffix = 2
            while (Test-Path -LiteralPath $target) {
                $target = Join-Path $targetDir ($base + ' (' + $suffix + ').wav')
                $suffix++
            }
            $tempOutput = Join-Path $targetDir ([guid]::NewGuid().ToString('N') + '.partial.wav')
            Write-Host ('Konwersja: ' + $source.Name)
            & $converter -hide_banner -loglevel error -nostdin -n -i $source.FullName -map 0:a:0 -vn -c:a pcm_s16le -rf64 auto $tempOutput
            if ($LASTEXITCODE -ne 0) { throw ('FFmpeg zakonczyl sie kodem ' + $LASTEXITCODE) }
            Move-Item -LiteralPath $tempOutput -Destination $target
            $tempOutput = $null
            Write-Host ('Zapisano: ' + $target) -ForegroundColor Green
            $ok++
        } catch {
            Write-Host ('BLAD: ' + $inputFile + ' - ' + $_.Exception.Message) -ForegroundColor Red
            $failed++
        } finally {
            if ($tempOutput -and (Test-Path -LiteralPath $tempOutput)) { Remove-Item -LiteralPath $tempOutput }
        }
    }
    Write-Host ("`nGotowe. Poprawnie: {0}. Bledy: {1}." -f $ok, $failed)
    if ($failed -gt 0) { exit 1 }
} catch {
    Write-Host ('BLAD: ' + $_.Exception.Message) -ForegroundColor Red
    Write-Host 'Jesli pobieranie sie nie powiodlo, sprawdz internet i uruchom ponownie.'
    exit 1
}