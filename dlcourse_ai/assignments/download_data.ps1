Import-Module BitsTransfer

$files = "train_32x32.mat", "test_32x32.mat"
$url = "http://ufldl.stanford.edu/housenumbers"
$outputFolder = "$PSScriptRoot\data"

function Download-File {
    [cmdletbinding()]
    Param (
        [parameter(ValueFromPipeline)][string]$file
    )
    Process {
        $source = "$($url)/$($file)"
        $destination = "$($outputFolder)\$($file)"

        Write-Output "Downloading $($file)"

        Start-BitsTransfer -TransferType Download -Source $source -Destination $destination
    }
}

$start_time = Get-Date

if(!(Test-Path -Path $outputFolder)){
    New-Item -Path $outputFolder -ItemType Directory
}

$files | Download-File

Write-Output "Time taken: $((Get-Date).Subtract($start_time).Seconds) second(s)"