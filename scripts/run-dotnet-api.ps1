$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\..\src\AttendanceLivenessDemo.Api"
$env:ASPNETCORE_URLS = "http://0.0.0.0:5088"
dotnet run
