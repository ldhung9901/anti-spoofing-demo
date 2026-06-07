$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\..\src\AttendanceLivenessDemo.Api"
$env:ASPNETCORE_URLS = "http://localhost:5088"
dotnet run
