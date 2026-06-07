#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../src/AttendanceLivenessDemo.Api"
export ASPNETCORE_URLS="http://localhost:5088"
dotnet run
