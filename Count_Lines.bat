Powershell Script to count lines:
Go to the top-level directory and run:

Get-Content *.py | Measure-Object -Line
Get-Content emba*/*.py | Measure-Object -Line
Get-Content test*/*.py | Measure-Object -Line