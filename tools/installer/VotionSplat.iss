; VotionSplat 3.0 installer. AppId stays the same for upgrades.
#define AppVersion "3.0.0"

[Setup]
AppId={{8F3C2A1E-7B64-4D19-9C5A-2E6F0B8D4A17}}
AppName=VotionSplat
AppVersion={#AppVersion}
AppPublisher=Yik
DefaultDirName=C:\VotionSplat
DefaultGroupName=VotionSplat
DisableDirPage=no
DisableProgramGroupPage=yes
OutputDir=D:\VotionSplat_build\Output
OutputBaseFilename=VotionSplat-{#AppVersion}-Setup
SetupIconFile=..\..\app\assets\icon.ico
WizardImageFile=..\..\app\assets\installer_wizard.png
WizardImageStretch=yes
WizardImageBackColor=$000000
Compression=lzma2/fast
SolidCompression=no
DiskSpanning=yes
DiskSliceSize=max
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
UninstallDisplayIcon={app}\app\assets\icon.ico

[Dirs]
Name: "{app}"; Permissions: users-modify

[Files]
Source: "D:\VotionSplat_build\stage\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "D:\VotionSplat_build\redist\vc_redist.x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Icons]
Name: "{group}\VotionSplat"; Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: "-m app.main"; WorkingDir: "{app}"; IconFilename: "{app}\app\assets\icon.ico"; AppUserModelID: "Yik.VotionSplat.3"
Name: "{autodesktop}\VotionSplat"; Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: "-m app.main"; WorkingDir: "{app}"; IconFilename: "{app}\app\assets\icon.ico"; AppUserModelID: "Yik.VotionSplat.3"
Name: "{group}\VotionSplat install check"; Filename: "{app}\install_check.log"

[Registry]
Root: HKLM; Subkey: "SYSTEM\CurrentControlSet\Control\FileSystem"; ValueType: dword; ValueName: "LongPathsEnabled"; ValueData: "1"

[Run]
Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/install /quiet /norestart"; StatusMsg: "Installing the Visual C++ runtime..."; Flags: waituntilterminated
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\tools\installer\relocate_venvs.ps1"" -AppDir ""{app}"""; StatusMsg: "Pointing Python at this folder..."; Flags: runhidden waituntilterminated
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\tools\installer\verify_install.ps1"" -AppDir ""{app}"""; StatusMsg: "Checking the install..."; Flags: runhidden waituntilterminated
Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: "-m app.main"; WorkingDir: "{app}"; Description: "Launch VotionSplat"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: files; Name: "{app}\install_check.log"

[Code]
function InitializeSetup(): Boolean;
var
  Code: Integer;
begin
  if not Exec('nvidia-smi.exe', '--query-gpu=name --format=csv,noheader', '', SW_HIDE, ewWaitUntilTerminated, Code) then
    MsgBox('nvidia-smi was not found. Install an NVIDIA driver 570 or newer before training.', mbInformation, MB_OK);
  Result := True;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  FreeBytes, TotalBytes: Int64;
begin
  Result := True;
  if CurPageID = wpSelectDir then
  begin
    if GetSpaceOnDisk64(ExtractFileDrive(WizardDirValue), FreeBytes, TotalBytes) then
      if FreeBytes < Int64(150) * 1024 * 1024 * 1024 then
        MsgBox('This drive has less than 150 GB free. Models and scene outputs need that space.', mbInformation, MB_OK);
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  LogPath, Text: AnsiString;
begin
  if CurStep = ssPostInstall then
  begin
    LogPath := ExpandConstant('{app}\install_check.log');
    if LoadStringFromFile(LogPath, Text) and (Pos('FAIL', String(Text)) > 0) then
      MsgBox('The install check failed. See ' + LogPath, mbError, MB_OK);
  end;
end;
