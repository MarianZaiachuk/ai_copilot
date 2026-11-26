#define MyAppName "AI Copilot"
#define MyAppVersion "1.3.2"
#define MyAppPublisher "MarianZaiachuk"
#define MyAppURL "https://github.com/MarianZaiachuk/ai_copilot"
#define MyAppExeName "ZoomAICopilot.exe"
#ifndef BuildVariant
  #define BuildVariant ""
#endif
#ifndef SourceDir
  #define SourceDir "..\dist\ZoomAICopilot"
#endif

[Setup]
AppId={{D9A2F8B4-71E3-4E2A-B8C5-6E4F3D2A1091}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={autopf}\ZoomAICopilot
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
OutputDir=..\release
#if BuildVariant == "Lite"
OutputBaseFilename=ZoomAICopilot-Lite-Setup-v{#MyAppVersion}
#else
OutputBaseFilename=ZoomAICopilot-Setup-v{#MyAppVersion}
#endif
Compression=lzma2/fast
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=4
SolidCompression=no
WizardStyle=modern
CloseApplications=yes

[Languages]
Name: "ukrainian"; MessagesFile: "compiler:Languages\Ukrainian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "vcredist"; Description: "Встановити Microsoft Visual C++ 2015–2022 Redistributable (x64)"; GroupDescription: "Необхідні компоненти системи:"; Check: NeedsVCRedist

[Files]
; Explicit clean allowlist: never include local .env, logs, context_sources.json, or prompt_profiles.json
Source: "{#SourceDir}\ZoomAICopilot.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\.env.example"; DestDir: "{app}"; Flags: ignoreversion
#if FileExists("redist\vc_redist.x64.exe")
Source: "redist\vc_redist.x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall; Tasks: vcredist; Check: NeedsVCRedist
#endif

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autoprograms}\{#MyAppName} — Діагностика"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--diagnose"
Name: "{autoprograms}\{#MyAppName} — Режим сумісності"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--software"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
#if FileExists("redist\vc_redist.x64.exe")
Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/install /quiet /norestart"; StatusMsg: "Встановлення Visual C++ Redistributable (x64)..."; Flags: waituntilterminated; Tasks: vcredist; Check: NeedsVCRedist
#else
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -Command ""try {{ [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $p = Join-Path $env:TEMP 'vc_redist.x64.exe'; Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile $p -UseBasicParsing -TimeoutSec 30; Start-Process -FilePath $p -ArgumentList '/install','/quiet','/norestart' -Wait }} catch {{}}"""; StatusMsg: "Завантаження та встановлення Visual C++ Redistributable (x64)..."; Flags: runhidden waituntilterminated; Tasks: vcredist; Check: NeedsVCRedist
#endif
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
function NeedsVCRedist(): Boolean;
var
  Installed: Cardinal;
  Major: Cardinal;
begin
  Result := True;
  if RegQueryDWordValue(HKLM64, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Installed', Installed) then
  begin
    if (Installed = 1) and RegQueryDWordValue(HKLM64, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Major', Major) then
    begin
      if Major >= 14 then
        Result := False;
    end;
  end;
end;
