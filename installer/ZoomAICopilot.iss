#define MyAppName "AI Copilot"
#define MyAppVersion "1.2"
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
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
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

[Files]
; Explicit clean allowlist: never include local .env, logs, context_sources.json, or prompt_profiles.json
Source: "{#SourceDir}\ZoomAICopilot.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\.env.example"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
