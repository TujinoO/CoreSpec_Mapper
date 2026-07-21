#define MyAppName "CoreSpec Mapper V5.3"
#define MyAppVersion "5.3.0"
#define MyAppPublisher "CoreSpec Mapper Project"
#define MyAppExeName "CoreSpecMapperV5_3.exe"
#ifndef AppSource
  #define AppSource "..\release\staging\CoreSpecMapperV5_3"
#endif

[Setup]
AppId={{B476F4CD-8AE8-4938-AB24-A53B5DC3D5D3}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} Stable
AppPublisher={#MyAppPublisher}
VersionInfoVersion=5.3.0.0
VersionInfoDescription=CoreSpec Mapper V5.3 Stable Setup
VersionInfoCompany={#MyAppPublisher}
VersionInfoProductName=CoreSpec Mapper V5.3 Stable
DefaultDirName={localappdata}\Programs\CoreSpec Mapper V5.3
DefaultGroupName=CoreSpec Mapper V5.3
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\release
OutputBaseFilename=CoreSpec_Mapper_V5.3.0_Stable_Setup
SetupIconFile=..\src\corespec_mapper\resources\corespec_logo.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no
UsePreviousAppDir=yes
UsePreviousTasks=yes

[Languages]
Name: "chinesesimp"; MessagesFile: "ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "快捷方式："; Flags: checkedonce

[Files]
Source: "{#AppSource}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\docs\CoreSpec_Mapper_V5_3_稳定版技术方法与系统设计_2026-07-21.docx"; DestDir: "{app}\Documentation"; Flags: ignoreversion
Source: "..\docs\CoreSpec_Mapper_V5_3_稳定版用户使用指南_2026-07-21.docx"; DestDir: "{app}\Documentation"; Flags: ignoreversion
Source: "..\release_notes_V5.3.0_Stable.md"; DestDir: "{app}\Documentation"; Flags: ignoreversion

[Icons]
Name: "{group}\CoreSpec Mapper V5.3"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{group}\V5.3 用户使用指南"; Filename: "{app}\Documentation\CoreSpec_Mapper_V5_3_稳定版用户使用指南_2026-07-21.docx"
Name: "{group}\卸载 CoreSpec Mapper V5.3"; Filename: "{uninstallexe}"
Name: "{autodesktop}\CoreSpec Mapper V5.3"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "启动 CoreSpec Mapper V5.3"; Flags: nowait postinstall skipifsilent
