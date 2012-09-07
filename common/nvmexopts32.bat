@echo off
rem ********************************************************************
rem MATLAB parameters
rem ********************************************************************
set MATLAB=c:\Program Files\MATLAB\R2010a
set MW_TARGET_ARCH=win32

rem ********************************************************************
rem Main Visual Studio Configuration
rem ********************************************************************
set VS=C:\MSVS2008
set VC=%VS%\VC
set PATH=%VC%\BIN\;%VS%\Common7\IDE;%VS%\SDK\v2.0\bin;%VS%\Common7\Tools;%VS%\Common7\Tools\bin;%VC%\VCPackages;%MATLAB_BIN%;%PATH%
set INCLUDE=%VC%\ATLMFC\INCLUDE;%VC%\INCLUDE;%INCLUDE%
set LIB=%VC%\ATLMFC\LIB;%VC%\LIB;%VS%\LIB;%MATLAB%\extern\lib\win32\microsoft;%LIB%

rem ********************************************************************
rem Microsoft SDK Specific
rem ********************************************************************
set SDK=c:/Program Files/Microsoft SDKs/Windows/v6.0A
set PATH=%SDK%\Bin;%PATH%
set INCLUDE=%SDK%\Include;%INCLUDE%
set LIB=%SDK%\Lib;%LIB%

rem ********************************************************************
rem Compiler parameters
rem ********************************************************************
rem set COMPILER=cl
set COMPILER=nvcc
set COMPFLAGS= -DMX_COMPAT_32 -c -Xcompiler "/c /Zp8 /GR /W3 /EHs /D_CRT_SECURE_NO_DEPRECATE /D_SCL_SECURE_NO_DEPRECATE /D_SECURE_SCL=0 /DMATLAB_MEX_FILE /nologo /MT"
set OPTIMFLAGS=-Xcompiler "/O2 /Oy- /DNDEBUG"
set DEBUGFLAGS=-Xcompiler "/Zi /Fd"%OUTDIR%%MEX_NAME%%MEX_EXT%.pdb""
set NAME_OBJECT=

rem ********************************************************************
rem Linker parameters
rem ********************************************************************
set LIBLOC=%MATLAB%\extern\lib\win32\microsoft
set LINKER=link
set MEX=libmx.lib libmex.lib libmat.lib
set STD=kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib uuid.lib odbc32.lib odbccp32.lib
set LINKFLAGS=/dll /export:%ENTRYPOINT% /MAP /LIBPATH:"%LIBLOC%" /NODEFAULTLIB:msvcrt.lib /NODEFAULTLIB:libcmtd.lib /NODEFAULTLIB:msvcrtd.lib %MEX% oleaut32.lib libcmt.lib /implib:%LIB_NAME%.x /MACHINE:x86 %STD% /NODEFAULTLIB:libc.lib /MANIFEST
set LINKOPTIMFLAGS=
set LINKDEBUGFLAGS=/DEBUG /PDB:"%OUTDIR%%MEX_NAME%%MEX_EXT%.pdb"
set LINK_FILE=
set LINK_LIB=
set NAME_OUTPUT=/out:"%OUTDIR%%MEX_NAME%%MEX_EXT%"
set RSP_FILE_INDICATOR=@

rem ********************************************************************
rem Resource compiler parameters
rem ********************************************************************
set RC_COMPILER=rc /fo "%OUTDIR%mexversion.res"
set RC_LINKER=

set POSTLINK_CMDS=del "%OUTDIR%%MEX_NAME%.map"
set POSTLINK_CMDS1=del %LIB_NAME%.x
set POSTLINK_CMDS2=mt -outputresource:"%OUTDIR%%MEX_NAME%%MEX_EXT%";2 -manifest "%OUTDIR%%MEX_NAME%%MEX_EXT%.manifest"
set POSTLINK_CMDS3=del "%OUTDIR%%MEX_NAME%%MEX_EXT%.manifest" 
