REM cl.exe /fp:fast /Ox /openmp /I. run.c win.c 

REM NOTE RUN IN x64 native tools command prompt for VS 2022
REM NOTE: USE CMAKE INSTEAD...

cl.exe /fp:fast /Ox /openmp /I. run.c win.c /link C:\Users\jonas\Downloads\pcre-8.45\pcre-8.45\build\Release\pcre.lib
copy C:\Users\jonas\Downloads\pcre-8.45\pcre-8.45\build\Release\pcre.dll .
