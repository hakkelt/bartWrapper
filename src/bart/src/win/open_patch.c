/* Copyright 2021. Tamás Hakkel
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2022 Tamás Hakkel <hakkelt@gmail.com>
 */

#include <string.h>
#include <fcntl.h>
#include <fileapi.h>
#include <share.h>
#include <stdarg.h>
#include "win/open_patch.h"

const UINT DRIVE_UNKNOWN = 0;

/** Sometimes, _sopen fails to open files when driver letter is expressed in a UNIX format
 * (e.g. /c/ instead of c:\). If the first three characters of the path are something like /c/,
 * then the letter between the slashed is checked whether it is an existing drive letter.
 * If it is, then we need to transform the path before opening.
 */
int open_patched(const char *__filename, int __flags, ...)
{
    va_list argp;
    va_start(argp, __flags);
    if (__filename[0] == '/' && __filename[1] != '\0' && (__filename[2] == '/' || __filename[2] == '\\')) {
        char *filename = strdup(__filename);
        filename[0] = __filename[1];
        filename[1] = ':';
        filename[2] = '\\';
        filename[3] = '\0';
        if (DRIVE_UNKNOWN != GetDriveTypeA(filename)) {
            filename[2] = '/';
            filename[3] = __filename[3];
            int fid = _sopen(filename, __flags | _O_BINARY, _SH_DENYNO, argp);
            free(filename);
            va_end(argp);
            return fid;
        }
        free(filename);
    }
    int fid = _sopen(__filename, __flags | _O_BINARY, _SH_DENYNO, argp);
    va_end(argp);
    return fid;
}
