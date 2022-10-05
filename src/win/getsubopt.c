/* Copyright 2021. Tamás Hakkel
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2022 Tamás Hakkel <hakkelt@gmail.com>
 */

#include <string.h>
#include "win/getsubopt.h"

int getsubopt(char **restrict optionp, char *const *restrict tokens, char **restrict valuep)
{
    if (**optionp == '\0')
        return -1;

    char *commaPos;
	for (commaPos = *optionp; *commaPos != ',' && *commaPos != '\0'; commaPos++);

    char *keyEndPos;
	for (keyEndPos = *optionp; *keyEndPos != '=' && keyEndPos < commaPos; keyEndPos++);
    size_t tokenLength = keyEndPos - *optionp;

    for (int i = 0; tokens[i] != NULL; i++)
        if (strlen(tokens[i]) == tokenLength && strncmp(*optionp, tokens[i], tokenLength) == 0) {
            *valuep = keyEndPos == commaPos ? NULL : keyEndPos + 1;
            if (*commaPos != '\0')
                *commaPos++ = '\0';
            *optionp = commaPos;
            return i;
        }

    *valuep = *optionp;
    if (*commaPos != '\0')
        *commaPos++ = '\0';
    *optionp = commaPos;
    return -1;
}
