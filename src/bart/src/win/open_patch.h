/* Copyright 2021. Tamás Hakkel
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2021 Tamás Hakkel <hakkelt@gmail.com>
 */

#ifndef OPEN_PATCH_H
#define OPEN_PATCH_H

#define open(pathname, flags, ...) open_patched(pathname, flags __VA_OPT__(,) __VA_ARGS__)

int open_patched(const char*, int, ...);

#endif /* OPEN_PATCH_H */
