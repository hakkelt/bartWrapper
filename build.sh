#!/bin/sh

#if [ ! -f "../bart/lib/libmain.a" ]; then
    cd ../bart
    make
    ../bart/ar_lock.sh rsU ../bart/lib/libmain.a ../bart/src/bart.o
    cd ../bartConnector
#fi

## Merge static BART libraries into a single archive
# See: https://stackoverflow.com/a/23621751
#if [ -f "libbart.a" ]; then
    echo "create libbart.a" > libbart.mri
    for f in `ls ../bart/lib/*.a`
    do
        echo "addlib $f" >> libbart.mri
    done
    echo "save" >> libbart.mri
    echo "end" >> libbart.mri
    ar -M < libbart.mri
    rm libbart.mri

    if [[ -z "${JAVA_HOME}" ]]; then
    echo "You must set the JAVA_HOME environmental variable"
    exit 1
    fi
#fi

#gcc -static -Wl,--whole-archive -lpthread -Wl,--no-whole-archive -Wl,--allow-multiple-definition -Og -ffast-math -Wmissing-prototypes -g -std=gnu11 -fopenmp -Wall -Wextra -I"$JAVA_HOME/include" -I"$JAVA_HOME/include/win32" -I"../bart/src"         -o bart BartConnector.c libbart.a -L/usr/lib -lfftw3f -L/mingw64/lib -lopenblas -lgfortran -lquadmath -lpng -lz -lm -lrt

gcc  -static -Wl,--whole-archive -lpthread -Wl,--no-whole-archive -Wl,--allow-multiple-definition -Og -ffast-math -Wmissing-prototypes -g -std=gnu11 -fopenmp -Wall -Wextra -I"$JAVA_HOME/include" -I"$JAVA_HOME/include/win32" -I"../bart/src" -shared -o bart.dll src/native/BartConnector.c libbart.a -L/usr/lib -lfftw3f -L/mingw64/lib -lopenblas -lgfortran -lquadmath -lpng -lz -lm -lrt

## Gether library archives to be statically linked to the dll
#LIBS=`ls ../bart/lib/*.a`

#gcc -static -Wl,--whole-archive -lpthread -Wl,--no-whole-archive -Wl,--allow-multiple-definition -O3 -ffast-math -Wmissing-prototypes -g -std=gnu11 -fopenmp -Wall -Wextra -DNOLAPACKE -iquote ../bart/src/ -I/usr/include/ -I/mingw64/include/OpenBLAS -I"$JAVA_HOME/include" -I"$JAVA_HOME/include/win32" -I"../bart/src" -L"../bart/lib" $LIBS -shared -o bart.dll BartConnector.c -L/usr/lib -lfftw3f -L/mingw64/lib -lopenblas -lgfortran -lquadmath -lpng -lz -lm -lrt

#gcc -static -Wl,--whole-archive -lpthread -Wl,--no-whole-archive -Wl,--allow-multiple-definition -O3 -ffast-math -Wmissing-prototypes -g -std=gnu11 -fopenmp -Wall -Wextra -DNOLAPACKE -g -MMD -MF ./.bart.d -iquote ../bart/src/ -I/usr/include/ -I/mingw64/include/OpenBLAS -I"$JAVA_HOME/include" -I"$JAVA_HOME/include/win32" -DMAIN_LIST="avg, bench, bin, bitmask, cabs, caldir, calmat, carg, casorati, cc, ccapply, cdf97, circshift, conj, conv, conway, copy, cpyphs, creal, crop, delta, ecalib, ecaltwo, epg, estdelay, estdims, estshift, estvar, extract, fakeksp, fft, fftmod, fftrot, fftshift, filter, flatten, flip, fmac, homodyne, index, invert, itsense, join, looklocker, lrmatrix, mandelbrot, mip, moba, mobafit, nlinv, noise, normalize, nrmse, nufft, ones, pattern, phantom, pics, pocsense, poisson, pol2mask, poly, repmat, reshape, resize, rmfreq, rof, roistat, rss, rtnlinv, sake, saxpy, scale, sdot, show, signal, slice, spow, sqpics, squeeze, ssa, std, svd, tgv, threshold, toimg, traj, transpose, twixread, upat, var, vec, version, walsh, wave, wavelet, wavepsf, whiten, window, wshfl, zeros, zexp, ()" -shared -o bart BartConnector.c $LIBS -L/usr/lib -lfftw3f  -L/mingw64/lib -lopenblas -lgfortran -lquadmath -lpng -lz -lm -lrt

#gcc -static -Wl,--whole-archive -lpthread -Wl,--no-whole-archive -Wl,--allow-multiple-definition -O3 -ffast-math -Wmissing-prototypes -g -std=gnu11 -fopenmp -Wall -Wextra -DNOLAPACKE -g -MMD -MF ./.bart.d -iquote /d/mr_package/Camera/MRISubsystem/Applications/RS2D/Spinlab-src/other_source/bart/src/ -I/usr/include/ -I/mingw64/include/OpenBLAS -DMAIN_LIST="avg, bench, bin, bitmask, cabs, caldir, calmat, carg, casorati, cc, ccapply, cdf97, circshift, conj, conv, conway, copy, cpyphs, creal, crop, delta, ecalib, ecaltwo, epg, estdelay, estdims, estshift, estvar, extract, fakeksp, fft, fftmod, fftrot, fftshift, filter, flatten, flip, fmac, homodyne, index, invert, itsense, join, looklocker, lrmatrix, mandelbrot, mip, moba, mobafit, nlinv, noise, normalize, nrmse, nufft, ones, pattern, phantom, pics, pocsense, poisson, pol2mask, poly, repmat, reshape, resize, rmfreq, rof, roistat, rss, rtnlinv, sake, saxpy, scale, sdot, show, signal, slice, spow, sqpics, squeeze, ssa, std, svd, tgv, threshold, toimg, traj, transpose, twixread, upat, var, vec, version, walsh, wave, wavelet, wavepsf, whiten, window, wshfl, zeros, zexp, ()" -include src/main.h -Dmain_real=main_bart -o bart src/main.c /d/mr_package/Camera/MRISubsystem/Applications/RS2D/Spinlab-src/other_source/bart/src/bart.o lib/libbox.a lib/libgrecon.a lib/libsense.a lib/libnoir.a lib/libiter.a lib/liblinops.a lib/libwavelet.a lib/liblowrank.a lib/libnoncart.a lib/libcalib.a lib/libsimu.a lib/libsake.a lib/libdfwavelet.a lib/libnlops.a lib/libmoba.a lib/libgeom.a lib/libnum.a lib/libmisc.a lib/libnum.a lib/libmisc.a lib/libwin.a lib/liblapacke.a -L/usr/lib -lfftw3f  -L/mingw64/lib -lopenblas -lgfortran -lquadmath -lpng -lz   -lm -lrt
