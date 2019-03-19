#!/usr/bin/env python3
# author: Bernhard Hörl

import re


def RA2sec(RA):
    hours, minutes, seconds = map(float, re.split(r":", RA))
    return 3600 * hours + 60 * minutes + seconds


def sec2RA(seconds):
    hours = int(seconds / 3600)
    seconds -= hours * 3600
    minutes = int(seconds / 60)
    seconds -= minutes * 60
    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"


def DEC2arcsec(DEC):
    degrees, arcminutes, arcseconds = map(float, re.split(r":", DEC))
    return 3600 * degrees + 60 * arcminutes + arcseconds


def arcsec2DEC(arcseconds):
    degrees = int(arcseconds / 3600)
    arcseconds -= degrees * 3600
    arcminutes = int(arcseconds / 60)
    arcseconds -= arcminutes * 60
    return f"{degrees:+03d}:{arcminutes:02d}:{arcseconds:04.1f}"


def open_map(fname, delim=","):
    with open(fname) as f:
        lines = f.readlines()
        labels = [*lines[0].split(delim)]
        labels[-1] = labels[-1][:-1]
        data = {}
        for label in labels:
            data[label] = []
        for line in lines[1:]:
            for i, col in enumerate(line.split(delim)):
                key = labels[i]
                if i:
                    val = float(col)
                else:
                    val = int(col)
                data[key].append(val)
    return data
