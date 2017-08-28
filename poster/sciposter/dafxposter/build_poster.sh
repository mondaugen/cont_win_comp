#!/bin/bash
export TEXINPUTS=.:../../../paper/:
pdflatex dafx-poster.tex
BSTINPUTS=.:../../../paper/: BIBINPUTS=.:../../../paper/: bibtex dafx-poster.aux
pdflatex dafx-poster.tex
pdflatex dafx-poster.tex
