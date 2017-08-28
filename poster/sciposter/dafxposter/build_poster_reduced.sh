#!/bin/bash
export TEXINPUTS=.:../../../paper/:
pdflatex dafx-poster-reduced.tex
BSTINPUTS=.:../../../paper/: BIBINPUTS=.:../../../paper/: bibtex dafx-poster-reduced.aux
pdflatex dafx-poster-reduced.tex
pdflatex dafx-poster-reduced.tex
