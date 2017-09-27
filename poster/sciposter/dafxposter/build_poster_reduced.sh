#!/bin/bash
export TEXINPUTS=.:../../../paper/:
pdflatex dafx-poster-reduced.tex
BSTINPUTS=.:../../../paper/: BIBINPUTS=.:../../../paper/: bibtex ../../../paper/paper.aux
#dafx-poster-reduced.aux
pdflatex dafx-poster-reduced.tex
pdflatex dafx-poster-reduced.tex
mv dafx-poster-reduced.pdf Esterer_Nicholas_DAFx_2017_Poster_A0.pdf
