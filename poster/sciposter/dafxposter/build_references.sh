#!/bin/bash
export TEXINPUTS=.:../../../paper/:
pdflatex references.tex
BSTINPUTS=.:../../../paper/: BIBINPUTS=.:../../../paper/: bibtex ../../../paper/paper.aux
#references.aux
pdflatex references.tex
pdflatex references.tex
