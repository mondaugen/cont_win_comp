#! /bin/bash

# Builds the thesis.tex file complete with bibliography.

# Convert all eps files to pdf

pdflatex paper.tex

BIBINPUTS=. bibtex paper.aux
pdflatex paper.tex
pdflatex paper.tex
mv paper.pdf Esterer_Nicholas_dafx_2017.pdf
#dvipdf thesis.dvi
