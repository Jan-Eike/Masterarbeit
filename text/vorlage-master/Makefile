.PHONY: all
all: article

.PHONY:article
article:
	latexmk -pdf ${PVC} -pdflatex="pdflatex -synctex=1 -interaction=nonstopmode" thesis.tex

.PHONY: watch
watch: PVC=-pvc
watch: article

.PHONY: clean
clean:
	latexmk -C -bibtex
	rm *.loa *.lol *.tdo

.PHONY: zip
zip:
	zip thesis -x *.git/* -x *.synctex.gz -x .gitlab-ci.yml -r .

.PHONY: bibtool
bibtool:
	bibtool -R -r keep_bibtex -r field -r improve -r month -s references.bib -o references.bib
