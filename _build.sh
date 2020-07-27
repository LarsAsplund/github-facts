#!/usr/bin/env sh

cd $(dirname "$0")

set -ev

Rscript -e "bookdown::render_book('01-index.Rmd', 'bookdown::gitbook')"
Rscript -e "bookdown::render_book('01-index.Rmd', 'bookdown::pdf_book')"
Rscript -e "bookdown::render_book('01-index.Rmd', 'bookdown::epub_book')"
