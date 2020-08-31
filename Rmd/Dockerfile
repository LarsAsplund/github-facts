FROM btdi/bookdown

WORKDIR /src

RUN apt update -qq && apt install -y python3 librsvg2-bin git \
  && Rscript \
    -e "install.packages('reticulate')" \
    -e "reticulate::install_miniconda()"

CMD ["./_build.sh"]
