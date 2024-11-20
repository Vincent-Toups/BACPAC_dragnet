# Use the Rocker/verse image as the base
FROM docker.io/rocker/verse:latest

# Install dependencies for adding PPAs
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN apt update && apt install git 

RUN apt-get update && \
    apt-get install -y \
    software-properties-common \
    git \
    sqlite3 \
    libx11-6 \
    emacs \
    && rm -rf /var/lib/apt/lists/*

        
# Set environment variable for DISPLAY
ENV DISPLAY=:0

# Verify the installations
RUN emacs --version
RUN git --version
RUN sqlite3 --version

RUN apt-get update && apt-get install -y python3 python3-pip


RUN R -e "install.packages(c('gbm','pROC','pracma','haven'))"

RUN pip3 install --break-system-packages scikit-learn bokeh plotnine jupyterlab jupyter_bokeh ipywidgets jupyterlab_code_formatter jupyterlab-git nltk llama_cpp_python tqdm openai gensim tensorflow polars

# If you want to set up a dev environment, for instance
# USER rstudio
# COPY install.el /home/rstudio/install.el
# RUN emacs --batch -l /home/rstudio/install.el
# COPY init.el /home/rstudio/.emacs.d/init.el
# USER root

# Set the default command to start Emacs
CMD ["emacs"]
