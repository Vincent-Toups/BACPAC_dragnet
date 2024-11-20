library(tidyverse);

meta_data <- read_csv("derived_data/meta-data.csv") %>%
    filter(domain=="DM") %>%
    filter(archive==F) %>%
    filter(theoretical_model == F) %>%
    select(file, institution) %>%
    rowwise() %>%
    mutate(study={
        read_csv(file) %>% pull(STUDYID) %>% `[[`(1)
    }, file_hash = {
        system(sprintf("md5sum \"%s\"", file), intern=T)
    }) %>%
    ungroup() %>% filter(study=="PHENO")

 
 
