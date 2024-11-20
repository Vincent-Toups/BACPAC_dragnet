library(tidyverse);

tidymap <- function(lst, f){
    Map(f,lst);
}

meta_data <- read_csv("./meta-data.csv") %>%
    mutate(leading_dir={        
        str_split(file,"/") %>%
            tidymap(function(e) e[3]) %>%
            unlist();
    }) %>%
    filter(!is.na(domain));


