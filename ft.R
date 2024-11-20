library(tidyverse)

df <- read_csv("derived_data/ft_combined.csv");


                                        # 1. Table with STUDYID, USUBJID Count, and Row Count
table_df <- df %>%
    group_by(STUDYID) %>%
    summarise(USUBJID_Count = n_distinct(USUBJID), Row_Count = n())
print(table_df)


                                        # 2. Bar graph of total number of FTTEST of each type
ggplot(df, aes(x = factor(FTTEST, levels = names(sort(table(df$FTTEST), decreasing=F))))) +
    geom_bar() +
    coord_flip() +
    labs(x = 'FTTEST', y = 'Count')
ggsave("figures/ft_counts.png", height=8.5, width=8.5);


                                        # 3. Pivot table and geom_tile plot

df_wide <- df %>% select(STUDYID, USUBJID, FTTEST) %>%
    distinct() %>%
    mutate(value = 1) %>%
    spread(key = FTTEST, value = value, fill = 0);

df_long_again <- df_wide %>% pivot_longer(cols=`Babinski Sign`:`What side is back pain more dominant`) %>%
    rename(FTTEST=name, done=value);

USUBJID_counts <- df_long_again %>% group_by(USUBJID) %>% summarise(cc = sum(done)) %>% arrange(cc);



ggplot(df_long_again, aes(x = factor(FTTEST, levels = names(sort(table(df$FTTEST), decreasing=T))),
                          y = factor(USUBJID, levels = USUBJID_counts$USUBJID), fill = factor(done))) +
    geom_tile() +
    labs(x = 'FTTEST', y = 'USUBJID', fill = 'Value') +
    scale_fill_manual(values = c('0' = 'white', '1' = 'blue')) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1),
          axis.text.y = element_blank());
ggsave("figures/ft_vectors.png",height=11, width=8.5)

