library(tidyverse);

ex <- read_csv("derived_data/ex_combined.csv");

                                        # Load the ggplot2 package
library(ggplot2)

                                        # Create the bar graph

ex %>% group_by(EXTRT) %>% tally() %>%
    ggplot(aes(x = reorder(EXTRT, n), y = n)) +
    geom_bar(stat = "identity") +
    coord_flip() +
    labs(title = "Treatment Count", x = "Treatment", y = "Count") +
    theme_minimal()

ggsave("figures/ex_treatments.png");

                                        # Assuming your data frame is named `df`
wex <- ex %>%
    select(STUDYID, USUBJID, VISITNUM, EXTRT) %>%
    distinct() %>%
    mutate(value = 1) %>%
    unite('VISITNUM_EXTRT', VISITNUM, EXTRT, sep = '_') %>%
    pivot_wider(
        names_from = VISITNUM_EXTRT,
        values_from = value,
        values_fill = 0);


                                        # Replace spaces in column names with underscores
names(wex) <- str_replace_all(names(wex), ' ', '_')

                                        # Extract and sort columns based on the <N> values
sorted_columns <- names(wex)[order(as.numeric(str_extract(names(wex), '^[0-9]+')))]

                                        # Rearrange columns: STUDYID and USUBJID first, followed by sorted columns
wex <- wex %>%
    select(STUDYID, USUBJID, all_of(sorted_columns))




                                        # Convert to long format
long_df <- wex %>%
    select(-STUDYID) %>%
    pivot_longer(
        cols = `4_Exercise`:`26_Non-spinal_fusion`,
        names_to = 'VISITNUM_TREATMENT',
        values_to = 'value'
    ) %>%a
    mutate(
        VISITNUM = as.numeric(str_extract(VISITNUM_TREATMENT, '^[0-9]+')),
        TREATMENT = str_extract(VISITNUM_TREATMENT, '[^0-9_]+$')
    )

                                        # Create the imagesc style plot
ggplot(long_df, aes(x = VISITNUM_TREATMENT, y = USUBJID, fill = value)) +
    geom_tile() +
    scale_fill_gradient(low = 'white', high = 'blue') +
    theme_minimal() +
    labs(x = 'VISITNUM_TREATMENT', y = 'USUBJID', fill = 'Value') +
    theme(axis.text.x = element_text(angle = 90, hjust = 1));
ggsave("figures/treatment_imagesc.png", height=11, width=8.5);

selected_data <- wex %>%
    select(`4_Exercise`:`26_Non-spinal_fusion`)

                                        # Performing PCA
pca_result <- prcomp(selected_data, center = TRUE, scale. = TRUE)

                                        # Extracting the rotated vectors
wex_r <- as.data.frame(pca_result$x) %>% as_tibble() %>% mutate(USUBJID=wex$USUBJID, STUDYID=wex$STUDYID);

long_r <- wex_r %>%
    select(-STUDYID) %>%
    pivot_longer(
        cols = `PC1`:`PC90`,
        names_to = 'PC',
        )


ggplot(long_r, aes(x = PC, y = USUBJID, fill = value)) +
    geom_tile() +
    scale_fill_gradient(low = 'white', high = 'blue') +
    theme_minimal() +
    labs(x = 'PC', y = 'USUBJID', fill = 'Value');
ggsave("figures/treatment_pca_imagesc.png");

max_k <- 10
wss <- numeric(max_k)

for (k in 1:max_k) {
    set.seed(123)  # For reproducibility
    kmeans_result <- kmeans(pca_result$x, centers = k)
    wss[k] <- kmeans_result$tot.withinss
}

                                        # Creating a data frame for ggplot
df <- data.frame(K = 1:max_k, WSS = wss)

                                        # Using ggplot2 for plotting
ggplot(df, aes(x = K, y = WSS)) +
    geom_line(aes(group = 1)) +
    geom_point(size = 3) +
    labs(x = 'Number of clusters K',
         y = 'Total within-clusters sum of squares') +
      theme_minimal()

kmeans_result <- kmeans(pca_result$x, centers = 7)

wex_cc <- wex %>% mutate(cluster=kmeans_result$cluster) %>% arrange(cluster) %>%
    mutate(cluster_index = row_number());

long_cc <- wex_cc %>%
    select(-STUDYID) %>%
    pivot_longer(
        cols = `4_Exercise`:`26_Non-spinal_fusion`,
        names_to = 'VISITNUM_TREATMENT',
        values_to = 'value'
    )

                                        # Create the imagesc style plot
ggplot(long_cc, aes(x = VISITNUM_TREATMENT, y = cluster_index, fill = factor(value*cluster))) +
    geom_tile() +
    theme_minimal() +
    labs(x = 'VISITNUM_TREATMENT', y = 'Cluster', fill = 'Value') +
    theme(axis.text.x = element_text(angle = 90, hjust = 1));
ggsave("figures/treatment_clustered_imagesc.png", height=11, width=8.5);

ggplot(long_cc, aes(x = VISITNUM_TREATMENT, y = cluster_index, fill = factor(value*cluster))) +
    geom_tile() +
    theme_minimal() +
    labs(x = 'VISITNUM_TREATMENT', y = 'Subject', fill = 'Cluster') +
    theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
    scale_fill_manual(values = c(
                          '0' = 'white',
                          '1' = '#F8766D',
                          '2' = '#00BFC4',
                          '3' = '#A3A500',
                          '4' = '#E76BF3',
                          '5' = '#FF61CC',
                          '6' = '#00BF7D',
                          '7' = '#9590FF'
    ))
ggsave('figures/treatment_clustered_imagesc.png', height=11, width=8.5)
