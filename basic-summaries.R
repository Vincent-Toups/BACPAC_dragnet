library(tidyverse);
meta_data <- read_csv("derived_data/meta-data.csv") %>% filter(duplicate==F)

data_submitted <- meta_data %>%
    filter(!is.na(domain) & !archive) %>%
    group_by(domain, institution, file) %>%    
    tally() %>%
    rowwise() %>%
    mutate(study={
        d = read_csv(file,n_max=1) %>% pull(STUDYID);
    }) %>%
    ungroup() %>%
    mutate(study=sprintf("%s (%s)", study, institution)) %>%
    filter(!is.na(domain)) %>%
    select(-file,-institution) %>%
    select(-n) %>%
    mutate(dummy=T) %>%
    pivot_wider(names_from = domain, values_from = dummy, values_fn = function(x) {T}) %>%
    arrange(apply(is.na((.)), 1, sum)) %>%
    select(study, DM, SC, QS, EX, FT);
print(data_submitted %>% mutate(across(DM:SC,~ifelse(is.na(.),F,T)))

 
library(tidyverse);
dm_data <- read_csv("derived_data/meta-data.csv") %>% filter(duplicate==F) %>% filter(domain=="DM" & leading_dir != "ARCHIVE")

dm_data %>% group_by(USUBJID) %>% tally()

f <- function(ac, file){
    unique(c(ac, names(read_csv(file))))
}

conserved_columns <- Reduce(f, dm_data$file, c());
conserved_columns <- conserved_columns[!(conserved_columns %in% "AGE_UNIT")]

data <- do.call(rbind, Map(function(filename){
    has_cc <- conserved_columns[conserved_columns %in% names(read_csv(filename))];
    read_csv(filename) %>% select(all_of(has_cc));
}, dm_data$file))

ggplot(data, aes(AGE)) + geom_density();


library(ggplot2)
library(dplyr)


# Create directories for plots and tables if they don't exist
if (!dir.exists("plots")) dir.create("plots")
if (!dir.exists("tables")) dir.create("tables")

# Function to check if a column is categorical
is_categorical <- function(column) {
  is.logical(column) || is.factor(column) || is.character(column)
}

raw_data <- data;
                                        # Summary and plots for each column
for (col_name in names(data)) {

    if (col_name == "RACEMULT"){
        data <- raw_data %>% filter(!(RACEMULT %in% "N/A")) %>%
            filter(!is.na(RACEMULT)) %>% 
            mutate(RACEMULT=sprintf("%s ...",str_sub(RACEMULT,1,15)));
    } else {
        data <- raw_data;

    }
                                        # Skip identifier columns
    if (col_name %in% c("USUBJID")) next
    
                                        # Summary statistics for numerical columns
    if (is.numeric(data[[col_name]])) {
        cat(sprintf("Num column %s\n", col_name))
        summary_stats <- summary(data[[col_name]])
        hist_plot <- ggplot(data, aes_string(x = col_name)) +
            geom_histogram(binwidth = 1, fill = "blue", color = "black") +
            theme_minimal() +
            labs(x = col_name, y = "Count")
        
                                        # Save histogram plot
        ggsave(paste0("plots/", col_name, "_histogram.png"), hist_plot, width = 10, height = 6)
        
                                        # Save summary statistics table
        write.csv(summary_stats %>% as.list() %>% as.tibble(), paste0("tables/", col_name, "_summary.csv"), row.names = TRUE)
    }
    
                                        # Bar plots for categorical columns
    if (is_categorical(data[[col_name]])) {
        cat(sprintf("Cat column %s\n", col_name))

                                        # Create frequency table including NA values
        freq_table <- as.data.frame(table(data[[col_name]], useNA = "ifany"))
        
                                        # Sum the counts of the groups, including NA
        total_count <- sum(freq_table$Freq)
        
                                        # Check if the sum matches the total number of rows in raw_data
        if (total_count == nrow(raw_data)) {
            cat(sprintf("The sum of the groups for column '%s' matches the total number of rows in raw_data.\n", col_name))
        } else {
            cat(sprintf("Mismatch for column '%s': Sum of groups is %d, total rows in raw_data is %d.\n", col_name, total_count, nrow(raw_data)))
        }

        bar_plot <- ggplot(data, aes_string(x = col_name)) +
            geom_bar(fill = "steelblue") +
            theme_minimal() +
            theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
            labs(x = col_name, y = "Frequency")
        
                                        # Save bar plot
        ggsave(paste0("plots/", col_name, "_barplot.png"), bar_plot, width = 10, height = 6)
        
                                        # Frequency table
        freq_table <- as.data.frame(table(data[[col_name]]))
        
                                        # Save frequency table
        write.csv(freq_table, paste0("tables/", col_name, "_frequency.csv"), row.names = FALSE)
    }
}
