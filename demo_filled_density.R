library(tidyverse);                                        # load required libraries


                                        # Load data
data <- read_csv('derived_data/demographics-with-projection.csv')

                                        # Create the filled 2D density plot with facets
p <- ggplot(data, aes(x=E1, y=E2)) +
    geom_density_2d_filled(aes(fill=..level..)) +
    facet_wrap(~`STUDYID`) +
    theme_minimal() +
    guides(fill=FALSE);

# Display the plot
#print(p)

# Save the plot
ggsave('figures/filled_density_faceted_plot.png', plot=p, width=10, height=10)


p <- ggplot(data, aes(x=E1, y=E2)) +
    geom_density_2d_filled(aes(fill=..level..)) +
    facet_wrap(~`Group`) +
    theme_minimal() +
    guides(fill=FALSE);

# Display the plot
#print(p)

# Save the plot
ggsave('figures/filled_density_faceted_plot_outcome.png', plot=p, width=10, height=10)

