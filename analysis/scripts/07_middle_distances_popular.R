
.file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
.here <- if (length(.file_arg)) dirname(normalizePath(sub("^--file=", "", .file_arg[1]))) else getwd()
source(file.path(.here, "common.R"))

MIDDLE_LEVELS <- unname(CATEGORY_LABELS[c("mcu", "m", "ml", "l")])
POPULAR_COLOR <- COLLECTION_COLORS[["popular"]]

plot_middle_distances <- function(df) {
  sub <- only_distances(df, MIDDLE_LEVELS)
  ggplot(sub, aes(x = year, y = mean)) +
    geom_line(color = POPULAR_COLOR) +
    geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.2, fill = POPULAR_COLOR, color = NA) +
    facet_wrap(~category, nrow = 1, axes = "all_x", scales = "fixed") +
    scale_y_continuous(labels = label_percent()) +
    labs(x = NULL, y = "Prevalence (%)") +
    theme_shot_distance(strip.text = element_text(size = 18))
}

main <- function() {
  with_intermediate("middle_distances_popular_by_year.csv", function(df) {
    save_figure(plot_middle_distances(df),
                figure_path("popular_middle_distances.png"), 18, 5)
  })
}

main()
