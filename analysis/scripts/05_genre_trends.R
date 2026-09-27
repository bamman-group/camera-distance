
.file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
.here <- if (length(.file_arg)) dirname(normalizePath(sub("^--file=", "", .file_arg[1]))) else getwd()
source(file.path(.here, "common.R"))

MIDDLE_LEVELS <- unname(CATEGORY_LABELS[c("mcu", "m", "ml", "l")])

plot_genre_trends <- function(df) {
  df$category <- factor(df$category, levels = MIDDLE_LEVELS)
  genre_levels <- unique(df$genre)
  df$genre <- factor(df$genre, levels = genre_levels)

  ggplot(df, aes(x = year, y = mean, color = genre)) +
    geom_point(size = 0.8) +
    facet_wrap(~category, nrow = 1, axes = "all_x", scales = "fixed") +
    scale_color_govorit_moskva(length(genre_levels)) +
    scale_y_continuous(labels = label_percent()) +
    labs(x = NULL, y = "Prevalence (%)", color = "") +
    theme_shot_distance()
}

main <- function() {
  with_intermediate("genre_trends_by_year.csv", function(df) {
    save_figure(plot_genre_trends(df),
                figure_path("genre_trends_popular.png"), 18, 5)
  })
}

main()
