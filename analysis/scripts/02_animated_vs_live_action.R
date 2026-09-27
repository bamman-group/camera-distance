
.file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
.here <- if (length(.file_arg)) dirname(normalizePath(sub("^--file=", "", .file_arg[1]))) else getwd()
source(file.path(.here, "common.R"))
suppressPackageStartupMessages(library(ggpattern))

STATUS_LEVELS <- c("Live-action", "Animated")
STATUS_COLORS <- c("#FFC155", "#7B2E23")
STATUS_PATTERNS <- c("Live-action" = "none", "Animated" = "circle")



plot_whole_movie_bars <- function(df) {
  dodge <- position_dodge(width = 0.7)
  ggplot(only_distances(df), aes(x = category, y = mean, fill = status, pattern = status)) +
    geom_col_pattern(
      position = dodge, width = 0.7,
      colour = "black", linewidth = 0.3,
      pattern_fill = "white", pattern_colour = "white",
      pattern_density = 0.3, pattern_spacing = 0.025, pattern_angle = 45,
      pattern_key_scale_factor = 0.6
    ) +
    geom_errorbar(aes(ymin = lo, ymax = hi), position = dodge, width = 0.2) +
    scale_fill_manual(values = STATUS_COLORS) +
    scale_pattern_manual(values = STATUS_PATTERNS) +
    scale_y_continuous(labels = label_percent()) +
    labs(x = NULL, y = "Prevalence (%)", fill = "", pattern = "") +
    theme_shot_distance(axis.text.x = element_text(angle = 30, hjust = 1))
}


main <- function() {
  with_intermediate("animated_whole_movie_bootstrap_intervals.csv", function(whole) {
    whole$status <- factor(whole$status, levels = STATUS_LEVELS)

    save_figure(plot_whole_movie_bars(whole), figure_path("animated_vs_live_bars.png"), 12, 5)
  })
}

main()
