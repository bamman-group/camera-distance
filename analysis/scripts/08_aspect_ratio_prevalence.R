
.file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
.here <- if (length(.file_arg)) dirname(normalizePath(sub("^--file=", "", .file_arg[1]))) else getwd()
source(file.path(.here, "common.R"))
suppressPackageStartupMessages(library(patchwork))

HISTORICAL_ORDER <- c("1.33:1", "1.37:1", "2.35:1", "1.85:1", "2.39:1")

plot_one_ratio <- function(sub, color, title, show_ylab) {
  ggplot(sub, aes(x = year, y = mean)) +
    geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.2, fill = color, color = NA) +
    geom_line(color = color) +
    scale_y_continuous(labels = label_percent(), limits = c(0, 1)) +
    labs(x = NULL, y = if (show_ylab) "Share of popular movies" else NULL, title = title) +
    theme_shot_distance(plot.title = element_text(size = 13, face = "bold", hjust = 0.5))
}

plot_aspect_ratio_prevalence <- function(df) {
  ratio_levels <- intersect(HISTORICAL_ORDER, unique(df$aspect_ratio))
  ratio_levels <- c(ratio_levels, setdiff(unique(df$aspect_ratio), ratio_levels))
  colors <- unname(govorit_moskva(length(ratio_levels)))

  plots <- lapply(seq_along(ratio_levels), function(i) {
    sub <- df[df$aspect_ratio == ratio_levels[i], ]
    plot_one_ratio(sub, colors[i], ratio_levels[i], show_ylab = i %in% c(1, 4))
  })

  top <- plots[[1]] + plots[[2]] + plots[[3]] + plot_layout(nrow = 1)
  bottom <- plot_spacer() + plots[[4]] + plots[[5]] + plot_spacer() +
    plot_layout(nrow = 1, widths = c(0.5, 1, 1, 0.5))
  top / bottom
}

main <- function() {
  with_intermediate("aspect_ratio_prevalence_by_year.csv", function(df) {
    save_figure(plot_aspect_ratio_prevalence(df),
                figure_path("aspect_ratio_prevalence_popular.png"), 12, 8)
  })
}

main()
