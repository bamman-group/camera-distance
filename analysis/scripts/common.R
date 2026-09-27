
.script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) == 0) return(getwd())
  dirname(normalizePath(sub("^--file=", "", file_arg[1])))
}

SCRIPT_DIR <- .script_dir()
ANALYSIS_DIR <- dirname(SCRIPT_DIR)
INTERMEDIATE_DIR <- file.path(ANALYSIS_DIR, "data", "analysis_outputs")
FIGURES_DIR <- file.path(ANALYSIS_DIR, "figures", "outputs")
ARTICLE_DIR <- file.path(ANALYSIS_DIR, "figures", "article")

dir.create(FIGURES_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(ARTICLE_DIR, recursive = TRUE, showWarnings = FALSE)

ARTICLE_FIGURES <- c(
  "popular_all_categories.png",
  "prestige_all_categories.png",
  "popular_all_categories_fixed_scale.png",
  "prestige_all_categories_fixed_scale.png",
  "all_distances_by_decade.png",
  "gender_diff_by_decade.png",
  "animated_vs_live_bars.png",
  "gender_rate_popular.png",
  "gender_rate_paired_groups.png",
  "popular_middle_distances.png",
  "popular_gender_diff_by_year.png",
  "prestige_gender_diff_by_year.png",
  "digitization_by_year_popular.png",
  "digitization_by_year_prestige.png",
  "digitization_by_year_indie.png",
  "genre_trends_popular.png",
  "aspect_ratio_trends_popular.png",
  "aspect_ratio_prevalence_popular.png",
  "mcu_gender_diff_popular.png",
  "mcu_gender_diff_prestige.png",
  "mcu_gender_diff_indie.png"
)

figure_path <- function(name) {
  file.path(if (name %in% ARTICLE_FIGURES) ARTICLE_DIR else FIGURES_DIR, name)
}

suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(scales))

source(file.path(ANALYSIS_DIR, "figures", "govorit_moskva_palette.R"))

CATEGORY_LABELS <- c(
  xcu = "Extreme close-up",
  cu = "Close-up",
  mcu = "Medium close-up",
  m = "Medium",
  ml = "Medium long",
  l = "Long",
  xls = "Extreme long",
  na = "N/A"
)
CATEGORY_DISPLAY_ORDER <- c("xcu", "cu", "mcu", "m", "ml", "l", "xls", "na")
CATEGORY_LEVELS <- unname(CATEGORY_LABELS[CATEGORY_DISPLAY_ORDER])

COLLECTION_ORDER <- c("popular", "prestige", "indie")
COLLECTION_LABELS <- c(popular = "Popular", prestige = "Prestige", indie = "Indie")
COLLECTION_COLORS <- setNames(unname(govorit_moskva(3)), COLLECTION_ORDER)

DISTANCE_LEVELS <- unname(CATEGORY_LABELS[setdiff(CATEGORY_DISPLAY_ORDER, "na")])
DISTANCE_ROW_LEVELS <- rev(DISTANCE_LEVELS)

only_distances <- function(df, levels = DISTANCE_LEVELS) {
  df <- df[df$category %in% levels, ]
  df$category <- factor(as.character(df$category), levels = levels)
  df
}

read_intermediate <- function(name) {
  read.csv(file.path(INTERMEDIATE_DIR, name), check.names = FALSE, stringsAsFactors = FALSE)
}

with_intermediate <- function(name, draw) {
  if (!file.exists(file.path(INTERMEDIATE_DIR, name))) {
    message(sprintf(
      "SKIPPED: %s not found in %s -- re-run the matching Python script and copy it over.",
      name, INTERMEDIATE_DIR
    ))
    return(invisible(FALSE))
  }
  draw(read_intermediate(name))
  invisible(TRUE)
}

theme_shot_distance <- function(...) {
  theme_minimal() +
    theme(
      axis.line = element_line(colour = "black", linewidth = 0.6),
      panel.grid.major = element_line(colour = "#e5e5e5", linewidth = 0.3),
      panel.grid.minor = element_blank(),
      text = element_text(size = 13),
      axis.title = element_text(size = 15),
      axis.text = element_text(size = 11),
      strip.text = element_text(size = 12),
      legend.text = element_text(size = 11),
      legend.title = element_text(size = 12),
      plot.title = element_text(hjust = 0.5)
    ) +
    theme(...)
}


GENDER_COLORS <- unname(govorit_moskva(2))
STATUS_COLORS <- unname(govorit_moskva_palette[c(4, 6)])

PANEL_LEVELS <- c("Collection", "Animation")
SCOPE_LEVELS <- c("popular", "prestige", "indie", "animation (human)", "animation (non-human)")

plot_bars <- function(df, colors, group_levels, title, ylab = "Share", facets = TRUE) {
  df$`_scope` <- factor(df$`_scope`, levels = SCOPE_LEVELS)
  df$panel <- factor(df$panel, levels = PANEL_LEVELS)
  df$group <- factor(df$group, levels = group_levels)

  dodge <- position_dodge(width = 0.7)
  p <- ggplot(df, aes(x = `_scope`, y = mean, fill = group)) +
    geom_col(position = dodge, width = 0.7) +
    geom_errorbar(aes(ymin = lo, ymax = hi), position = dodge, width = 0.2)
  if (facets) {
    p <- p + facet_wrap(~panel, scales = "free_x", nrow = 1)
  }
  p +
    scale_fill_manual(values = colors) +
    scale_y_continuous(labels = label_percent()) +
    labs(x = "", y = ylab, fill = "", title = title) +
    theme_shot_distance(axis.text.x = element_text(angle = 30, hjust = 1))
}

plot_collection_categories <- function(df, collection, ylim = NULL) {
  sub <- only_distances(df[df$collection == collection, ])
  color <- COLLECTION_COLORS[[collection]]
  ggplot(sub, aes(x = year, y = mean)) +
    geom_line(color = color) +
    geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.2, fill = color, color = NA) +
    facet_wrap(~category, ncol = 2, dir = "h", axes = "all_x",
               scales = if (is.null(ylim)) "free_y" else "fixed") +
    scale_y_continuous(labels = label_percent(), limits = ylim) +
    labs(x = NULL, y = "Prevalence (%)", title = COLLECTION_LABELS[[collection]]) +
    theme_shot_distance(plot.title = element_text(size = 16, face = "bold"))
}


save_figure <- function(plot, path, width, height) {
  ggsave(path, plot, width = width, height = height, dpi = 300)
}
