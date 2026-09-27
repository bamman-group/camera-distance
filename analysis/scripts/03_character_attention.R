
.file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
.here <- if (length(.file_arg)) dirname(normalizePath(sub("^--file=", "", .file_arg[1]))) else getwd()
source(file.path(.here, "common.R"))
suppressPackageStartupMessages(library(ggpattern))


plot_gender_diff <- function(df, collection) {
  sub <- df[df$collection == collection, ]
  sub$category <- unname(CATEGORY_LABELS[sub$category])
  sub <- only_distances(sub, RATE_LEVELS)
  color <- COLLECTION_COLORS[[collection]]

  ggplot(sub, aes(x = year, y = diff_mean)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey40", linewidth = 0.4) +
    geom_ribbon(aes(ymin = diff_lo, ymax = diff_hi), alpha = 0.2, fill = color, color = NA) +
    geom_line(color = color) +
    facet_wrap(~category, nrow = 1, axes = "all_x", scales = "fixed") +
    scale_y_continuous(labels = label_percent()) +
    labs(x = NULL, y = "Share of frames, women − men",
         title = COLLECTION_LABELS[[collection]]) +
    theme_shot_distance(plot.title = element_text(size = 16, face = "bold"))
}



plot_mcu_gender_diff <- function(df, collection) {
  sub <- df[df$collection == collection & df$category == "mcu", ]
  color <- COLLECTION_COLORS[[collection]]

  ggplot(sub, aes(x = year, y = diff_mean)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey40", linewidth = 0.4) +
    geom_ribbon(aes(ymin = diff_lo, ymax = diff_hi), alpha = 0.2, fill = color, color = NA) +
    geom_line(color = color) +
    scale_y_continuous(labels = label_percent()) +
    labs(x = NULL, y = "Medium close-up, women − men (%)",
         title = COLLECTION_LABELS[[collection]]) +
    theme_shot_distance(plot.title = element_text(size = 16, face = "bold"))
}



RATE_LEVELS <- unname(CATEGORY_LABELS[c("mcu", "m", "ml", "l")])
GENDER_PATTERNS <- c("Men" = "none", "Women" = "circle")

# Same decade bucketing and minimum as all_distances_by_decade.png in
# 01_temporal_trends.R, so the two decade figures cover the same decades.
DECADE_MIN_MOVIES <- 10

plot_gender_diff_decade <- function(df) {
  sub <- df[df$n_movies > DECADE_MIN_MOVIES, ]
  sub$category <- unname(CATEGORY_LABELS[sub$category])
  sub <- only_distances(sub, RATE_LEVELS)
  if (nrow(sub) == 0) {
    stop(sprintf("no decade rests on more than %d movies", DECADE_MIN_MOVIES))
  }
  sub$decade <- factor(sprintf("%ds", as.integer(sub$decade)),
                       levels = sprintf("%ds", sort(unique(as.integer(sub$decade)))))
  sub$collection <- factor(sub$collection, levels = COLLECTION_ORDER)

  dodge <- position_dodge(width = 0.8, preserve = "single")
  ggplot(sub, aes(x = decade, y = diff_mean, fill = collection)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey40", linewidth = 0.4) +
    geom_col(position = dodge, width = 0.8) +
    geom_errorbar(aes(ymin = diff_lo, ymax = diff_hi), position = dodge, width = 0.25) +
    facet_wrap(~category, ncol = 2, dir = "h", axes = "all_x", scales = "free_y") +
    scale_fill_govorit_moskva(3, labels = unname(COLLECTION_LABELS)) +
    scale_y_continuous(labels = label_percent()) +
    labs(x = NULL, y = "Share of frames, women − men", fill = "") +
    theme_shot_distance(axis.text.x = element_text(angle = 45, hjust = 1))
}

plot_gender_rate_table <- function(df) {
  df$category <- unname(CATEGORY_LABELS[df$category])
  df <- only_distances(df, RATE_LEVELS)

  series <- function(label, mean, lo, hi) {
    data.frame(category = df$category, series = label, mean = mean, lo = lo, hi = hi)
  }
  long <- rbind(
    series("Men", df$m_mean, df$m_lo, df$m_hi),
    series("Women", df$w_mean, df$w_lo, df$w_hi)
  )
  long$series <- factor(long$series, levels = c("Men", "Women"))

  dodge <- position_dodge(width = 0.7)
  ggplot(long, aes(x = category, y = mean, fill = series, pattern = series)) +
    geom_col_pattern(
      position = dodge, width = 0.7,
      colour = "black", linewidth = 0.3,
      pattern_fill = "white", pattern_colour = "white",
      pattern_density = 0.3, pattern_spacing = 0.025, pattern_angle = 45,
      pattern_key_scale_factor = 0.6
    ) +
    geom_errorbar(aes(ymin = lo, ymax = hi), position = dodge, width = 0.2) +
    scale_fill_manual(values = GENDER_COLORS) +
    scale_pattern_manual(values = GENDER_PATTERNS) +
    scale_y_continuous(labels = label_percent()) +
    labs(x = NULL, y = "Prevalence (%)", fill = "", pattern = "") +
    theme_shot_distance(axis.text.x = element_text(angle = 30, hjust = 1))
}



PAIRED_GROUPS <- c("Animated human", "Animated non-human", "Live-action")

plot_paired_gender_rates <- function(df) {
  sub <- df
  sub$category <- unname(CATEGORY_LABELS[sub$category])
  sub <- only_distances(sub)
  sub$category <- droplevels(sub$category)
  sub$group <- factor(sub$group, levels = PAIRED_GROUPS)

  separators <- seq_len(nlevels(sub$category) - 1) + 0.5

  dodge <- position_dodge(width = 0.6, preserve = "single")
  ggplot(sub, aes(x = category, y = diff_mean, fill = group)) +
    geom_vline(xintercept = separators, color = "grey75", linewidth = 0.4) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey40", linewidth = 0.4) +
    geom_col(position = dodge, width = 0.6) +
    geom_errorbar(aes(ymin = diff_lo, ymax = diff_hi), position = dodge, width = 0.15) +
    scale_fill_govorit_moskva(3) +
    scale_y_continuous(labels = label_percent()) +
    labs(x = NULL, y = "Prevalence difference, women − men (%)", fill = "") +
    theme_shot_distance(
      axis.text.x = element_text(angle = 30, hjust = 1),
      panel.grid.major.x = element_blank()
    )
}


main <- function() {
  with_intermediate("gender_rate_diff_by_year.csv", function(gender_diff) {
    for (collection in c("popular", "prestige")) {
      save_figure(plot_gender_diff(gender_diff, collection),
                  figure_path(sprintf("%s_gender_diff_by_year.png", collection)), 18, 5)
    }
    for (collection in c("popular", "prestige", "indie")) {
      save_figure(plot_mcu_gender_diff(gender_diff, collection),
                  figure_path(sprintf("mcu_gender_diff_%s.png", collection)), 10, 6)
    }
  })

  with_intermediate("gender_rate_diff_by_decade.csv", function(gender_diff_decade) {
    save_figure(plot_gender_diff_decade(gender_diff_decade),
                figure_path("gender_diff_by_decade.png"), 12, 8)
  })

  with_intermediate("gender_rate_popular.csv", function(popular) {
    p <- plot_gender_rate_table(popular) +
      theme(
        legend.text = element_text(size = 16.5),
        legend.title = element_text(size = 18),
        legend.key.size = unit(1.8, "lines")
      )
    save_figure(p, figure_path("gender_rate_popular.png"), 9, 6)
  })

  with_intermediate("gender_rate_paired_groups.csv", function(paired) {
    save_figure(plot_paired_gender_rates(paired),
                figure_path("gender_rate_paired_groups.png"), 10, 6)
  })

}

main()
