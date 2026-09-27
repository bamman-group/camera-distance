
.file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
.here <- if (length(.file_arg)) dirname(normalizePath(sub("^--file=", "", .file_arg[1]))) else getwd()
source(file.path(.here, "common.R"))

FIXED_Y_LIMITS <- list(
  popular  = c(0, 0.55),
  prestige = c(0, 0.60)
)

DECADE_MIN_MOVIES <- 10

# A year resting on a handful of films gives a bootstrap interval that is
# either degenerate (n = 1 resamples to itself, so lo == hi) or so wide it
# swamps the panel, so drop those years rather than plotting them.
YEAR_MIN_MOVIES <- 5


plot_decade_bars <- function(df) {
  sub <- only_distances(df)
  sub <- sub[sub$n_movies > DECADE_MIN_MOVIES, ]
  if (nrow(sub) == 0) {
    stop(sprintf("no decade rests on more than %d movies", DECADE_MIN_MOVIES))
  }
  sub$decade <- factor(sprintf("%ds", as.integer(sub$decade)),
                       levels = sprintf("%ds", sort(unique(as.integer(sub$decade)))))
  sub$collection <- factor(sub$collection, levels = COLLECTION_ORDER)

  dodge <- position_dodge(width = 0.8, preserve = "single")
  ggplot(sub, aes(x = decade, y = mean, fill = collection)) +
    geom_col(position = dodge, width = 0.8) +
    geom_errorbar(aes(ymin = lo, ymax = hi), position = dodge, width = 0.25) +
    facet_wrap(~category, ncol = 2, dir = "h", axes = "all_x", scales = "free_y") +
    scale_fill_govorit_moskva(3, labels = unname(COLLECTION_LABELS)) +
    scale_y_continuous(labels = label_percent()) +
    labs(x = NULL, y = "Prevalence (%)", fill = "") +
    theme_shot_distance(axis.text.x = element_text(angle = 45, hjust = 1))
}


main <- function() {
  with_intermediate("bootstrap_intervals_by_year.csv", function(all_bootstrapped) {
    dropped <- all_bootstrapped[all_bootstrapped$n_movies < YEAR_MIN_MOVIES, ]
    all_bootstrapped <- all_bootstrapped[all_bootstrapped$n_movies >= YEAR_MIN_MOVIES, ]
    if (nrow(all_bootstrapped) == 0) {
      stop(sprintf("no year rests on at least %d movies", YEAR_MIN_MOVIES))
    }
    for (coll in COLLECTION_ORDER) {
      years <- sort(unique(dropped$year[dropped$collection == coll]))
      if (length(years)) {
        message(sprintf("%s: dropped %d year(s) with fewer than %d movies: %s",
                        coll, length(years), YEAR_MIN_MOVIES,
                        paste(as.integer(years), collapse = ", ")))
      }
    }

    all_bootstrapped$category <- factor(all_bootstrapped$category, levels = CATEGORY_LEVELS)
    all_bootstrapped$collection <- factor(all_bootstrapped$collection, levels = COLLECTION_ORDER)

    for (collection in c("popular", "prestige")) {
      save_figure(plot_collection_categories(all_bootstrapped, collection),
                  figure_path(sprintf("%s_all_categories.png", collection)), 9, 12)
    }


    for (collection in c("popular", "prestige")) {
      save_figure(plot_collection_categories(all_bootstrapped, collection, ylim = FIXED_Y_LIMITS[[collection]]),
                  figure_path(sprintf("%s_all_categories_fixed_scale.png", collection)), 9, 12)
    }
  })

  with_intermediate("bootstrap_intervals_by_decade.csv", function(by_decade) {
    save_figure(plot_decade_bars(by_decade),
                figure_path("all_distances_by_decade.png"), 12, 14)
  })
}

main()
