
.file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
.here <- if (length(.file_arg)) dirname(normalizePath(sub("^--file=", "", .file_arg[1]))) else getwd()
source(file.path(.here, "common.R"))

plot_digitization_by_year <- function(df, collection, xlim) {
  ggplot(df, aes(x = year, y = digitized)) +
    geom_col(fill = COLLECTION_COLORS[[collection]], colour = "black", linewidth = 0.2, width = 0.8) +
    coord_cartesian(xlim = xlim) +
    labs(x = NULL, y = "Digitized movies") +
    theme_shot_distance()
}

main <- function() {
  years <- unlist(lapply(COLLECTION_ORDER, function(collection) {
    path <- file.path(INTERMEDIATE_DIR, sprintf("digitization_by_year_%s.csv", collection))
    if (!file.exists(path)) return(NULL)
    read_intermediate(sprintf("digitization_by_year_%s.csv", collection))$year
  }))
  xlim <- range(years)

  for (collection in COLLECTION_ORDER) {
    with_intermediate(sprintf("digitization_by_year_%s.csv", collection), function(df) {
      save_figure(
        plot_digitization_by_year(df, collection, xlim),
        figure_path(sprintf("digitization_by_year_%s.png", collection)), 14, 5
      )
    })
  }
}

main()
