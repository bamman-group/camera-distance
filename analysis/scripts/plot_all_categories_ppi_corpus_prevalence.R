
suppressPackageStartupMessages({
  library(ggplot2)
  library(scales)
})

.script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 0) return(getwd())
  dirname(normalizePath(sub("^--file=", "", file_arg[1])))
}
ARTICLE_DIR <- file.path(.script_dir(), "..", "figures", "article")
dir.create(ARTICLE_DIR, recursive = TRUE, showWarnings = FALSE)
source(file.path(.script_dir(), "..", "figures", "govorit_moskva_palette.R"))

theme_shot_distance <- function(...) {
  theme_minimal() +
    theme(
      axis.line = element_line(colour = "black", linewidth = 0.6),
      panel.grid.major = element_line(colour = "#e5e5e5", linewidth = 0.3),
      panel.grid.minor = element_blank(),
      text = element_text(size = 13),
      axis.title = element_text(size = 15),
      axis.text = element_text(size = 10),
      strip.text = element_text(size = 11),
      legend.text = element_text(size = 11),
      legend.title = element_text(size = 12),
      plot.title = element_text(hjust = 0.5)
    ) +
    theme(...)
}

args <- commandArgs(trailingOnly = TRUE)
csv_dir <- if (length(args) >= 1) args[1] else file.path(.script_dir(), "..", "data", "analysis_outputs")
collection <- if (length(args) >= 2) args[2] else "popular"

CATEGORY_LABELS <- c(
  xcu = "Extreme close-up", cu = "Close-up", mcu = "Medium close-up", m = "Medium",
  ml = "Medium long", l = "Long", xls = "Extreme long", na = "Intertitle"
)

SERIES_COLORS <- c(
  "Raw" = unname(govorit_moskva_palette[["Lampshade Orange"]]),
  "Corrected" = unname(govorit_moskva_palette[["Floor Brick"]])
)

load_categories <- function(categories) {
  do.call(rbind, lapply(categories, function(cat) {
    path <- file.path(csv_dir, sprintf("%s_ppi_corpus_prevalence_%s.csv", cat, collection))
    df <- read.csv(path, stringsAsFactors = FALSE)
    rbind(
      data.frame(category = CATEGORY_LABELS[[cat]], series = "Raw",
                 year = df$year, mean = df$raw_mean, lo = df$raw_lo, hi = df$raw_hi),
      data.frame(category = CATEGORY_LABELS[[cat]], series = "Corrected",
                 year = df$year, mean = df$ppi_mean, lo = df$ppi_lo, hi = df$ppi_hi)
    )
  }))
}

plot_group <- function(categories, out_path) {
  long <- load_categories(categories)
  long$category <- factor(long$category, levels = unname(CATEGORY_LABELS[categories]))
  long$series <- factor(long$series, levels = c("Raw", "Corrected"))

  long$mean <- pmax(long$mean, 0)
  long$lo <- pmax(long$lo, 0)

  p <- ggplot(long, aes(x = year, y = mean, color = series, fill = series)) +
    geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.2, colour = NA) +
    geom_line(linewidth = 0.6) +
    facet_grid(category ~ series, scales = "free_y") +
    coord_cartesian(ylim = c(0, NA)) +
    scale_color_manual(values = SERIES_COLORS, guide = "none") +
    scale_fill_manual(values = SERIES_COLORS, guide = "none") +
    scale_y_continuous(labels = label_percent()) +
    labs(x = NULL, y = "Prevalence (%)") +
    theme_shot_distance()

  ggsave(out_path, p, width = 8, height = 10, dpi = 300, limitsize = FALSE)
  cat(sprintf("Wrote %s\n", out_path))
}

plot_group(c("xcu", "cu", "xls", "na"),
           file.path(ARTICLE_DIR, sprintf("categories_edge_ppi_corpus_prevalence_%s.png", collection)))
plot_group(c("mcu", "m", "ml", "l"),
           file.path(ARTICLE_DIR, sprintf("categories_middle_ppi_corpus_prevalence_%s.png", collection)))
