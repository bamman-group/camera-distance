
suppressPackageStartupMessages({
  library(ggplot2)
  library(scales)
})

.script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 0) return(getwd())
  dirname(normalizePath(sub("^--file=", "", file_arg[1])))
}
source(file.path(.script_dir(), "..", "..", "analysis", "figures", "govorit_moskva_palette.R"))

theme_shot_distance <- function(...) {
  theme_minimal() +
    theme(
      axis.line = element_line(colour = "black", linewidth = 0.6),
      panel.grid.major = element_line(colour = "#e5e5e5", linewidth = 0.3),
      panel.grid.minor = element_blank(),
      text = element_text(size = 13),
      axis.title = element_text(size = 15),
      axis.text = element_text(size = 11),
      plot.title = element_text(hjust = 0.5)
    ) +
    theme(...)
}

args <- commandArgs(trailingOnly = TRUE)
out_path <- if (length(args) >= 1) args[1] else "../figures/article/tv_penetration.png"

df <- data.frame(
  year = c(1950, 1951, 1952, 1953, 1954, 1955, 1956, 1960, 1965, 1970, 1975, 1977, 1978, 1979, 1980),
  percent = c(9.0, 23.5, 34.1, 44.6, 55.7, 64.5, 72.0, 87.1, 92.6, 95.3, 97.1, 97.4, 97.6, 97.7, 97.9)
)

p <- ggplot(df, aes(x = year, y = percent / 100)) +
  geom_line(color = unname(govorit_moskva_palette[["Wall Teal"]]), linewidth = 0.8) +
  geom_point(color = unname(govorit_moskva_palette[["Wall Teal"]]), size = 2) +
  scale_y_continuous(labels = label_percent(), limits = c(0, 1)) +
  scale_x_continuous(breaks = df$year) +
  labs(x = NULL, y = "Households with a TV set (%)") +
  theme_shot_distance(axis.text.x = element_text(angle = 30, hjust = 1))

ggsave(out_path, p, width = 8, height = 5, dpi = 300)
cat(sprintf("Wrote %s\n", out_path))
