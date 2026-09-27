
govorit_moskva_palette <- c(
  "Wall Teal"        = "#4A8890",
  "Lampshade Orange" = "#EA8A2A",
  "Floor Brick"      = "#7B2E23",
  "Building Mauve"   = "#988391",
  "Sky Yellow"       = "#FFC155",
  "Caption Pink"     = "#FFC4C1"
)

govorit_moskva <- function(n = 5) {
  if (n < 1 || n > length(govorit_moskva_palette)) {
    stop("n must be between 1 and ", length(govorit_moskva_palette))
  }
  govorit_moskva_palette[seq_len(n)]
}

scale_fill_govorit_moskva <- function(n = 5, ...) {
  ggplot2::scale_fill_manual(values = unname(govorit_moskva(n)), ...)
}

scale_color_govorit_moskva <- function(n = 5, ...) {
  ggplot2::scale_color_manual(values = unname(govorit_moskva(n)), ...)
}
