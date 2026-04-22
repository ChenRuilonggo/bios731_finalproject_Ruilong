# ─────────────────────────────────────────────────────────────
# BIOS 731 Final Project — Figures
# Ruilong Chen
#
# Input : results/all_results.csv
# Output: figures/fig_coverage.pdf
#         figures/fig_width.pdf
#         figures/fig_fpr.pdf
#         figures/fig_time.pdf
#         figures/fig_summary_tile.pdf
#
# The script auto-discovers which (n, p) combos exist in data
# so it works correctly for both --quick and full runs.
# ─────────────────────────────────────────────────────────────

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(scales)
  library(patchwork)
  library(forcats)
})

dir.create("figures", showWarnings = FALSE)

ALPHA   <- 0.05
NOMINAL <- 1 - ALPHA

theme_proj <- function() {
  theme_bw(base_size = 11) +
    theme(
      strip.background = element_rect(fill = "#f0f0f0", colour = "grey70"),
      strip.text       = element_text(size = 9),
      legend.position  = "bottom",
      legend.title     = element_text(face = "bold"),
      panel.grid.minor = element_blank()
    )
}

METHOD_COLORS <- c(
  "Debiased Lasso"       = "#E41A1C",
  "Bootstrap Percentile" = "#377EB8",
  "Bootstrap-t"          = "#4DAF4A"
)
METHOD_SHAPES <- c(
  "Debiased Lasso"       = 16,
  "Bootstrap Percentile" = 17,
  "Bootstrap-t"          = 15
)

# ── 1. Load & tidy ────────────────────────────────────────────
raw <- read_csv("results/all_results.csv", show_col_types = FALSE)

long <- raw |>
  pivot_longer(
    cols      = matches("^(debiased|pct|boot_t)_(bias|cover|width|fpr)$"),
    names_to  = c("method_key", "metric"),
    names_sep = "_(?=bias$|cover$|width$|fpr$)"
  ) |>
  mutate(
    method = recode(method_key,
      debiased = "Debiased Lasso",
      pct      = "Bootstrap Percentile",
      boot_t   = "Bootstrap-t"
    ),
    method     = factor(method, levels = names(METHOD_COLORS)),
    rho        = factor(rho),
    error_dist = recode(error_dist,
      normal = "Normal errors", t3 = "t(3) errors"),
    structure  = fct_relevel(structure, "independent", "ar1", "block")
  )

summ <- long |>
  group_by(method, n, p, rho, structure, error_dist, metric) |>
  summarise(value = mean(value, na.rm = TRUE), .groups = "drop")

get_metric <- function(m) {
  summ |> filter(metric == m) |> rename(!!m := value) |> select(-metric)
}

cov_df  <- get_metric("cover")
wid_df  <- get_metric("width")
fpr_df  <- get_metric("fpr")

time_df <- raw |>
  select(n, p, rho, structure, error_dist,
         `Debiased Lasso`       = time_debiased,
         `Bootstrap Percentile` = time_pct,
         `Bootstrap-t`          = time_boot_t) |>
  pivot_longer(cols = c("Debiased Lasso", "Bootstrap Percentile", "Bootstrap-t"),
               names_to = "method", values_to = "time") |>
  mutate(
    method     = factor(method, levels = names(METHOD_COLORS)),
    rho        = factor(rho),
    error_dist = recode(error_dist,
      normal = "Normal errors", t3 = "t(3) errors"),
    structure  = fct_relevel(structure, "independent", "ar1", "block")
  ) |>
  group_by(method, n, p, rho, structure, error_dist) |>
  summarise(time = mean(time, na.rm = TRUE), .groups = "drop")

# Discover all (n, p) combinations present in data
np_combos <- cov_df |> distinct(n, p) |> arrange(n, p)
message(sprintf("Found %d (n, p) combination(s): %s",
  nrow(np_combos),
  paste(sprintf("(%d,%d)", np_combos$n, np_combos$p), collapse = " ")))


# ── 2. Helper: build one panel ────────────────────────────────
panel_coverage <- function(n_val, p_val, df) {
  df |> filter(n == n_val, p == p_val) |>
    ggplot(aes(x = rho, y = cover, colour = method,
               shape = method, group = method)) +
    geom_hline(yintercept = NOMINAL, linetype = "dashed",
               colour = "grey40", linewidth = 0.5) +
    geom_line(linewidth = 0.7) +
    geom_point(size = 2.5) +
    facet_grid(error_dist ~ structure, drop = TRUE) +
    scale_colour_manual(values = METHOD_COLORS, name = "Method") +
    scale_shape_manual(values  = METHOD_SHAPES, name = "Method") +
    scale_y_continuous(labels = percent_format(accuracy = 1),
                       limits = c(0, 1.02)) +
    labs(title = sprintf("n = %d, p = %d", n_val, p_val),
         x = expression(rho), y = "Coverage") +
    theme_proj()
}

panel_width <- function(n_val, p_val, df) {
  df |> filter(n == n_val, p == p_val) |>
    ggplot(aes(x = rho, y = width, colour = method,
               shape = method, group = method)) +
    geom_line(linewidth = 0.7) +
    geom_point(size = 2.5) +
    facet_grid(error_dist ~ structure, drop = TRUE) +
    scale_colour_manual(values = METHOD_COLORS, name = "Method") +
    scale_shape_manual(values  = METHOD_SHAPES, name = "Method") +
    labs(title = sprintf("n = %d, p = %d", n_val, p_val),
         x = expression(rho), y = "CI width") +
    theme_proj()
}

panel_fpr <- function(n_val, p_val, df) {
  df |> filter(n == n_val, p == p_val) |>
    ggplot(aes(x = rho, y = fpr, colour = method,
               shape = method, group = method)) +
    geom_hline(yintercept = ALPHA, linetype = "dashed",
               colour = "grey40", linewidth = 0.5) +
    geom_line(linewidth = 0.7) +
    geom_point(size = 2.5) +
    facet_grid(error_dist ~ structure, drop = TRUE) +
    scale_colour_manual(values = METHOD_COLORS, name = "Method") +
    scale_shape_manual(values  = METHOD_SHAPES, name = "Method") +
    scale_y_continuous(labels = percent_format(accuracy = 1),
                       limits = c(0, NA)) +
    labs(title = sprintf("n = %d, p = %d", n_val, p_val),
         x = expression(rho), y = "FPR") +
    theme_proj()
}


# ── 3. Assemble multi-panel figures ───────────────────────────
assemble_and_save <- function(panel_fn, df, title, caption, fname,
                              width = 14, height_per_row = 5) {
  panels <- lapply(seq_len(nrow(np_combos)), function(i)
    panel_fn(np_combos$n[i], np_combos$p[i], df)
  )
  # Arrange in rows of 2
  ncols <- min(2, length(panels))
  fig   <- wrap_plots(panels, ncol = ncols, guides = "collect") &
    theme(legend.position = "bottom")
  fig   <- fig + plot_annotation(
    title   = title,
    caption = caption,
    theme   = theme(plot.title = element_text(face = "bold", size = 13))
  )
  nrows <- ceiling(length(panels) / ncols)
  ggsave(fname, fig,
         width  = width,
         height = max(4, nrows * height_per_row),
         device = cairo_pdf)
  message("Saved ", fname)
}

assemble_and_save(
  panel_coverage, cov_df,
  title   = "Figure 1: Empirical Coverage of 95% Confidence Intervals",
  caption = "Dashed line = nominal 95% level.",
  fname   = "figures/fig_coverage.pdf"
)

assemble_and_save(
  panel_width, wid_df,
  title   = "Figure 2: Average Confidence Interval Width",
  caption = "",
  fname   = "figures/fig_width.pdf"
)

assemble_and_save(
  panel_fpr, fpr_df,
  title   = "Figure 3: False Positive Rate for a Null Coefficient",
  caption = "Dashed line = nominal 5% level.",
  fname   = "figures/fig_fpr.pdf"
)


# ── 4. Figure 4: Computation Time ─────────────────────────────
# Use the first available (n, p) combo as the representative scenario
rep_n <- np_combos$n[1];  rep_p <- np_combos$p[1]

fig_time <- time_df |>
  filter(n == rep_n, p == rep_p) |>
  ggplot(aes(x = rho, y = time, fill = method)) +
    geom_col(position = "dodge", width = 0.65) +
    facet_grid(error_dist ~ structure, drop = TRUE) +
    scale_fill_manual(values = METHOD_COLORS, name = "Method") +
    scale_y_continuous(labels = label_number(suffix = " s")) +
    labs(
      title   = sprintf(
        "Figure 4: Mean Computation Time per Replicate (n = %d, p = %d)",
        rep_n, rep_p),
      x       = expression(rho ~ "(predictor correlation)"),
      y       = "Time (seconds)",
      caption = "Averaged over simulation replicates."
    ) +
    theme_proj() +
    theme(legend.position = "bottom")

ggsave("figures/fig_time.pdf", fig_time,
       width = 12, height = 6, device = cairo_pdf)
message("Saved figures/fig_time.pdf")


# ── 5. Figure 5: Summary Tile (coverage deficit) ──────────────
tile_df <- cov_df |>
  mutate(
    scenario = paste0(
      "n=", n, " p=", p, "\n",
      structure, " ρ=", rho, "\n",
      error_dist
    ),
    deficit = cover - NOMINAL
  )

clamp_lim <- max(abs(tile_df$deficit), na.rm = TRUE)
clamp_lim <- min(clamp_lim, 0.35)   # cap colour scale for readability

fig_tile <- ggplot(tile_df,
    aes(x = method, y = fct_rev(scenario), fill = deficit)) +
  geom_tile(colour = "white", linewidth = 0.3) +
  geom_text(aes(label = sprintf("%.2f", cover)),
            size = 2.3, colour = "black") +
  scale_fill_gradient2(
    low      = "#d73027", mid = "white", high = "#1a9850",
    midpoint = 0,
    limits   = c(-clamp_lim, clamp_lim),
    oob      = squish,
    name     = "Coverage\n– nominal",
    labels   = label_number(style_positive = "plus", accuracy = 0.01)
  ) +
  scale_x_discrete(position = "top") +
  labs(
    title   = "Figure 5: Coverage Deficit (empirical − 0.95) by Scenario",
    x = NULL, y = NULL,
    caption = "Green = above nominal; red = below nominal. Values show empirical coverage."
  ) +
  theme_proj() +
  theme(
    axis.text.x     = element_text(size = 9, face = "bold"),
    axis.text.y     = element_text(size = 6.5),
    legend.position = "right"
  )

tile_height <- max(6, nrow(tile_df) / 3 * 0.35 + 2)
ggsave("figures/fig_summary_tile.pdf", fig_tile,
       width = 10, height = tile_height, device = cairo_pdf)
message("Saved figures/fig_summary_tile.pdf")

message("\nAll figures written to figures/")
