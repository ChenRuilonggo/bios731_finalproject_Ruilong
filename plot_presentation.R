# plot_presentation.R — clean figures for Beamer slides
# Output: figures/slide_overall.pdf
#         figures/slide_rho.pdf
#         figures/slide_np.pdf
#         figures/slide_structure.pdf
#         figures/slide_error.pdf

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(scales)
  library(forcats)
})

dir.create("figures", showWarnings = FALSE)

NOMINAL <- 0.95

METHOD_COLORS <- c(
  "Naive"        = "#888888",
  "Debiased"     = "#E41A1C",
  "Percentile"   = "#377EB8",
  "Bootstrap-t"  = "#4DAF4A"
)
METHOD_SHAPES <- c(
  "Naive"        = 4,
  "Debiased"     = 16,
  "Percentile"   = 17,
  "Bootstrap-t"  = 15
)

raw <- read_csv("results/all_results.csv", show_col_types = FALSE)

# Tidy: include naive
long <- raw |>
  pivot_longer(
    cols      = matches("^(naive|debiased|pct|boot_t)_cover$"),
    names_to  = "method_key",
    values_to = "cover"
  ) |>
  mutate(
    method_key = sub("_cover$", "", method_key),
    method = recode(method_key,
      naive    = "Naive",
      debiased = "Debiased",
      pct      = "Percentile",
      boot_t   = "Bootstrap-t"
    ),
    method = factor(method, levels = names(METHOD_COLORS))
  )

theme_slide <- function() {
  theme_bw(base_size = 16) +
    theme(
      legend.position  = "bottom",
      legend.title     = element_blank(),
      legend.text      = element_text(size = 14),
      axis.title       = element_text(size = 15),
      axis.text        = element_text(size = 13),
      strip.text       = element_text(size = 13),
      strip.background = element_rect(fill = "#f0f0f0", colour = "grey70"),
      panel.grid.minor = element_blank(),
      plot.title       = element_blank()
    )
}

hline <- geom_hline(yintercept = NOMINAL, linetype = "dashed",
                    colour = "grey40", linewidth = 0.6)

pct_y <- scale_y_continuous(labels = percent_format(accuracy = 1),
                              limits = c(0, 1.02))

# ── 1. Overall: coverage + bias + width (3-panel facet) ──────
overall_cover <- long |>
  group_by(method) |>
  summarise(value = mean(cover, na.rm = TRUE), .groups = "drop") |>
  mutate(metric = "Coverage")

overall_bias <- raw |>
  pivot_longer(cols = matches("^(naive|debiased|pct|boot_t)_bias$"),
               names_to = "method_key", values_to = "value") |>
  mutate(method_key = sub("_bias$", "", method_key),
         method = recode(method_key,
           naive="Naive", debiased="Debiased",
           pct="Percentile", boot_t="Bootstrap-t"),
         method = factor(method, levels = names(METHOD_COLORS))) |>
  group_by(method) |>
  summarise(value = mean(value, na.rm = TRUE), .groups = "drop") |>
  mutate(metric = "Bias")

overall_width <- raw |>
  pivot_longer(cols = matches("^(naive|debiased|pct|boot_t)_width$"),
               names_to = "method_key", values_to = "value") |>
  mutate(method_key = sub("_width$", "", method_key),
         method = recode(method_key,
           naive="Naive", debiased="Debiased",
           pct="Percentile", boot_t="Bootstrap-t"),
         method = factor(method, levels = names(METHOD_COLORS))) |>
  group_by(method) |>
  summarise(value = mean(value, na.rm = TRUE), .groups = "drop") |>
  mutate(metric = "CI Width")

overall_all <- bind_rows(overall_cover, overall_bias, overall_width) |>
  mutate(metric = factor(metric, levels = c("Coverage", "Bias", "CI Width")))

p_overall <- ggplot(overall_all, aes(x = method, y = value, fill = method)) +
  geom_col(width = 0.6, show.legend = FALSE) +
  geom_hline(data = data.frame(metric = factor("Coverage", levels = c("Coverage","Bias","CI Width")),
                                yint = NOMINAL),
             aes(yintercept = yint), linetype = "dashed",
             colour = "grey30", linewidth = 0.8) +
  geom_hline(data = data.frame(metric = factor("Bias", levels = c("Coverage","Bias","CI Width")),
                                yint = 0),
             aes(yintercept = yint), linetype = "dashed",
             colour = "grey30", linewidth = 0.8) +
  geom_text(aes(label = ifelse(metric == "Coverage",
                               percent(value, accuracy = 0.1),
                               round(value, 3))),
            vjust = ifelse(overall_all$value >= 0, -0.4, 1.2),
            size = 4.5, fontface = "bold") +
  scale_fill_manual(values = METHOD_COLORS) +
  facet_wrap(~metric, scales = "free_y", nrow = 1) +
  labs(x = NULL, y = NULL) +
  theme_slide() +
  theme(axis.text.x = element_text(size = 11, angle = 15, hjust = 1))

ggsave("figures/slide_overall.pdf", p_overall,
       width = 11, height = 5, device = cairo_pdf)
message("Saved figures/slide_overall.pdf")

# ── 2. Coverage by rho ───────────────────────────────────────
by_rho <- long |>
  group_by(method, rho) |>
  summarise(cover = mean(cover, na.rm = TRUE), .groups = "drop") |>
  mutate(rho = factor(rho))

p_rho <- ggplot(by_rho, aes(x = rho, y = cover,
                              colour = method, shape = method, group = method)) +
  hline +
  geom_line(linewidth = 1) +
  geom_point(size = 4) +
  scale_colour_manual(values = METHOD_COLORS) +
  scale_shape_manual(values  = METHOD_SHAPES) +
  pct_y +
  labs(x = expression(rho ~ "(predictor correlation)"), y = "Coverage") +
  theme_slide()

ggsave("figures/slide_rho.pdf", p_rho,
       width = 8, height = 5, device = cairo_pdf)
message("Saved figures/slide_rho.pdf")

# ── 3. Coverage by (n, p) ────────────────────────────────────
by_np <- long |>
  mutate(np = paste0("n=", n, "\np=", p)) |>
  group_by(method, np) |>
  summarise(cover = mean(cover, na.rm = TRUE), .groups = "drop") |>
  mutate(np = factor(np, levels = c("n=100\np=200","n=100\np=500",
                                     "n=200\np=200","n=200\np=500")))

p_np <- ggplot(by_np, aes(x = np, y = cover, fill = method)) +
  geom_col(position = "dodge", width = 0.7) +
  hline +
  scale_fill_manual(values = METHOD_COLORS) +
  pct_y +
  labs(x = NULL, y = "Coverage") +
  theme_slide()

ggsave("figures/slide_np.pdf", p_np,
       width = 9, height = 5, device = cairo_pdf)
message("Saved figures/slide_np.pdf")

# ── 4. Coverage by covariance structure ──────────────────────
by_struct <- long |>
  group_by(method, structure) |>
  summarise(cover = mean(cover, na.rm = TRUE), .groups = "drop") |>
  mutate(structure = fct_relevel(structure, "independent", "ar1", "block"))

p_struct <- ggplot(by_struct, aes(x = structure, y = cover, fill = method)) +
  geom_col(position = "dodge", width = 0.7) +
  hline +
  scale_fill_manual(values = METHOD_COLORS) +
  pct_y +
  labs(x = "Covariance structure", y = "Coverage") +
  theme_slide()

ggsave("figures/slide_structure.pdf", p_struct,
       width = 8, height = 5, device = cairo_pdf)
message("Saved figures/slide_structure.pdf")

# ── 5. Coverage by error distribution ────────────────────────
by_err <- long |>
  mutate(error_dist = recode(error_dist,
    normal = "Normal(0,1)", t3 = "t(3)/sqrt(3)")) |>
  group_by(method, error_dist) |>
  summarise(cover = mean(cover, na.rm = TRUE), .groups = "drop")

p_err <- ggplot(by_err, aes(x = error_dist, y = cover, fill = method)) +
  geom_col(position = "dodge", width = 0.6) +
  hline +
  scale_fill_manual(values = METHOD_COLORS) +
  pct_y +
  labs(x = "Error distribution", y = "Coverage") +
  theme_slide()

ggsave("figures/slide_error.pdf", p_err,
       width = 7, height = 5, device = cairo_pdf)
message("Saved figures/slide_error.pdf")

message("\nAll presentation figures saved to figures/")
