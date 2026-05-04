library(dplyr)
library(ggplot2)

rcand_vals <- c(2, 6, 10, 20, 40, 100)

grid <- expand.grid(
  rcand = rcand_vals,
  intersection_ratio = seq(0.1, 1, 0.05),
  same_stance_ratio = seq(0.1, 1, 0.05)
)

grid <- grid %>%
  rowwise() %>%
  mutate(
    K = round(rcand * intersection_ratio),
    M = round(K * same_stance_ratio),
    EAA = ifelse(K > 0, M / K, 0)
  ) %>%
  ungroup()

ggplot(grid, aes(
  x = intersection_ratio,
  y = same_stance_ratio,
  fill = EAA
)) +
  geom_tile() +
  facet_wrap(~ rcand, ncol = 3) +
  scale_fill_gradientn(
    colors = colorRampPalette(c("blue", "white", "red"))(100),
    limits = c(0, 1)
  ) +
  labs(
    title = "EAA conditioned on number of candidate solutions",
    x = "Intersection ratio (coverage)",
    y = "Same stance ratio (agreement)",
    fill = "EAA"
  ) +
  theme_minimal()