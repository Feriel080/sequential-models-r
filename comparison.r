library(tidyverse)
library(jsonlite)
library(ggplot2)

output = "text/results/comparison"
dir.create(output, recursive = TRUE, showWarnings = FALSE)

# text classification
models_results <- list(
  BRNN = fromJSON("text/results/brnn/full_results.json", simplifyVector = FALSE),
  GRU = fromJSON("text/results/gru/full_results.json", simplifyVector = FALSE),
  LSTM = fromJSON("text/results/lstm/full_results.json", simplifyVector = FALSE),
  RNN = fromJSON("text/results/rnn/full_results.json", simplifyVector = FALSE)
  # Transformer = fromJSON("time_series/results/transformer/multivariate/full_results.json", simplifyVector = FALSE)
)

models_results <- map(models_results, ~ .x[[length(.x)]])

models <- names(models_results)
colors <- c("#DC143C", "#DA70D6", "#9400d3", "#4169e1") #"#008b8b")
names(colors) <- models


training_times <- map_dbl(models_results, "execution_time_seconds")
training_formatted <- map_chr(models_results, "execution_time_formatted")
best_epochs <- map_dbl(models_results, "best_epoch")
stop_epochs <- map_dbl(models_results, "Stopping epoch")
val_losses <- map_dbl(models_results, "best_validation_loss")
# r2_scores <- map_dbl(models_results, ~ .x[["test_metrics"]][["R2 score"]])

# training time
df_time <- tibble(
  model = models,
  secs  = training_times,
  label = training_formatted
)

p_time <- ggplot(df_time, aes(model, secs, fill = model)) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(y = secs + max(secs) * 0.01, label = label),
            vjust = 0, size = 3.5) +
  scale_fill_manual(values = colors) +
  labs(title = "Training time", y = "Time (s)", x = NULL) +
  theme(
    legend.position = "none",
    plot.title = element_text(hjust = 0.5)) +
  theme_minimal()

ggsave(file.path(output, "Training_time.png"), p_time, width = 10, height = 4, dpi = 300)

# best epoc vs best stop epoch
df_epochs <- tibble(
  model = rep(models, 2),
  epoch = c(best_epochs, stop_epochs),
  type = rep(c("best epoch", "stop epoch"), each = length(models))
)

p_epochs <- ggplot(df_epochs, aes(model, epoch, fill = type)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.6) +
    geom_text(aes(y = epoch + max(epoch) * 0.01, label = epoch),
                position = position_dodge(width = 0.7),
                vjust = 0, size = 3.5) +
    scale_fill_manual(values = c("best epoch" = "#DC143C", "stop epoch" = "#DA70D6")) +
    labs(title = "Best epoch and stop epoch for each model",
        y = "Epoch", x = NULL, fill = NULL) +
    theme(plot.title = element_text(hjust = 0.5)) + 
    theme_minimal()

ggsave(file.path(output, "Best_epoch.png"), p_epochs, width = 10, height = 4, dpi = 300)

# Best validation loss 
df_valloss <- tibble(model = models, loss = val_losses)

p_valloss <- ggplot(df_valloss, aes(model, loss, fill = model)) +
    geom_col(show.legend = FALSE) +
    geom_text(aes(y = loss + max(loss) * 0.01, label = sprintf("%.4f", loss)), vjust = 0, size = 3.5) +
    scale_fill_manual(values = colors) +
    labs(title = "Best validation loss for each model", y = "Best val loss", x = NULL) + 
    theme(
        plot.title = element_text(hjust = 0.5)
    ) +
    theme_minimal()

ggsave(file.path(output, "Best_val_loss.png"), p_valloss, width = 10, height = 4, dpi = 300)

# R2 score comparison
# df_r2_scores <- tibble(model = models, score = r2_scores)

# p_valloss <- ggplot(df_r2_scores, aes(model, score, fill = model)) +
#     geom_col(show.legend = FALSE) +
#     geom_text(aes(y = score + max(score) * 0.01, label = sprintf("%.4f", score)), vjust = 0, size = 3.5) +
#     scale_fill_manual(values = colors) +
#     labs(title = "R2 Score for each model", y = "R2 score", x = NULL) + 
#     theme_minimal() +
#     theme(plot.title = element_text(hjust = 0.5)) 
    
# ggsave(file.path(output, "R2_scores.png"), p_valloss, width = 10, height = 4, dpi = 300)

# All metrics comparison (text classification) 
metric_names <- c("accuracy", "precision", "recall", "f1")

df_metrics <- map_dfr(models, function(m) {
  tm <- models_results[[m]][["test_metrics"]]
  tibble(
    model  = m,
    metric = metric_names,
    value  = map_dbl(metric_names, ~ tm[[.x]])
  )
})

p_metrics <- ggplot(df_metrics, aes(metric, value, fill = model)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  geom_text(aes(y = value + 0.003, label = sprintf("%.3f", value)),
            position = position_dodge(width = 0.8),
            vjust = 0, size = 2.5) +
  scale_fill_manual(values = colors) +
  scale_y_continuous(limits = c(0, 1.12)) +
  labs(title = "Model performance comparison",
       y = "Score", x = NULL, fill = NULL)

ggsave(file.path(output, "metrics_comparison.png"),
       p_metrics, width = 14, height = 5, dpi = 300)

# Per-class F1-score 
df_f1 <- map_dfr(models, function(m) {
  report <- models_results[[m]][["test_metrics"]][["classification_report"]]
  map_dfr(names(report), function(cls) {
    tibble(model = m, class = cls, f1 = report[[cls]][["f1_score"]])
  })
})

all_classes <- unique(df_f1$class)

p_f1 <- ggplot(df_f1, aes(class, f1, fill = model)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  geom_text(aes(y = f1 + 0.003, label = sprintf("%.2f", f1)),
            position = position_dodge(width = 0.8),
            vjust = 0, size = 2.5) +
  scale_fill_manual(values = colors) +
  scale_y_continuous(limits = c(0, 1.12)) +
  labs(title = "F1-score per class comparison",
       x = "Class", y = "F1-score", fill = NULL)

ggsave(file.path(output, "per_class_f1-score_comparison.png"),
       p_f1, width = 12, height = 6, dpi = 300)