library(tidyverse)
library(tidymodels)
library(data.table)
library(jsonlite)
library(crayon)
library(ggplot2)
library(timetk)       
library(keras3)        
library(text2vec)      
library(textrecipes)   
library(yardstick)     
library(probably)      

source("save_results.r")

set.seed(42)
theme_set(theme_minimal())

fmt_time <- function(secs) format(.POSIXct(secs, tz = "UTC"), "%H:%M:%S")

cat("Choose a number for a specific task:\n",
    "1. Time series forecasting - Univariate.\n",
    "2. Time series forecasting - Multivariate.\n",
    "3. Text classification\n")
# choice <- as.integer(readline())
choice <- 1
train_losses <- c()
val_losses <- c()

if (choice %in% c(1, 2)) {

  if (choice == 1) {
    cat("Time series forecasting - Univariate.\n")
    output_folder <- "time_series/results/brnn/Univariate"
    csv_path <- "time_series/datasets/pressure_lags.csv"
  } else {
    cat("Time series forecasting - Multivariate.\n")
    output_folder <- "time_series/results/brnn/multivariate"
    csv_path <- "time_series/datasets/4features_lags.csv"
  }
  dir.create(output_folder, recursive = TRUE, showWarnings = FALSE)

  data <- fread(csv_path) %>%
    as_tibble() %>%
    mutate(`Date Time` = as.POSIXct(`Date Time`, format = "%Y-%m-%d %H:%M:%S")) %>%
    arrange(`Date Time`)

  TARGET <- "p (mbar)"
  FEATURES <- setdiff(names(data), c("Date Time", TARGET))

  split_point <- floor(nrow(data) * 0.8)

  train_val_df <- data[1:split_point, ]
  test_df <- data[(split_point + 1):nrow(data), ]

  # Inner val split (last 12.5 % of train_val)
  val_n <- floor(nrow(train_val_df) * 0.125)
  train_df <- train_val_df[1:(nrow(train_val_df) - val_n), ]
  val_df <- train_val_df[(nrow(train_val_df) - val_n + 1):nrow(train_val_df), ]

  rec <- readRDS(paste0("time_series/results/scaler_", choice, ".rds"))

  X_train <- bake(rec, train_df) %>% select(all_of(FEATURES)) %>% as.matrix()
  X_val <- bake(rec, val_df)   %>% select(all_of(FEATURES)) %>% as.matrix()
  X_test <- bake(rec, test_df)  %>% select(all_of(FEATURES)) %>% as.matrix()
  y_train <- train_df[[TARGET]]
  y_val <- val_df[[TARGET]]
  y_test <- test_df[[TARGET]]

  create_sequences <- function(X, y, seq_len = 24) {
    n <- nrow(X)
    idx <- seq_len(n - seq_len)
    Xs <- array(0, dim = c(length(idx), seq_len, ncol(X)))
    ys <- numeric(length(idx))
    for (i in idx) {
      Xs[i, , ] <- X[i:(i + seq_len - 1), ]
      ys[i] <- y[i + seq_len]
    }
    list(X = Xs, y = ys)
  }

  SEQ_LENGTH <- 24

  tr <- create_sequences(X_train, y_train, SEQ_LENGTH)
  vl <- create_sequences(X_val,   y_val,   SEQ_LENGTH)
  te <- create_sequences(X_test,  y_test,  SEQ_LENGTH)

  BATCH_SIZE <- 64L
  EPOCHS <- 50L
  HIDDEN_SIZE <- 64L
  NUM_LAYERS <- 2L
  DROPOUT <- 0.2
  LEARNING_RATE <- 0.001
  input_size <- dim(tr$X)[3]

  hyperparams <- list(
    epochs = EPOCHS,
    hidden_size = HIDDEN_SIZE,
    num_layers = NUM_LAYERS,
    dropout = DROPOUT,
    learning_rate = LEARNING_RATE
  )

  build_brnn_ts <- function() {
    inputs <- layer_input(shape = c(SEQ_LENGTH, input_size))

    x <- inputs

    # Stack bidirectional RNN layers
    for (l in seq_len(NUM_LAYERS)) {
      return_seq <- (l < NUM_LAYERS)
      x <- x %>%
        bidirectional(
          layer_simple_rnn(
            units = HIDDEN_SIZE,
            return_sequences = return_seq,
            dropout = DROPOUT,
            activation = "tanh"
          )
        )
    }

    outputs <- x %>% layer_dense(units = 1)
    keras_model(inputs, outputs, name = "BRNN_ts")
  }

  model <- build_brnn_ts()

  model %>% compile(
    optimizer = optimizer_adam(learning_rate = LEARNING_RATE),
    loss = "mse"
  )

  callbacks_list <- list(
    callback_early_stopping(
      monitor = "val_loss",
      patience = 5,
      restore_best_weights = TRUE,
      verbose = 1
    ),
    callback_reduce_lr_on_plateau(
      monitor = "val_loss",
      factor = 0.5,
      patience = 3,
      verbose = 1
    ),
    callback_model_checkpoint(
      filepath = file.path(output_folder, "brnn_best.keras"),
      monitor = "val_loss",
      save_best_only = TRUE,
      verbose = 1
    )
  )

  cat(blue("Training started...\n"))
  start_time <- proc.time()["elapsed"]

  history <- model %>% fit(
    x = tr$X,
    y = tr$y,
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    validation_data = list(vl$X, vl$y),
    callbacks = callbacks_list,
    verbose = 1
  )

  execution_time <- proc.time()["elapsed"] - start_time

  train_losses <- history$metrics$loss
  val_losses <- history$metrics$val_loss
  stop_epoch <- length(train_losses)

  write_json(
    list(`Train losses` = train_losses, `Val Losses` = val_losses),
    file.path(output_folder, "train_val_losses.json"),
    pretty = TRUE, auto_unbox = TRUE
  )

  y_pred <- as.vector(predict(model, te$X))
  y_true <- te$y

  results_df <- tibble(truth = y_true, estimate = y_pred)
  r2 <- rsq(results_df, truth, estimate)$.estimate
  mae <- mae(results_df, truth, estimate)$.estimate
  mse <- rmse(results_df, truth, estimate)$.estimate ^ 2
  rmse <- rmse(results_df, truth, estimate)$.estimate

  metrics <- list(`R2 score` = r2, MAE = mae, mse = mse, rmse = rmse)
  cat(sprintf("\nR²: %.4f | MAE: %.4f | MSE: %.4f | RMSE: %.4f\n",
              r2, mae, mse, rmse))

  n_show <- min(200, length(y_true))
  pred_df <- tibble(
    step = 1:n_show,
    Actual = y_true[1:n_show],
    Predicted = y_pred[1:n_show]
  ) %>% pivot_longer(-step, names_to = "Series", values_to = "Value")

  p_pred <- ggplot(pred_df, aes(step, Value, colour = Series)) +
    geom_line(alpha = 0.75) +
    scale_colour_manual(values = c(Actual = "steelblue", Predicted = "tomato")) +
    labs(title = sprintf("BRNN — Predicted vs Actual\nR² = %.4f, RMSE = %.4f", r2, rmse),
         x = "Time steps", y = "Pressure (mbar)") +
    theme(legend.position = "bottom",
          axis.text.x = element_text(angle = 0, hjust = 0.5),
          plot.title = element_text(hjust = 0.5)
        )
  ggsave(file.path(output_folder, "predicted_vs_real.png"),
         p_pred, width = 12, height = 6, dpi = 300)

  loss_df <- tibble(
    epoch = rep(seq_len(stop_epoch), 2),
    loss = c(train_losses, val_losses),
    split = rep(c("Train", "Validation"), each = stop_epoch)
  )
} else {
  cat("==========Text classification==========\n")
  output_folder <- "text/results/brnn"
  dir.create(output_folder, recursive = TRUE, showWarnings = FALSE)

  data <- fread("text/AG News/cleaned_data.csv") %>% as_tibble()

  class_mapping <- data %>%
    distinct(`Class Index`, category) %>%
    arrange(`Class Index`) %>%
    deframe()                        

  NUM_CLASSES <- length(class_mapping)

  set.seed(42)
  split1 <- initial_split(data, prop = 0.8, strata = `Class Index`)
  train_val_df <- training(split1)
  test_df <- testing(split1)

  split2 <- initial_split(train_val_df, prop = 0.9, strata = `Class Index`)
  train_df <- training(split2)
  val_df <- testing(split2)

  MAX_VOCAB_SIZE <- 25000L
  MAX_SEQ_LENGTH <- 100L
  EMBEDDING_DIM <- 100L
  BATCH_SIZE <- 128L
  HIDDEN_SIZE <- 128L
  NUM_LAYERS <- 2L
  DROPOUT <- 0.5
  LEARNING_RATE <- 0.001
  EPOCHS <- 30L

  # ── text2vec: vocabulary from training set only ────────────────────────────
  it_train <- itoken(train_df$text, tokenizer = word_tokenizer, progressbar = FALSE)

  vocab <- create_vocabulary(it_train) %>%
    prune_vocabulary(
      term_count_min = 1L,
      vocab_term_max = MAX_VOCAB_SIZE - 2L 
    )

  vectorizer <- vocab_vectorizer(vocab)
  vocab_size <- nrow(vocab) + 2L              # +2 for PAD/UNK

  
  term_to_idx <- setNames(
    seq_len(nrow(vocab)) + 1L,  
    vocab$term
  )

  # ── text2vec tokenize + pad/truncate ──────────────────────────────────────
  texts_to_matrix <- function(texts, term_idx, max_len) {
    tokens <- word_tokenizer(texts)
    t(sapply(tokens, function(toks) {
      toks <- toks[seq_len(min(length(toks), max_len))]
      idx <- term_idx[toks]
      idx[is.na(idx)] <- 1L                   # UNK
      length(idx) <- max_len                  # pad with NA → 0
      idx[is.na(idx)] <- 0L                   # PAD
      as.integer(idx)
    }))
  }

  X_train_mat <- texts_to_matrix(train_df$text, term_to_idx, MAX_SEQ_LENGTH)
  X_val_mat <- texts_to_matrix(val_df$text, term_to_idx, MAX_SEQ_LENGTH)
  X_test_mat <- texts_to_matrix(test_df$text, term_to_idx, MAX_SEQ_LENGTH)

  # ── Label encoding (0-based for Keras sparse_categorical_crossentropy) ─────
  classes <- sort(unique(data$`Class Index`))
  encode_lbl <- function(v) match(v, classes) - 1L

  y_train <- encode_lbl(train_df$`Class Index`)
  y_val <- encode_lbl(val_df$`Class Index`)
  y_test <- encode_lbl(test_df$`Class Index`)

  model <- keras_model_sequential(name = "brnn_text") %>%
    layer_embedding(
      input_dim = vocab_size,
      output_dim = EMBEDDING_DIM,
      mask_zero = TRUE,           # honours PAD=0, replaces pack_padded_sequence
      name = "embedding"
    ) %>%
    layer_simple_brnn(
      units = HIDDEN_SIZE,
      return_sequences = (NUM_LAYERS > 1),
      dropout = DROPOUT,
      activation = "tanh",
      name = "brnn_1"
    )

  if (NUM_LAYERS > 1) {
    for (l in seq_len(NUM_LAYERS - 2)) {
      model <- model %>%
        layer_simple_brnn(units = HIDDEN_SIZE, return_sequences = TRUE,
                         dropout = DROPOUT, activation = "tanh")
    }
    model <- model %>%
      layer_simple_brnn(units = HIDDEN_SIZE, dropout = DROPOUT, activation = "tanh")
  }

  model <- model %>%
    layer_dropout(DROPOUT) %>%
    layer_dense(units = NUM_CLASSES, activation = "softmax")

  model %>% compile(
    optimizer = optimizer_adam(learning_rate = LEARNING_RATE),
    loss = "sparse_categorical_crossentropy",
    metrics = c("accuracy")
  )

  summary(model)

  # ── Callbacks ──────────────────────────────────────────────────────────────
  callbacks_list <- list(
    callback_early_stopping(
      monitor = "val_loss",
      patience = 15,
      restore_best_weights = TRUE,
      verbose = 1
    ),
    callback_reduce_lr_on_plateau(
      monitor = "val_loss",
      factor = 0.5,
      patience = 3,
      verbose = 1
    ),
    callback_model_checkpoint(
      filepath = file.path(output_folder, "best_model.keras"),
      monitor = "val_accuracy",
      save_best_only = TRUE,
      verbose = 1
    )
  )

  cat(blue("\nTraining started...\n"))
  start_time <- proc.time()["elapsed"]

  history <- model %>% fit(
    x = X_train_mat,
    y = y_train,
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    validation_data = list(X_val_mat, y_val),
    callbacks = callbacks_list,
    verbose = 1
  )

  execution_time <- proc.time()["elapsed"] - start_time

  train_losses <- history$metrics$loss
  val_losses <- history$metrics$val_loss
  stop_epoch <- length(train_losses)

  write_json(
    list(`Train losses` = train_losses,
         `Val Losses` = val_losses,
         `stop epoch` = stop_epoch,
         `execution time` = execution_time),
    file.path(output_folder, "train_val_losses.json"),
    pretty = TRUE, auto_unbox = TRUE
  )

  # Evaluation 
  probs_mat  <- predict(model, X_test_mat)                
  preds <- max.col(probs_mat) - 1L               
  labels_fct <- factor(unname(class_mapping)[y_test + 1L], levels = unname(class_mapping))
  preds_fct  <- factor(unname(class_mapping)[preds  + 1L], levels = unname(class_mapping))

  # yardstick metrics
  eval_df <- tibble(truth = labels_fct, estimate = preds_fct) %>%
    bind_cols(as_tibble(probs_mat, .name_repair = ~ unname(class_mapping)))

  accuracy  <- accuracy(eval_df, truth, estimate)$.estimate
  precision <- precision(eval_df, truth, estimate, estimator = "macro")$.estimate
  recall <- recall(eval_df, truth, estimate, estimator = "macro")$.estimate
  f1 <- f_meas(eval_df, truth, estimate, estimator = "macro")$.estimate

  # Multiclass ROC-AUC (one-vs-rest, macro — equivalent to sklearn weighted)
  roc_auc <- roc_auc(eval_df, truth, all_of(unname(class_mapping)),
                     estimator = "macro_weighted")$.estimate

  # Per-class report
  report <- per_class_prec_recall_f1(eval_df$truth, eval_df$estimate, class_mapping)

  metrics <- list(
    accuracy = accuracy,
    precision = precision,
    recall = recall,
    f1 = f1,
    roc_auc_weighted = roc_auc,
    classification_report = report
  )

  cat(sprintf("Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f | AUC: %.4f\n",
              accuracy, precision, recall, f1, roc_auc))

  # Confusion matrix 
  cm_df <- conf_mat(eval_df, truth, estimate)$table %>%
    as_tibble() %>%
    rename(Predicted = Prediction, Actual = Truth)

  p_cm <- ggplot(cm_df, aes(Actual, Predicted, fill = n)) +
    geom_tile(colour = "white") +
    geom_text(aes(label = n), colour = "white", size = 4, fontface = "bold") +
    scale_fill_gradient(low = "#dce9f5", high = "#1c6ba0") +
    labs(title = "Confusion matrix — test set") +
    theme(
      axis.text.x = element_text(angle = 0, hjust = 0.5),
      legend.position = "none",
      plot.title = element_text(hjust = 0.5),
    )

  ggsave(file.path(output_folder, "confusion_matrix.png"),
         p_cm, width = 10, height = 8, dpi = 300)

  hyperparams <- list(
    vocab_size = vocab_size,
    max_sequence_length = MAX_SEQ_LENGTH,
    embedding_dim = EMBEDDING_DIM,
    batch_size = BATCH_SIZE,
    hidden_size = HIDDEN_SIZE,
    num_layers = NUM_LAYERS,
    dropout = DROPOUT,
    learning_rate = LEARNING_RATE,
    epochs = EPOCHS
  )

  # Learning curve
  loss_df <- tibble(
    epoch = rep(seq_len(stop_epoch), 2),
    loss = c(train_losses, val_losses),
    split = rep(c("Train", "Validation"), each = stop_epoch)
  )
}
p_lc <- ggplot(loss_df, aes(epoch, loss, colour = split)) +
  geom_line(linewidth = 0.8) +
  scale_colour_manual(values = c(Train = "steelblue", Validation = "darkorange")) +
  facet_wrap(~split, scales = "free_y") +
  labs(title = "brnn — learning curve", x = "Epoch", y = "Loss") +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 0, hjust = 0.5),
    plot.title = element_text(hjust = 0.5)
  )
ggsave(file.path(output_folder, "learning_curve.png"),
        p_lc, width = 12, height = 5, dpi = 300)

save_to_file(
  train_losses, val_losses, execution_time,
  BATCH_SIZE, hyperparams, stop_epoch, metrics,
  file_path = file.path(output_folder, "full_results.json")
)