library(torch)
library(tidyverse)
library(ggplot2)
library(data.table)
library(crayon)
library(jsonlite)
library(keras)

early_stop <- callback_early_stopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights  = True
)


source('save_results.r')

set.seed(42)
torch_manual_seed(42)
theme_set(theme_minimal())

# Helper function to create sequences
create_sequences <- function(X, y, seq_length = 24) {
  n <- nrow(X)
  n_features <- ncol(X)
  X_seq <- array(0, dim = c(n - seq_length, seq_length, n_features))
  y_seq <- numeric(n - seq_length)
  
  for (i in 1:(n - seq_length)) {
    X_seq[i, , ] <- X[i:(i + seq_length - 1), , drop = FALSE]
    y_seq[i] <- y[i + seq_length]
  }
  
  list(X = X_seq, y = y_seq)
}

DEVICE <- if (cuda_is_available()) "cuda" else "cpu"

cat("Choose a number for a specific dataset:\n1. Time series forecasting - Univariate.\n2. Time series forecasting - Multivariate.\n3. Text classification\n")
choice <- as.integer(readline())

if (choice == 1 || choice == 2) {
  output_folder <- "time_series/"
  
  if (choice == 1) {
    dir.create("time_series/results/brnn/Univariate", recursive = TRUE, showWarnings = FALSE)
    output_folder <- "time_series/results/brnn/Univariate"
    data <- fread("time_series/datasets/pressure_lags.csv")
    data <- data %>% 
      mutate(`Date Time` = as.POSIXct(`Date Time`, format = "%Y-%m-%d %H:%M:%S")) %>%
      column_to_rownames("Date Time")
  } else if (choice == 2) {
    dir.create("time_series/results/brnn/multivariate", recursive = TRUE, showWarnings = FALSE)
    output_folder <- "time_series/results/brnn/multivariate"
    data <- fread("time_series/datasets/4features_lags.csv")
    data <- data %>% 
      mutate(`Date Time` = as.POSIXct(`Date Time`, format = "%Y-%m-%d %H:%M:%S")) %>%
      column_to_rownames("Date Time")
  }
  
  TARGET <- 'p (mbar)'
  FEATURES <- setdiff(names(data), TARGET)
  
  X <- as.matrix(data[, FEATURES, with = FALSE])
  y <- data[[TARGET]]
  
  split_point <- floor(length(y) * 0.8)
  X_train_val <- X[1:split_point, , drop = FALSE]
  X_test <- X[(split_point + 1):nrow(X), , drop = FALSE]
  y_train_val <- y[1:split_point]
  y_test <- y[(split_point + 1):length(y)]
  
  val_split <- floor(length(y_train_val) * 0.125)
  X_train <- X_train_val[1:(nrow(X_train_val) - val_split), , drop = FALSE]
  X_val <- X_train_val[(nrow(X_train_val) - val_split + 1):nrow(X_train_val), , drop = FALSE]
  y_train <- y_train_val[1:(length(y_train_val) - val_split)]
  y_val <- y_train_val[(length(y_train_val) - val_split + 1):length(y_train_val)]
  
  # Standard scaling (equivalent to sklearn StandardScaler)
  scale_mean <- colMeans(X_train)
  scale_sd <- apply(X_train, 2, sd)
  
  X_train <- scale(X_train, center = scale_mean, scale = scale_sd)
  X_val <- scale(X_val, center = scale_mean, scale = scale_sd)
  X_test <- scale(X_test, center = scale_mean, scale = scale_sd)
  
  # Save scaler (equivalent to pickle)
  saveRDS(list(mean = scale_mean, sd = scale_sd), 
          file = paste0("time_series/results/scaler_", choice, ".rds"))
  
  SEQ_LENGTH <- 24
  
  train_seq <- create_sequences(X_train, y_train, SEQ_LENGTH)
  val_seq <- create_sequences(X_val, y_val, SEQ_LENGTH)
  test_seq <- create_sequences(X_test, y_test, SEQ_LENGTH)
  
  # Convert to tensors
  X_tr <- torch_tensor(train_seq$X, dtype = torch_float())
  y_tr <- torch_tensor(matrix(train_seq$y, ncol = 1), dtype = torch_float())
  X_v <- torch_tensor(val_seq$X, dtype = torch_float())
  y_v <- torch_tensor(matrix(val_seq$y, ncol = 1), dtype = torch_float())
  X_te <- torch_tensor(test_seq$X, dtype = torch_float())
  y_te <- torch_tensor(matrix(test_seq$y, ncol = 1), dtype = torch_float())
  
  BATCH_SIZE <- 64
  EPOCHS <- 50
  input_size <- dim(train_seq$X)[3]
  hidden_size <- 64
  num_layers <- 2
  dropout <- 0.2
  learning_rate <- 0.001
  
  # Create dataloaders
  train_ds <- tensor_dataset(X_tr, y_tr)
  train_loader <- dataloader(train_ds, batch_size = BATCH_SIZE, shuffle = FALSE)
  
  val_ds <- tensor_dataset(X_v, y_v)
  val_loader <- dataloader(val_ds, batch_size = BATCH_SIZE, shuffle = FALSE)
  
  # Define BRNN model
  brnn <- nn_rnn(
    input_size = input_size,
    hidden_size = hidden_size,
    num_layers = num_layers,
    batch_first = TRUE,
    dropout = if (num_layers > 1) dropout else 0,
    nonlinearity = "tanh",
    bidirectional = TRUE
  )$to(DEVICE)
  
  fc <- nn_linear(hidden_size * 2, 1)$to(DEVICE)
  
  params <- c(brnn$parameters, fc$parameters)
  criterion <- nn_mse_loss()
  optimizer <- optim_adam(params, lr = learning_rate)
  
  # Forward function
  forward <- function(x) {
    out <- brnn(x)
    brnn_out <- out[[1]]
    seq_len <- brnn_out$size()[[2]]  # Get last time step
    last_step <- brnn_out[, seq_len, , drop = FALSE]
    fc(last_step)
  }
  
  train_losses <- c()
  val_losses <- c()
  times_per_epoch <- c()
  early_stopping <- EarlyStopping()
  stop_epoch <- EPOCHS - 1
  saved_checkpoint <- FALSE
  
  cat(blue("Training started...\n"))
  start_time <- Sys.time()
  
  for (epoch in 1:EPOCHS) {
    start_epoch <- Sys.time()
    
    # Training
    brnn$train()
    fc$train()
    total_train_loss <- 0
    
    for (batch in train_loader) {
      bx <- batch[[1]]$to(DEVICE)
      by <- batch[[2]]$to(DEVICE)
      
      optimizer$zero_grad()
      output <- forward(bx)
      loss <- criterion(output, by)
      loss$backward()
      optimizer$step()
      total_train_loss <- total_train_loss + loss$item()
    }
    
    avg_train_loss <- total_train_loss / length(train_loader)
    train_losses <- c(train_losses, avg_train_loss)
    
    # Validation
    brnn$eval()
    fc$eval()
    total_val_loss <- 0
    
    with_no_grad({
      for (batch in val_loader) {
        batch_x <- batch[[1]]$to(DEVICE)
        batch_y <- batch[[2]]$to(DEVICE)
        output <- forward(batch_x)
        loss <- criterion(output, batch_y)
        total_val_loss <- total_val_loss + loss$item()
      }
    })
    
    avg_val_loss <- total_val_loss / length(val_loader)
    val_losses <- c(val_losses, avg_val_loss)
    
    end_epoch <- Sys.time()
    epoch_time <- as.numeric(difftime(end_epoch, start_epoch, units = "secs"))
    times_per_epoch <- c(times_per_epoch, format(.POSIXct(epoch_time, origin = "1970-01-01"), "%H:%M:%S"))
    
    if (early_stopping$check(avg_val_loss, brnn)) {
      brnn <- early_stopping$load_best_model(brnn)
      torch_save(list(brnn = brnn$state_dict(), fc = fc$state_dict()),
                 paste0(output_folder, "/brnn_model_checkpoint.pth"))
      saved_checkpoint <- TRUE
      cat("Early stopping\n")
      stop_epoch <- epoch
      break
    }
    
    cat(sprintf("Epoch [%d/%d], Train Loss: %.4f, Val Loss: %.4f\n",
                epoch, EPOCHS, avg_train_loss, avg_val_loss))
  }
  
  execution_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
  
  if (!saved_checkpoint) {
    torch_save(list(brnn = brnn$state_dict(), fc = fc$state_dict()),
               paste0(output_folder, "/brnn_model.pth"))
  }
  
  # Save losses to JSON
  write_json(list(`Train losses` = train_losses, `Val Losses` = val_losses),
             paste0(output_folder, "/train_val_losses.json"),
             pretty = TRUE, auto_unbox = TRUE)
  
  # Plot learning curve
  epochs_range <- 1:stop_epoch
  
  df_loss <- data.frame(
    Epoch = rep(epochs_range, 2),
    Loss = c(train_losses[1:stop_epoch], val_losses[1:stop_epoch]),
    Type = rep(c("Train Loss", "Val Loss"), each = stop_epoch)
  )
  
  p <- ggplot(df_loss, aes(x = Epoch, y = Loss, color = Type)) +
    geom_line(linewidth = 0.8) +
    scale_color_manual(values = c("Train Loss" = "blue", "Val Loss" = "orange")) +
    labs(title = "BRNN Learning Curve", x = "Epoch", y = "Loss (MSE)") +
    theme(legend.position = "bottom")
  
  ggsave(paste0(output_folder, "/learning_curve.png"), p, width = 12, height = 5, dpi = 300)
  
  # Evaluation
  brnn$eval()
  fc$eval()
  
  with_no_grad({
    y_pred <- as.matrix(forward(X_te$to(DEVICE))$cpu())
  })
  
  y_true <- matrix(test_seq$y, ncol = 1)
  
  # Calculate metrics (equivalent to sklearn metrics)
  ss_res <- sum((y_true - y_pred)^2)
  ss_tot <- sum((y_true - mean(y_true))^2)
  r2 <- 1 - ss_res / ss_tot
  mae <- mean(abs(y_true - y_pred))
  mse <- mean((y_true - y_pred)^2)
  rmse <- sqrt(mse)
  
  # Save full results
  save_to_file(
    train_losses, val_losses, execution_time, times_per_epoch, BATCH_SIZE,
    list(epochs = EPOCHS, hidden_size = hidden_size, num_layers = num_layers,
         dropout = dropout, learning_rate = learning_rate),
    stop_epoch,
    list(`R2 score` = r2, MAE = mae, mse = mse, rmse = rmse),
    file_path = paste0(output_folder, "/full_results.json")
  )
  
  # Plot predictions vs actual
  n_points <- min(200, length(y_true))
  df_pred <- data.frame(
    TimeSteps = 1:n_points,
    Actual = y_true[1:n_points, 1],
    Predicted = y_pred[1:n_points, 1]
  )
  
  p2 <- ggplot(df_pred, aes(x = TimeSteps)) +
    geom_line(aes(y = Actual, color = "Actual Pressure"), alpha = 0.7, linewidth = 0.8) +
    geom_line(aes(y = Predicted, color = "Predicted Pressure"), alpha = 0.7, linewidth = 0.8) +
    scale_color_manual(values = c("Actual Pressure" = "blue", "Predicted Pressure" = "red")) +
    labs(title = sprintf("BRNN Predicted vs Actual Values\nR² = %.4f, RMSE = %.4f", r2, rmse),
         x = "Time Steps", y = "Pressure (mbar)") +
    theme(legend.position = "bottom")
  
  ggsave(paste0(output_folder, "/predicted_vs_real.png"), p2, width = 12, height = 6, dpi = 300)
  
  cat(sprintf("\nFinal Metrics:\nR²: %.4f\nMAE: %.4f\nMSE: %.4f\nRMSE: %.4f\n", r2, mae, mse, rmse))
}