save_to_file <- function(train_losses, val_losses, execution_time, times_per_epoch, batch_size,
                        model_params, stop_epoch, metrics,
                        file_path="results/full_results.json") {
    best_train_loss <- if (length(train_losses) > 0) min(train_losses) else NULL
    best_val_loss = if (length(val_losses) > 0) min(val_losses) else NULL
    best_epoch = if (length(val_losses) > 0) which.min(val_losses) else NULL

    exec_formatted <- format(as.POSIXct(execution_time, origin = "1970-01-01", tz = "UTC"), "%H:%M:%S")

    results <- list(
        "execution_time_seconds" = round(execution_time, 2),
        "execution_time_formatted" = exec_formatted,
        "Batch_Size" = batch_size,
        "model_parameters" = model_params,
        "best_training_loss" = if (!is.null(best_train_loss)) round(best_train_loss, 6) else NULL,
        "best_validation_loss" = if (!is.null(best_val_loss)) round(best_val_loss, 6) else NULL,
        "best_epoch" = best_epoch,
        "Stopping epoch" = stop_epoch,
        "final_training_loss" = if (length(val_losses) > 0) round(tail(train_losses, 1), 6) else NULL,
        "final_validation_loss" = if (length(val_losses) > 0) round(tail(val_losses, 1), 6) else NULL,
        "test_metrics" = metrics,
        "loss_history" = list(
            "train" = round(train_losses, 6),
            "validation"= round(val_losses, 6),
            "time per epoch" = times_per_epoch
        )
    )

    data <- list()

    if (file.exists(file_path)) {
        content <- paste(readLine(file_path, warn = FALSE), collapse = "")
        content <- trimws(content)

        if (nchar(content) > 0){
            tryCatch({
                parsed <- jsonlite::fromJSON(content, simplifyVector = FALSE) # json array becomes an unnamed list, while a json object becomes a named list

                if (is.list(parsed) && !is.null(names(parsed))){
                    data <- list(parsed)
                } else if (is.list(parsed)){
                    data <- parsed
                }
            }, error = function(e) {
                data <- list()
            })
        } 
    }

    data[[length(data) + 1]] <- results

    jsonlite::write_json(data, file_path, pretty = TRUE, auto_unbox = TRUE)

    cat("Results have been saved.")
}
