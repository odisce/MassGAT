#!/usr/bin/env Rscript
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
# BiocManager::install("cliqueMS")

library(xcms)
library(cliqueMS)
library(tools)
library(MSnbase)


CONFIG <- list(
    input_dir  = "C:/Users/PP281296/CliqueMS/cliquems_mzml_c18",
    output_dir = "C:/Users/PP281296/CliqueMS/results",
    
    xcms_ppm       = 15,          
    xcms_peakwidth = c(5, 20),
    xcms_snthresh  = 10,          
    
    xcms_integrate = 1,
    
    cliquems_polarity = "positive", 
    cliquems_ppm      = 10          
)


process_file <- function(file_path, output_folder) {
    base_path <- sub("\\.(mzML|mzXML)$", "", basename(file_path), ignore.case = TRUE)
    peak_output <- file.path(output_folder, paste0(base_path, "_cliquems_results.csv"))
    
    if (file.exists(peak_output)) {
        cat("Output file already exists for", base_path, "- skipping\n")
        return(list(status = "skipped", file = base_path))
    }
    
    cat("Processing file:", file_path, "\n")
    time1 <- Sys.time()
    
    tryCatch({
        raw_data <- readMSData(file_path, mode = "onDisk")
                
        cwp <- CentWaveParam(
            ppm       = CONFIG$xcms_ppm,
            peakwidth = CONFIG$xcms_peakwidth,
            snthresh  = CONFIG$xcms_snthresh
        )
        
        cat(" -> Detecting chromatographic peaks...\n")
        peaks <- findChromPeaks(raw_data, param = cwp)
        
        cat(" -> Purging corrupted peaks to prevent NA Matrix...\n")
        pks <- chromPeaks(peaks)
        
        has_na <- apply(pks, 1, function(row) any(is.na(row)))
        has_zero_intensity <- pks[, "maxo"] <= 0 | pks[, "into"] <= 0
        is_flat <- (pks[, "rtmax"] - pks[, "rtmin"]) <= 0
        invalid_idx <- has_na | has_zero_intensity | is_flat
        invalid_idx[is.na(invalid_idx)] <- TRUE 
        
        if (any(invalid_idx)) {
            cat("    Removed", sum(invalid_idx), "invalid peaks.\n")
            chromPeaks(peaks) <- pks[!invalid_idx, , drop = FALSE]
        }
        
        cat(" -> Computing cliques...\n")
        ancliques <- getCliques(peaks, filter = TRUE)
        
        cat(" -> Finding isotopes...\n")
        ancliques <- getIsotopes(ancliques, maxGrade = 3, maxCharge = 3, ppm = CONFIG$cliquems_ppm)
        
        # Explicitly pull the datasets from the cliqueMS package into the script's memory
        data("positive.adinfo", package = "cliqueMS", envir = environment())
        data("negative.adinfo", package = "cliqueMS", envir = environment())
        
        # Assign the correct rules based on the polarity config
        if (CONFIG$cliquems_polarity == "positive") {
            adduct_rules <- positive.adinfo
        } else {
            adduct_rules <- negative.adinfo
        }
        #cat('\n \n Using adduct rules from the CSV file:\n')
        adduct_rules <- read.csv("/workspace/chroms_gnn/massgat_adducts_cliquems.csv")

        
        ancliques <- getAnnotation(
            ancliques, 
            polarity = CONFIG$cliquems_polarity, 
            ppm = CONFIG$cliquems_ppm, 
            adinfo = adduct_rules
        )
        
        cat(" -> Extracting and saving results...\n")
        if (isS4(ancliques)) {
            results_df <- ancliques@peaklist
        } else {
            results_df <- ancliques$peaklist
        }
        write.csv(results_df, peak_output, row.names = FALSE)
        
        time2 <- Sys.time()
        processing_time <- difftime(time2, time1, units = "mins")
        
        cat("Successfully processed", base_path, "in", round(processing_time, 2), "minutes\n")
        
        rm(raw_data, peaks, ancliques, results_df)
        gc()
        
        return(list(status = "success", file = base_path, processing_time = processing_time, peak_output = peak_output))
        
    }, error = function(e) {
        cat("ERROR processing", base_path, ":", conditionMessage(e), "\n")
        return(list(status = "error", file = base_path, error = conditionMessage(e)))
    })
}


process_all_files <- function(input_folder, output_folder) {
    if (!dir.exists(input_folder)) stop("Input folder does not exist: ", input_folder)
    if (!dir.exists(output_folder)) dir.create(output_folder, recursive = TRUE)
    
    ms_files <- list.files(input_folder, pattern = "\\.(mzML|mzXML)$", full.names = TRUE, ignore.case = TRUE)
    if (length(ms_files) == 0) stop("No mzML or mzXML files found in input folder.")
    
    cat("Found", length(ms_files), "MS files to process\n")
    
    results <- list()
    start_time <- Sys.time()
    
    for (i in seq_along(ms_files)) {
        cat("\n--- Processing file", i, "of", length(ms_files), "---\n")
        results[[i]] <- process_file(ms_files[i], output_folder)
    }
    
    end_time <- Sys.time()
    total_time <- difftime(end_time, start_time, units = "mins")
    
    success_count <- sum(sapply(results, function(x) x$status == "success"))
    error_count <- sum(sapply(results, function(x) x$status == "error"))
    skipped_count <- sum(sapply(results, function(x) x$status == "skipped"))
    
    cat("\n=== PROCESSING SUMMARY ===\n")
    cat("Total files:", length(ms_files), "\nSuccessful:", success_count, "\nErrors:", error_count, "\nSkipped:", skipped_count, "\nTotal time:", round(total_time, 2), "mins\n")
    
    return(results)
}

main <- function() {
    if (file.exists("/workspace") && file.exists("/data")) {
        cat("Running in container environment\n")
        input_folder <- "/data"
        output_folder <- "/output"
    } else {
        input_folder <- CONFIG$input_dir
        output_folder <- CONFIG$output_dir
    }
    
    if (!dir.exists(output_folder)) dir.create(output_folder, recursive = TRUE)
    
    log_file <- file.path(output_folder, paste0("xcms_cliquems_log_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".txt"))
    sink(log_file, split = TRUE)
    
    tryCatch({
        results <- process_all_files(input_folder, output_folder)
        error_count <- sum(sapply(results, function(x) x$status == "error"))
        exit_code <- ifelse(error_count > 0, 1, 0)
    }, finally = {
        sink()
        cat("Log saved to:", log_file, "\n")
    })
    
    quit(status = exit_code)
}

main()
