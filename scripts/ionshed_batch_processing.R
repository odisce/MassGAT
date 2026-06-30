#!/usr/bin/env Rscript
library(ionshed)
library(arrow)
library(xcms)
library(MSnbase)


# Function to process a single mzML/mzXML file
process_file <- function(file_path, output_folder) {
    # Check if folder exists and create it if not
    if (!dir.exists(output_folder)) {
        dir.create(output_folder, recursive = TRUE)
    }
    
    cat("Processing file:", file_path, "\n")
    time1 <- Sys.time()
    
    # Get basename for output files (handle both mzML and mzXML)
    base_path <- basename(file_path)
    base_path <- sub("\\.(mzML|mzXML)$", "", base_path, ignore.case = TRUE)
    
    # Check if output files already exist (skip if they do)
    peak_output <- file.path(output_folder, paste0(base_path, "_peak.feather"))
    roi_output <- file.path(output_folder, paste0(base_path, "_signal.feather"))
    
    if (file.exists(peak_output) && file.exists(roi_output)) {
        cat("Output files already exist for", base_path, "- skipping\n")
        return(list(status = "skipped", file = base_path))
    }
    
    tryCatch({
        # Read the mzML/mzXML file as an MSnExp object
        cat("Reading MS file...\n")
        MSnExpObj <- readMSData(file_path, mode = "onDisk", msLevel. = 1L)
        cat("MSnExp object created\n")
        
        # Compute the peak list and the signals
        cat("Computing peak list...\n")
        res <- SpMat_DescendingPeakPicking(MSnExpObj, rttol = 5, ppm = 3, mz_res = 0.0001, rt_res = 0.1)
        cat("Peak list computed\n")

        if (!is.null(res$peak_info)) {
            cat(paste("nrow(res$peak_info):", nrow(res$peak_info)), "\n")
        }
        if (!is.null(res$roi_data)) {
            cat(paste("nrow(res$roi_data):", nrow(res$roi_data)), "\n")
        }
        # Extract the XICs
        cat("Debug: About to call extract_XICs...\n")
        rttol <- 10
        roi_res <- tryCatch({
            extract_XICs(roi_ls = res, rttol = rttol)
        }, error = function(e) {
            cat("Error in extract_XICs:", e$message, "\n")
            return(NULL)
        })
        if (!is.null(roi_res)) {
            cat("XICs extracted\n")
        } else {
            cat("XICs extraction failed\n")
        }

        # Save results in arrow format
        cat("Saving results...\n")
        write_feather(res$peak_info, peak_output)
        write_feather(res$roi_data, roi_output)
        
        time2 <- Sys.time()
        processing_time <- difftime(time2, time1, units = "mins")
        
        cat("Successfully processed", base_path, "in", round(processing_time, 2), "minutes\n")
        cat("Output files saved:\n")
        cat("  Peak info:", peak_output, "\n")
        cat("  ROI data:", roi_output, "\n")
        
        # Clean up memory
        rm(MSnExpObj, res, roi_res)
        gc()
        
        return(list(
            status = "success", 
            file = base_path, 
            processing_time = processing_time,
            peak_output = peak_output,
            roi_output = roi_output
        ))
        
    }, error = function(e) {
        cat("ERROR processing", base_path, ":", conditionMessage(e), "\n")
        return(list(status = "error", file = base_path, error = conditionMessage(e)))
    })
}

# Function to process all mzML/mzXML files in a directory
process_all_files <- function(input_folder, output_folder) {
    cat("=======================================================\n")
    cat("Ionshed Batch Processing\n")
    cat("=======================================================\n")
    cat("Input folder:", input_folder, "\n")
    cat("Output folder:", output_folder, "\n")
    
    # Check if input folder exists
    if (!dir.exists(input_folder)) {
        stop("Input folder does not exist: ", input_folder)
    }
    
    # Create output folder if it doesn't exist
    if (!dir.exists(output_folder)) {
        dir.create(output_folder, recursive = TRUE)
        cat("Created output folder:", output_folder, "\n")
    }
    
    # Find all mzML and mzXML files
    ms_files <- list.files(input_folder, pattern = "\\.(mzML|mzXML)$", full.names = TRUE, ignore.case = TRUE)
    
    if (length(ms_files) == 0) {
        stop("No mzML or mzXML files found in input folder: ", input_folder)
    }
    
    cat("Found", length(ms_files), "MS files to process\n")
    
    # Print file breakdown by type
    mzml_count <- sum(grepl("\\.mzML$", ms_files, ignore.case = TRUE))
    mzxml_count <- sum(grepl("\\.mzXML$", ms_files, ignore.case = TRUE))
    cat("  - mzML files:", mzml_count, "\n")
    cat("  - mzXML files:", mzxml_count, "\n")
    cat("=======================================================\n")
    
    # Initialize results tracking
    results <- list()
    start_time <- Sys.time()
    
    # Process each file
    for (i in seq_along(ms_files)) {
        file_path <- ms_files[i]
        
        cat("\n--- Processing file", i, "of", length(ms_files), "---\n")
        cat("File:", basename(file_path), "\n")
        
        # Process the file
        result <- process_file(file_path, output_folder)
        results[[i]] <- result
        
        # Print progress
        if (result$status == "success") {
            cat("✓ SUCCESS\n")
        } else if (result$status == "skipped") {
            cat("⊘ SKIPPED\n")
        } else {
            cat("✗ ERROR\n")
        }
    }
    
    # Summary
    end_time <- Sys.time()
    total_time <- difftime(end_time, start_time, units = "mins")
    
    success_count <- sum(sapply(results, function(x) x$status == "success"))
    error_count <- sum(sapply(results, function(x) x$status == "error"))
    skipped_count <- sum(sapply(results, function(x) x$status == "skipped"))
    
    cat("\n=======================================================\n")
    cat("PROCESSING SUMMARY\n")
    cat("=======================================================\n")
    cat("Total files:", length(ms_files), "\n")
    cat("Successful:", success_count, "\n")
    cat("Errors:", error_count, "\n")
    cat("Skipped:", skipped_count, "\n")
    cat("Total processing time:", round(total_time, 2), "minutes\n")
    
    # Save processing log
    log_file <- file.path(output_folder, paste0("processing_log_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".txt"))
    
    log_content <- c(
        "Ionshed Processing Log",
        paste("Date:", Sys.time()),
        paste("Input folder:", input_folder),
        paste("Output folder:", output_folder),
        paste("Total files:", length(ms_files)),
        paste("Successful:", success_count),
        paste("Errors:", error_count),
        paste("Skipped:", skipped_count),
        paste("Total time:", round(total_time, 2), "minutes"),
        "",
        "File Details:"
    )
    
    for (result in results) {
        if (result$status == "success") {
            log_content <- c(log_content, paste("SUCCESS:", result$file, 
                           "(" , round(result$processing_time, 2), "min)"))
        } else if (result$status == "error") {
            log_content <- c(log_content, paste("ERROR:", result$file, "-", result$error))
        } else {
            log_content <- c(log_content, paste("SKIPPED:", result$file))
        }
    }
    
    writeLines(log_content, log_file)
    cat("Processing log saved to:", log_file, "\n")
    
    return(results)
}

# Main execution
main <- function() {
    # Check if running in container environment
    if (file.exists("/workspace") && file.exists("/data")) {
        cat("Running in Singularity container environment\n")
        input_folder <- "/data"
        output_folder <- "/output"
    } else {
        # Use specified paths
        input_folder <- "/spi/scidospi/01_Applications/CHODEX/mzml"
        output_folder <- "/spi/scidospi/08_Team/PinartPaulHenri/processed_with_ionshed"
    }
    
    # Create output folder if it doesn't exist
    if (!dir.exists(output_folder)) {
        dir.create(output_folder, recursive = TRUE)
    }
    
    # Set up logging in output folder
    log_file <- file.path(output_folder, paste0("ionshed_processing_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".log"))
    sink(log_file, split = TRUE)
    
    tryCatch({
        # Process all files
        results <- process_all_files(input_folder, output_folder)
        
        # Exit with appropriate code
        error_count <- sum(sapply(results, function(x) x$status == "error"))
        if (error_count > 0) {
            cat("Completed with", error_count, "errors\n")
            exit_code <- 1
        } else {
            cat("All files processed successfully!\n")
            exit_code <- 0
        }
    }, finally = {
        sink()
        cat("Log saved to:", log_file, "\n")
    })
    
    quit(status = exit_code)
}

# Run main function
if (!interactive()) {
    main()
}
