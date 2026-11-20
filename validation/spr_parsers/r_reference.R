# Execute original upstream import expressions; do not reproduce their algorithm.
args <- commandArgs(trailingOnly = TRUE)
mode <- args[1]
input <- normalizePath(args[2], mustWork = TRUE)
destination <- args[3]

if (mode == "rki") {
  suppressPackageStartupMessages(library(dplyr))
  suppressPackageStartupMessages(library(tidyr))
  source <- normalizePath(args[4], mustWork = TRUE)
  expressions <- parse(source)
  lhs <- vapply(expressions, function(x) {
    if (is.call(x) && identical(x[[1]], as.name("<-")))
      paste(deparse(x[[2]]), collapse = "") else ""
  }, character(1))
  first <- which(lhs == "inputDFhuCap")
  last <- which(lhs == "outputHu")
  stopifnot(length(first) == 1, length(last) == 1, first < last)
  work <- tempfile("rki-import-")
  dir.create(work)
  dir.create(file.path(work, "input"))
  stopifnot(file.copy(input, file.path(work, "input", "2022-05-10_Kurven_Binning_huCapRaw.txt")))
  setwd(work)
  # Exclude the script's installer, workspace cleanup and later normalization.
  # Every evaluated expression is unchanged from the hash-checked upstream file.
  for (i in seq.int(first, last)) eval(expressions[[i]], envir = .GlobalEnv)
  keys <- unique(paste(outputHu$cycle, outputHu$flowcell, sep = ":"))
  curves <- lapply(keys, function(key) {
    d <- outputHu[paste(outputHu$cycle, outputHu$flowcell, sep = ":") == key, ]
    stopifnot(all(is.na(d$time) == is.na(d$RU)))
    d <- d[!is.na(d$time), ]
    list(name = key, time = unname(d$time), response = unname(d$RU))
  })
  reference <- "RKI binDataImport_final.R: original raw import through outputHu"
} else if (mode == "carterra") {
  # Independent workbook reader; this is not TitrationAnalysis execution.
  sheet <- readxl::read_excel(input, n_max = 3, .name_repair = "minimal")
  stopifnot(names(sheet)[1] == "Label", (ncol(sheet) - 1) %% 2 == 0)
  curves <- lapply(seq_len(nrow(sheet)), function(i) {
    list(name = sheet[[1]][i],
         time = as.numeric(sheet[i, seq(2, ncol(sheet), 2)]),
         response = as.numeric(sheet[i, seq(3, ncol(sheet), 2)]))
  })
  reference <- "readxl::read_excel: first three native Carterra workbook rows"
} else stop("Unknown reference mode")

jsonlite::write_json(list(reference = reference, version = R.version.string,
                         packages = as.list(vapply(c("readxl", "dplyr", "tidyr", "jsonlite"),
                           function(p) as.character(packageVersion(p)), character(1))),
                         curves = curves), destination, auto_unbox = TRUE, digits = NA)
