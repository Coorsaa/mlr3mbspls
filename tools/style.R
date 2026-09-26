#!/usr/bin/env Rscript

args = commandArgs(trailingOnly = TRUE)
allowed_args = c("--check", "--help", "-h")
unknown_args = setdiff(args, allowed_args)

if (length(unknown_args) > 0L) {
  stop(
    sprintf("Unknown style argument(s): %s", paste(unknown_args, collapse = ", ")),
    call. = FALSE
  )
}

if ("--help" %in% args || "-h" %in% args) {
  cat("Usage: Rscript tools/style.R [--check]\n")
  cat("\n")
  cat("Without --check, format all repository R code with styler.mlr.\n")
  cat("With --check, report files that require formatting and change nothing.\n")
  quit(status = 0L)
}

check_only = "--check" %in% args
required_styler_version = package_version("1.10.3")
required_mlr_version = package_version("0.1.0")
required_mlr_revision = "e950499afb6b28610400489f3ae6faacc6897358"

if (!file.exists("DESCRIPTION")) {
  stop("Run this script from the mlr3mbspls package root.", call. = FALSE)
}

if (!requireNamespace("styler", quietly = TRUE) ||
  !requireNamespace("styler.mlr", quietly = TRUE)) {
  stop(
    paste(
      "The pinned style dependencies are missing.",
      "See CONTRIBUTING.md for installation instructions."
    ),
    call. = FALSE
  )
}

if (packageVersion("styler") != required_styler_version) {
  stop(
    sprintf(
      "styler %s is required; found %s.",
      required_styler_version,
      packageVersion("styler")
    ),
    call. = FALSE
  )
}

if (packageVersion("styler.mlr") != required_mlr_version) {
  stop(
    sprintf(
      "styler.mlr %s is required; found %s.",
      required_mlr_version,
      packageVersion("styler.mlr")
    ),
    call. = FALSE
  )
}

mlr_description = packageDescription("styler.mlr")
installed_revision = mlr_description[["RemoteSha"]]
if (!is.null(installed_revision) && nzchar(installed_revision) &&
  installed_revision != required_mlr_revision) {
  stop(
    sprintf(
      "styler.mlr revision %s is required; found %s.",
      required_mlr_revision,
      installed_revision
    ),
    call. = FALSE
  )
}

R.cache::setCacheRootPath(file.path(getwd(), ".styler-cache"))
styler.mlr::cache_activate(
  cache_name = "mlr3mbspls-styler-1.10.3-mlr-e950499",
  verbose = FALSE
)

project_files = list.files(
  ".",
  recursive = TRUE,
  all.files = TRUE,
  full.names = TRUE,
  include.dirs = FALSE
)
project_files = sub("^\\./", "", project_files)
excluded = grepl(
  "^(\\.git|\\.agents|\\.codex|\\.Rproj\\.user|\\.styler-cache|docs|release)(/|$)|(^|/)AGENTS[.]md$|^inst/(AUDIT_REPORT|INFERENCE_RESEARCH)[.]md$",
  project_files
)
project_files = sort(project_files[!excluded])

source_extensions = c(".r", ".rmd", ".rmarkdown", ".qmd", ".rprofile")
source_files = project_files[
  vapply(
    tolower(project_files),
    function(file) any(endsWith(file, source_extensions)),
    logical(1L)
  )
]
markdown_files = project_files[endsWith(tolower(project_files), ".md")]
plain_r_files = intersect("inst/CITATION", project_files)
fenced_document_extensions = c(".rmd", ".rmarkdown", ".qmd")
fenced_document_files = unique(c(
  markdown_files,
  source_files[
    vapply(
      tolower(source_files),
      function(file) any(endsWith(file, fenced_document_extensions)),
      logical(1L)
    )
  ]
))

style_source_files = function(files, dry) {
  if (length(files) == 0L) {
    return(character())
  }

  result = NULL
  tryCatch(
    {
      invisible(utils::capture.output(
        result = suppressMessages(styler.mlr::style_file(files, dry = dry))
      ))
    },
    error = function(error) {
      stop(
        sprintf("Could not style repository R files: %s", conditionMessage(error)),
        call. = FALSE
      )
    }
  )

  as.character(result$file[result$changed])
}

assert_no_left_assignment = function(text, context) {
  parsed = tryCatch(
    parse(text = text, keep.source = TRUE),
    error = function(error) {
      stop(
        sprintf("Could not parse %s: %s", context, conditionMessage(error)),
        call. = FALSE
      )
    }
  )
  parse_data = getParseData(parsed)
  is_left_arrow = parse_data$token == "LEFT_ASSIGN" &
    parse_data$text %in% c("<-", "<<-")
  arrow_lines = parse_data$line1[is_left_arrow]
  comment_lines = grep("^[[:space:]]*#.*<-", text)
  arrow_lines = sort(unique(c(arrow_lines, comment_lines)))
  if (length(arrow_lines) > 0L) {
    stop(
      sprintf(
        paste(
          "%s still contains `<-` on line(s) %s.",
          "Use an explicit block when `=` would otherwise be parsed as a named argument."
        ),
        context,
        paste(arrow_lines, collapse = ", ")
      ),
      call. = FALSE
    )
  }
  invisible(text)
}

style_text = function(text, context) {
  if (length(text) == 0L) {
    return(text)
  }

  result = tryCatch(
    suppressMessages(styler.mlr::style_text(text)),
    error = function(error) {
      stop(
        sprintf("Could not style %s: %s", context, conditionMessage(error)),
        call. = FALSE
      )
    }
  )
  result = as.character(result)
  assert_no_left_assignment(result, context)
  result
}

parse_r_fence = function(line) {
  match = regexec(
    "^([ \\t]*)(`{3,}|~{3,})[ \\t]*(.*?)[ \\t]*$",
    line,
    perl = TRUE
  )
  fields = regmatches(line, match)[[1L]]
  if (length(fields) == 0L) {
    return(NULL)
  }

  info = fields[[4L]]
  is_r = grepl("^(?i:r)(?:[[:space:]].*)?$", info, perl = TRUE) ||
    grepl("^\\{(?i:r)(?:[[:space:],].*)?\\}$", info, perl = TRUE)
  if (!is_r) {
    return(NULL)
  }

  list(indent = fields[[2L]], marker = fields[[3L]])
}

is_closing_fence = function(line, marker) {
  marker_character = substr(marker, 1L, 1L)
  pattern = sprintf(
    "^[ \\t]*%s{%i,}[ \\t]*$",
    marker_character,
    nchar(marker)
  )
  grepl(pattern, line, perl = TRUE)
}

style_fenced_code = function(code, indent, context) {
  if (length(code) == 0L) {
    return(code)
  }

  has_shared_indent = nzchar(indent) &&
    all(!nzchar(code) | startsWith(code, indent))
  if (has_shared_indent) {
    code = ifelse(
      nzchar(code),
      substring(code, nchar(indent) + 1L),
      ""
    )
  }

  styled = style_text(code, context)
  if (has_shared_indent) {
    styled = ifelse(nzchar(styled), paste0(indent, styled), "")
  }
  styled
}

style_markdown = function(file) {
  input = readLines(file, warn = FALSE, encoding = "UTF-8")
  output = character()
  line_number = 1L

  while (line_number <= length(input)) {
    fence = parse_r_fence(input[[line_number]])
    if (is.null(fence)) {
      output = c(output, input[[line_number]])
      line_number = line_number + 1L
      next
    }

    closing_line = line_number + 1L
    while (closing_line <= length(input) &&
      !is_closing_fence(input[[closing_line]], fence$marker)) {
      closing_line = closing_line + 1L
    }
    if (closing_line > length(input)) {
      stop(
        sprintf("Unclosed R fence in %s at line %i.", file, line_number),
        call. = FALSE
      )
    }

    code = if (closing_line == line_number + 1L) {
      character()
    } else {
      input[seq.int(line_number + 1L, closing_line - 1L)]
    }
    context = sprintf("R fence in %s at line %i", file, line_number)
    styled = style_fenced_code(code, fence$indent, context)
    output = c(
      output,
      input[[line_number]],
      styled,
      input[[closing_line]]
    )
    line_number = closing_line + 1L
  }

  if (!identical(input, output) && !check_only) {
    writeLines(output, file, useBytes = TRUE)
  }
  !identical(input, output)
}

style_plain_r = function(file) {
  input = readLines(file, warn = FALSE, encoding = "UTF-8")
  output = style_text(input, file)
  if (!identical(input, output) && !check_only) {
    writeLines(output, file, useBytes = TRUE)
  }
  !identical(input, output)
}

dry = if (check_only) "on" else "off"
changed_source = style_source_files(source_files, dry = dry)
plain_source_files = source_files[
  endsWith(tolower(source_files), ".r") |
    endsWith(tolower(source_files), ".rprofile")
]
invisible(lapply(plain_source_files, function(file) {
  assert_no_left_assignment(
    readLines(file, warn = FALSE, encoding = "UTF-8"),
    file
  )
}))
changed_markdown = fenced_document_files[
  vapply(fenced_document_files, style_markdown, logical(1L))
]
changed_plain_r = plain_r_files[
  vapply(plain_r_files, style_plain_r, logical(1L))
]
changed = unique(c(changed_source, changed_markdown, changed_plain_r))

if (check_only && length(changed) > 0L) {
  cat("styler.mlr check failed. Run `Rscript tools/style.R`.\n")
  cat(paste0("- ", changed, "\n"), sep = "")
  quit(status = 1L)
}

if (length(changed) == 0L) {
  cat("styler.mlr check passed: all repository R code is formatted.\n")
} else {
  cat(sprintf("Formatted %i file(s) with styler.mlr:\n", length(changed)))
  cat(paste0("- ", changed, "\n"), sep = "")
}
