#!/usr/bin/env Rscript

if (!file.exists("DESCRIPTION") || !file.exists("README.md")) {
  stop("Run this script from the mlr3mbspls package root.", call. = FALSE)
}
if (!requireNamespace("pkgload", quietly = TRUE)) {
  stop("README verification requires the suggested package 'pkgload'.",
    call. = FALSE)
}

pkgload::load_all(quiet = TRUE)

lines = readLines("README.md", warn = FALSE, encoding = "UTF-8")
chunks = list()
line_number = 1L

while (line_number <= length(lines)) {
  opening = regexec(
    "^[ \\t]*(`{3,}|~{3,})[ \\t]*(.*?)[ \\t]*$",
    lines[[line_number]],
    perl = TRUE
  )
  fields = regmatches(lines[[line_number]], opening)[[1L]]
  if (!length(fields)) {
    line_number = line_number + 1L
    next
  }

  marker = fields[[2L]]
  info = fields[[3L]]
  is_r = grepl("^(?i:r)(?:[[:space:]].*)?$", info, perl = TRUE)
  closing_pattern = sprintf(
    "^[ \\t]*%s{%d,}[ \\t]*$",
    substr(marker, 1L, 1L),
    nchar(marker)
  )
  closing_line = line_number + 1L
  while (closing_line <= length(lines) &&
    !grepl(closing_pattern, lines[[closing_line]], perl = TRUE)) {
    closing_line = closing_line + 1L
  }
  if (closing_line > length(lines)) {
    stop(sprintf("Unclosed Markdown fence at README.md:%d.", line_number),
      call. = FALSE)
  }

  if (is_r) {
    code = if (closing_line == line_number + 1L) {
      character()
    } else {
      lines[seq.int(line_number + 1L, closing_line - 1L)]
    }
    chunks[[length(chunks) + 1L]] = list(
      line = line_number,
      code = code,
      evaluate = !grepl("(^|[[:space:]])no-eval($|[[:space:]])", info)
    )
  }
  line_number = closing_line + 1L
}

if (!length(chunks)) {
  stop("README.md contains no R code fences.", call. = FALSE)
}

example_environment = new.env(parent = globalenv())
old_device = getOption("device")
on.exit(options(device = old_device), add = TRUE)
options(device = function(...) {
  grDevices::pdf(file = tempfile("mlr3mbspls-readme-", fileext = ".pdf"))
})

for (chunk in chunks) {
  expression = tryCatch(
    parse(text = chunk$code, keep.source = TRUE),
    error = function(error) {
      stop(sprintf(
        "README R fence at line %d does not parse: %s",
        chunk$line,
        conditionMessage(error)
      ), call. = FALSE)
    }
  )
  if (isTRUE(chunk$evaluate)) {
    tryCatch(
      eval(expression, envir = example_environment),
      error = function(error) {
        stop(sprintf(
          "README R fence at line %d failed: %s",
          chunk$line,
          conditionMessage(error)
        ), call. = FALSE)
      }
    )
  }
}

graphics.off()
cat(sprintf(
  "README verification passed: %d R fences parsed; %d executable fences ran.\n",
  length(chunks),
  sum(vapply(chunks, `[[`, logical(1L), "evaluate"))
))
