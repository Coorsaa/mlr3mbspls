#' k-Nearest Neighbours with Gower Distance - Regression
#'
#' @title Learner: `regr.knngower`
#' @name LearnerRegrKNNGower
#' @format R6 class inheriting from [mlr3::LearnerRegr]
#'
#' @description
#' A *k*-nearest neighbours regressor using **Gower distance**
#' (Rcpp/Armadillo backend). Supports numeric, unordered factors, and
#' ordered factors with the same encodings as the classification learner.
#' Predictions are (optionally inverse-distance) **weighted means** of the
#' neighbour responses; the (weighted) variance provides an approximate
#' `se` when requested. If no eligible neighbours exist, prediction falls
#' back to the training mean/variance.
#'
#' @section Parameters (in `param_set`):
#' \describe{
#'   \item{`k`}{`integer(1)`. Number of neighbours. Default: `5`.}
#'   \item{`weights`}{`character(1)`. `"uniform"` or `"inverse"`.
#'     Default: `"inverse"`.}
#'   \item{`min_feature_frac`}{`numeric(1)` in `[0,1]`. Minimum fraction of
#'     comparable features required for a neighbour. Default: `0.2`.}
#'   \item{`na_handling`}{`character(1)`. `"pairwise"` (skip missing values
#'     per feature) or `"fail"`. Default: `"pairwise"`.}
#' }
#'
#' @section Prediction:
#' Returns numeric `response`; if requested, `se` is the square root of the
#' (weighted) neighbour variance (non-unbiased).
#'
#' @param id `character(1)`. Learner identifier (default `"regr.knngower"`).
#' @param param_vals `list()`. Named list of initial hyperparameters.
#'
#' @return An object of class `LearnerRegrKNNGower`.
#'
#' @examples
#' if (requireNamespace("mlr3", quietly = TRUE)) {
#'   library(mlr3)
#'   set.seed(1)
#'   n = 100
#'   df = data.frame(x1 = runif(n), x2 = runif(n))
#'   df$y = sin(3 * df$x1) + df$x2 + rnorm(n, sd = 0.1)
#'   task = TaskRegr$new("toy", backend = df, target = "y")
#'   lrn = lrn("regr.knngower", k = 10, weights = "inverse")
#'   lrn$train(task)
#'   lrn$predict(task)$score(msr("regr.rmse"))
#' }
#'
#' @export
LearnerRegrKNNGower = R6::R6Class("LearnerRegrKNNGower",
  inherit = mlr3::LearnerRegr,
  public = list(
    #' @description Create a new `LearnerRegrKNNGower` instance.
    initialize = function() {
      ps = paradox::ps(
        k = paradox::p_int(lower = 1L, default = 5L, tags = c("train", "predict")),
        weights = paradox::p_fct(levels = c("uniform", "inverse"),
          default = "inverse", tags = "predict"),
        min_feature_frac = paradox::p_dbl(lower = 0, upper = 1,
          default = 0.2, tags = "predict"),
        na_handling = paradox::p_fct(levels = c("pairwise", "fail"),
          default = "pairwise", tags = c("train", "predict"))
      )
      ps$set_values(k = 5L, weights = "inverse",
        min_feature_frac = 0.2, na_handling = "pairwise")

      super$initialize(
        id = "regr.knngower",
        feature_types = c("numeric", "integer", "factor", "ordered"),
        predict_types = c("response", "se"),
        param_set = ps,
        properties = c("missings"),
        packages = "mlr3mbspls"
      )
    }
  ),
  private = list(

    # same encoder as in the classif learner
    .encode_blocks = function(df, num_cols, cat_cols, ord_cols, ref = NULL) {
      n = nrow(df)

      # numeric
      if (length(num_cols)) {
        Xn = as.matrix(df[, num_cols, with = FALSE])
        storage.mode(Xn) = "double"
        if (is.null(ref)) {
          r_min = suppressWarnings(apply(Xn, 2, min, na.rm = TRUE))
          r_max = suppressWarnings(apply(Xn, 2, max, na.rm = TRUE))
          rng = r_max - r_min
          rng[!is.finite(rng) | rng <= 0] = 1.0
        } else {
          rng = ref$ranges_num
        }
      } else {
        Xn = matrix(numeric(0), nrow = n, ncol = 0)
        rng = numeric(0)
      }

      # categorical
      if (length(cat_cols)) {
        if (is.null(ref)) {
          cat_levels = lapply(cat_cols, function(cn) levels(as.factor(df[[cn]])))
        } else {
          cat_levels = ref$cat_levels
        }
        Xc = matrix(0L, nrow = n, ncol = length(cat_cols))
        for (j in seq_along(cat_cols)) {
          x = df[[cat_cols[j]]]
          lv = cat_levels[[j]]
          if (is.null(ref)) {
            code = as.integer(factor(x, levels = lv))
            code[is.na(code)] = 0L
          } else {
            m = match(as.character(x), lv)
            code = ifelse(is.na(x), 0L, ifelse(is.na(m), -1L, as.integer(m)))
          }
          Xc[, j] = code
        }
        storage.mode(Xc) = "integer"
      } else {
        Xc = matrix(integer(0), nrow = n, ncol = 0)
        cat_levels = list()
      }

      # ordered -> [0,1]
      if (length(ord_cols)) {
        if (is.null(ref)) {
          ord_levels = lapply(ord_cols, function(cn) levels(as.ordered(df[[cn]])))
        } else {
          ord_levels = ref$ord_levels
        }
        Xo = matrix(NA_real_, nrow = n, ncol = length(ord_cols))
        for (j in seq_along(ord_cols)) {
          x = df[[ord_cols[j]]]
          lv = ord_levels[[j]]
          if (is.null(ref)) {
            code = as.integer(as.ordered(x))
          } else {
            m = match(as.character(x), lv)
            code = ifelse(is.na(x), NA_integer_, as.integer(m))
          }
          L = length(lv)
          if (L <= 1L) {
            Xo[, j] = 0
          } else {
            Xo[, j] = (as.numeric(code) - 1) / (L - 1)
          }
        }
        storage.mode(Xo) = "double"
      } else {
        Xo = matrix(numeric(0), nrow = n, ncol = 0)
        ord_levels = list()
      }

      list(
        Xnum = Xn, Xcat = Xc, Xord = Xo,
        ranges_num = as.numeric(rng),
        cat_levels = cat_levels,
        ord_levels = ord_levels
      )
    },

    .train = function(task) {
      pv = self$param_set$get_values(tags = "train")
      y = task$truth()
      if (anyNA(y)) stop("Regression target contains missing values.")

      if (identical(pv$na_handling, "fail")) {
        x_all = task$data(cols = task$feature_names)
        if (anyNA(x_all)) stop("Training data contain missing features (na_handling = 'fail').")
      }

      df = task$data(cols = task$feature_names)
      types = task$feature_types
      num_cols = types[type %in% c("numeric", "integer"), id]
      cat_cols = types[type %in% c("factor"), id]
      ord_cols = types[type %in% c("ordered"), id]

      enc = private$.encode_blocks(df, num_cols, cat_cols, ord_cols, ref = NULL)

      y_num = as.numeric(y)
      y_var = stats::var(y_num, na.rm = TRUE)
      if (!is.finite(y_var) || is.na(y_var)) y_var <- 0

      list(
        num_cols = num_cols,
        cat_cols = cat_cols,
        ord_cols = ord_cols,
        Xnum = enc$Xnum,
        Xcat = enc$Xcat,
        Xord = enc$Xord,
        ranges_num = enc$ranges_num,
        cat_levels = enc$cat_levels,
        ord_levels = enc$ord_levels,
        y = y_num,
        y_mean = mean(y_num, na.rm = TRUE),
        y_var = y_var
      )
    },

    .predict = function(task) {
      st = self$model
      pv = self$param_set$get_values(tags = "predict")

      df = task$data(cols = task$feature_names)
      enc_te = private$.encode_blocks(
        df,
        num_cols = st$num_cols,
        cat_cols = st$cat_cols,
        ord_cols = st$ord_cols,
        ref = list(
          ranges_num = st$ranges_num,
          cat_levels = st$cat_levels,
          ord_levels = st$ord_levels
        )
      )

      # NOTE: call positionally
      out = knn_predict_regr_gower_cpp(
        st$Xnum, st$Xcat, st$Xord,
        enc_te$Xnum, enc_te$Xcat, enc_te$Xord,
        st$ranges_num, st$y,
        as.integer(pv$k),
        pv$weights,
        as.numeric(pv$min_feature_frac),
        as.numeric(st$y_mean),
        as.numeric(st$y_var)
      )

      response = as.numeric(out$mean)
      if ("se" %in% self$predict_type) {
        se = sqrt(pmax(0, as.numeric(out$var)))
        mlr3::PredictionRegr$new(task = task, response = response, se = se)
      } else {
        mlr3::PredictionRegr$new(task = task, response = response)
      }
    }
  )
)
