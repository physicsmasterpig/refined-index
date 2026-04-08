/*
 * _c_tet_index — C extension for the hot-path tetrahedron index computation.
 *
 * Provides:
 *   tet_index_series(m, e, qq_order)  →  dict[int, int]
 *   tet_degree_x2(m, e)              →  int
 *   poly_convolve(poly1, poly2, budget) → dict[int, int]
 *
 * These are drop-in replacements for the pure-Python versions in index_3d.py.
 * Results are bit-identical to the Python implementation.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright (c) 2024-2026  manifold-index contributors
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Helper: max/min for integers                                        */
/* ------------------------------------------------------------------ */
static inline long long imax(long long a, long long b) { return a > b ? a : b; }
static inline long long imin(long long a, long long b) { return a < b ? a : b; }

/* ------------------------------------------------------------------ */
/* tet_degree_x2(m, e) → int                                          */
/*                                                                     */
/* Returns 2 * tet_degree(m, e) as a plain integer.                    */
/* ------------------------------------------------------------------ */
static int
c_tet_degree_x2(int m, int e)
{
    int pos_m  = m > 0 ? m : 0;
    int pos_me = (m + e) > 0 ? (m + e) : 0;
    int pos_nm = (-m) > 0 ? (-m) : 0;
    int pos_e  = e > 0 ? e : 0;
    int pos_ne = (-e) > 0 ? (-e) : 0;
    int pos_nem = (-e - m) > 0 ? (-e - m) : 0;

    int half_sum = pos_m * pos_me + pos_nm * pos_e + pos_ne * pos_nem;

    int mx = 0;
    if (m > mx) mx = m;
    if (-e > mx) mx = -e;

    return half_sum + mx;  /* max{0,m,-e} is INSIDE the ½ per eq.(12): 2δ = half_sum + mx */
}

static PyObject *
py_tet_degree_x2(PyObject *self, PyObject *args)
{
    int m, e;
    if (!PyArg_ParseTuple(args, "ii", &m, &e))
        return NULL;
    return PyLong_FromLong(c_tet_degree_x2(m, e));
}

/* ------------------------------------------------------------------ */
/* inv_fact table management                                           */
/*                                                                     */
/* inv_fact[k] = 1 / prod_{j=1}^{k} (1 - qq^{2j})                    */
/*             = sum_{p>=0} c_p * qq^p,  truncated at inner_order.     */
/*                                                                     */
/* Stored as dense int64 arrays of length (inner_order + 1).           */
/* ------------------------------------------------------------------ */

/*
 * Extend the inv_fact table from its current size to include index `up_to`.
 *
 * inv_fact_table : array of pointers (realloc'd as needed)
 * inv_fact_count : current number of entries
 * inv_fact_cap   : current allocated capacity
 * inner_order    : polynomial truncation order
 *
 * Returns 0 on success, -1 on allocation failure.
 */
static int
extend_inv_fact(long long ***inv_fact_table,
                int *inv_fact_count,
                int *inv_fact_cap,
                int up_to,
                int inner_order)
{
    /* Grow the pointer array if needed */
    while (*inv_fact_cap <= up_to) {
        int new_cap = (*inv_fact_cap) * 2 + 16;
        long long **new_table = (long long **)realloc(
            *inv_fact_table, (size_t)new_cap * sizeof(long long *));
        if (!new_table) return -1;
        *inv_fact_table = new_table;
        *inv_fact_cap = new_cap;
    }

    int poly_len = inner_order + 1;

    while (*inv_fact_count <= up_to) {
        int k = *inv_fact_count;  /* index we're computing now */

        long long *new_poly = (long long *)calloc((size_t)poly_len, sizeof(long long));
        if (!new_poly) return -1;

        if (k == 0) {
            /* inv_fact[0] = 1 */
            new_poly[0] = 1;
        } else {
            /*
             * inv_fact[k] = inv_fact[k-1] * (1 + qq^{2k} + qq^{4k} + ...)
             *
             * For each coefficient in prev, propagate it forward by steps of 2k.
             */
            long long *prev = (*inv_fact_table)[k - 1];
            int step = 2 * k;
            for (int p = 0; p < poly_len; p++) {
                if (prev[p] == 0) continue;
                /* Add prev[p] * qq^{p + step*j} for j = 0, 1, 2, ... */
                for (int q = p; q < poly_len; q += step) {
                    new_poly[q] += prev[p];
                }
            }
        }

        (*inv_fact_table)[k] = new_poly;
        (*inv_fact_count)++;
    }

    return 0;
}

/*
 * Free the entire inv_fact table.
 */
static void
free_inv_fact(long long **table, int count)
{
    if (!table) return;
    for (int i = 0; i < count; i++) {
        free(table[i]);
    }
    free(table);
}

/* ------------------------------------------------------------------ */
/* it_direct(mm, ee, inner_order) → dense result array                 */
/*                                                                     */
/* Raw I_t(mm, ee) series (no MIt symmetry).                           */
/* Caller must free the returned array.                                */
/* Returns NULL on allocation failure.                                 */
/* ------------------------------------------------------------------ */
static long long *
it_direct(int mm, int ee, int inner_order)
{
    int poly_len = inner_order + 1;
    long long *result = (long long *)calloc((size_t)poly_len, sizeof(long long));
    if (!result) return NULL;

    int n_min = (-ee) > 0 ? (-ee) : 0;

    /* Build inv_fact table incrementally */
    long long **inv_fact_table = NULL;
    int inv_fact_count = 0;
    int inv_fact_cap = 0;

    for (int n = n_min; ; n++) {
        long long exp_qq = (long long)n * (n + 1) - (long long)(2 * n + ee) * mm;
        if (exp_qq > inner_order)
            break;

        /* Ensure inv_fact[n] and inv_fact[n+ee] are available */
        int need = n;
        if (n + ee > need) need = n + ee;
        if (need < 0) need = 0;

        if (extend_inv_fact(&inv_fact_table, &inv_fact_count, &inv_fact_cap,
                            need, inner_order) < 0) {
            free(result);
            free_inv_fact(inv_fact_table, inv_fact_count);
            return NULL;
        }

        long long *d1 = inv_fact_table[n];
        long long *d2;
        int ne = n + ee;

        /* Static unit polynomial for the case n+ee < 0 */
        long long unit_poly = 1;

        if (ne >= 0 && ne < inv_fact_count) {
            d2 = inv_fact_table[ne];
        } else if (ne < 0) {
            /* inv_fact for negative index = {0: 1} (unit) */
            d2 = &unit_poly;
        } else {
            /* Shouldn't happen if extend worked, but guard */
            d2 = &unit_poly;
        }

        int d2_len = (ne >= 0 && ne < inv_fact_count) ? poly_len : 1;
        long long sign = (n % 2 == 0) ? 1 : -1;

        /* Convolve d1 * d2, shift by exp_qq, add to result */
        for (int p1 = 0; p1 < poly_len; p1++) {
            if (d1[p1] == 0) continue;
            long long c1 = d1[p1];
            int budget = inner_order - (int)exp_qq - p1;
            if (budget < 0) continue;
            int p2_max = budget < (d2_len - 1) ? budget : (d2_len - 1);
            for (int p2 = 0; p2 <= p2_max; p2++) {
                if (d2[p2] == 0) continue;
                int total_pwr = (int)exp_qq + p1 + p2;
                result[total_pwr] += sign * c1 * d2[p2];
            }
        }
    }

    free_inv_fact(inv_fact_table, inv_fact_count);
    return result;
}

/* ------------------------------------------------------------------ */
/* tet_index_series(m, e, qq_order) → dict[int, int]                   */
/*                                                                     */
/* Full MIt(m, e) with symmetry.  Bit-identical to the Python version. */
/* ------------------------------------------------------------------ */
static PyObject *
py_tet_index_series(PyObject *self, PyObject *args)
{
    int m, e, qq_order;
    if (!PyArg_ParseTuple(args, "iii", &m, &e, &qq_order))
        return NULL;

    /* Non-integer guard (always true for C ints, but match Python API) */

    long long *raw = NULL;
    int raw_len;
    int shift = 0;
    int sign_m = 1;

    if (m + e >= 0) {
        /*
         * MIt(m,e) = (-qq)^m * I_t(-m-e, m)
         * Need raw keys up to qq_order - m (= qq_order + |m| when m < 0).
         */
        int inner_order = qq_order - m;
        if (inner_order < 0) inner_order = 0;
        raw = it_direct(-m - e, m, inner_order);
        raw_len = inner_order + 1;
        shift = m;
        sign_m = (m % 2 == 0) ? 1 : -1;
    } else {
        raw = it_direct(m, e, qq_order);
        raw_len = qq_order + 1;
        shift = 0;
        sign_m = 1;
    }

    if (!raw) {
        PyErr_SetString(PyExc_MemoryError,
                        "tet_index_series: allocation failed");
        return NULL;
    }

    /* Build Python dict from non-zero entries */
    PyObject *dict = PyDict_New();
    if (!dict) {
        free(raw);
        return NULL;
    }

    for (int p = 0; p < raw_len; p++) {
        if (raw[p] == 0) continue;
        int new_pwr = p + shift;
        if (new_pwr < 0 || new_pwr > qq_order) continue;
        long long coeff = sign_m * raw[p];
        if (coeff == 0) continue;

        /* Check if key already exists (possible when shift applied) */
        PyObject *key = PyLong_FromLong(new_pwr);
        PyObject *existing = PyDict_GetItem(dict, key);  /* borrowed ref */
        if (existing) {
            long long old_val = PyLong_AsLongLong(existing);
            coeff += old_val;
        }
        if (coeff != 0) {
            PyObject *val = PyLong_FromLongLong(coeff);
            PyDict_SetItem(dict, key, val);
            Py_DECREF(val);
        } else {
            /* Remove key if coefficient cancelled to zero */
            PyDict_DelItem(dict, key);
        }
        Py_DECREF(key);
    }

    free(raw);
    return dict;
}

/* ------------------------------------------------------------------ */
/* poly_convolve(poly1, poly2, budget) → dict[int, int]                */
/*                                                                     */
/* Multiply two sparse polynomials (given as dicts) with a budget      */
/* cutoff.  Only terms with power ≤ budget are kept.                   */
/* ------------------------------------------------------------------ */
static PyObject *
py_poly_convolve(PyObject *self, PyObject *args)
{
    PyObject *poly1, *poly2;
    int budget;
    if (!PyArg_ParseTuple(args, "O!O!i", &PyDict_Type, &poly1,
                          &PyDict_Type, &poly2, &budget))
        return NULL;

    /*
     * Convert both dicts to dense arrays for fast convolution.
     * This avoids the O(n*m) Python dict lookups.
     */
    int len = budget + 1;
    long long *a = (long long *)calloc((size_t)len, sizeof(long long));
    long long *b = (long long *)calloc((size_t)len, sizeof(long long));
    long long *c = (long long *)calloc((size_t)len, sizeof(long long));

    if (!a || !b || !c) {
        free(a); free(b); free(c);
        PyErr_SetString(PyExc_MemoryError, "poly_convolve: allocation failed");
        return NULL;
    }

    /* Fill array a from poly1 */
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(poly1, &pos, &key, &value)) {
        int p = (int)PyLong_AsLong(key);
        if (p >= 0 && p < len)
            a[p] = PyLong_AsLongLong(value);
    }

    /* Fill array b from poly2 */
    pos = 0;
    while (PyDict_Next(poly2, &pos, &key, &value)) {
        int p = (int)PyLong_AsLong(key);
        if (p >= 0 && p < len)
            b[p] = PyLong_AsLongLong(value);
    }

    /* Dense convolution with budget cutoff */
    for (int i = 0; i < len; i++) {
        if (a[i] == 0) continue;
        for (int j = 0; i + j < len; j++) {
            if (b[j] == 0) continue;
            c[i + j] += a[i] * b[j];
        }
    }

    /* Build result dict from non-zero entries */
    PyObject *result = PyDict_New();
    if (!result) {
        free(a); free(b); free(c);
        return NULL;
    }

    for (int p = 0; p < len; p++) {
        if (c[p] == 0) continue;
        PyObject *pk = PyLong_FromLong(p);
        PyObject *pv = PyLong_FromLongLong(c[p]);
        PyDict_SetItem(result, pk, pv);
        Py_DECREF(pk);
        Py_DECREF(pv);
    }

    free(a);
    free(b);
    free(c);
    return result;
}

/* ------------------------------------------------------------------ */
/* Module definition                                                   */
/* ------------------------------------------------------------------ */

static PyMethodDef module_methods[] = {
    {"tet_index_series", py_tet_index_series, METH_VARARGS,
     "tet_index_series(m, e, qq_order) -> dict[int, int]\n\n"
     "Compute the tetrahedron index I_Delta(m, e) as a {power: coeff} dict.\n"
     "Bit-identical to the pure-Python version but 5-50x faster."},
    {"tet_degree_x2", py_tet_degree_x2, METH_VARARGS,
     "tet_degree_x2(m, e) -> int\n\n"
     "Return 2 * tet_degree(m, e) as a plain integer (no fractions)."},
    {"poly_convolve", py_poly_convolve, METH_VARARGS,
     "poly_convolve(poly1, poly2, budget) -> dict[int, int]\n\n"
     "Multiply two sparse polynomials (dicts) with power ≤ budget."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_c_tet_index",                          /* module name */
    "C-accelerated tetrahedron index kernel.\n\n"
    "Drop-in replacements for the hot-path functions in index_3d.py.\n"
    "Falls back to pure Python automatically if this extension is unavailable.",
    -1,                                      /* per-interpreter state size */
    module_methods
};

PyMODINIT_FUNC
PyInit__c_tet_index(void)
{
    return PyModule_Create(&module_def);
}
