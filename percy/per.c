#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define EPS 1e-6
#define BETA_INC 1e-3
#define BATCH_SIZE 32
#define ELEM_COUNT 200

// Utility functions

static inline size_t min_size_t(size_t a, size_t b) { return a < b ? a : b; }
static inline size_t max_size_t(size_t a, size_t b) { return a > b ? a : b; }

int rand_int(int min, int max) { return min + rand() % (max - min + 1); }

double rand_double_range(double min, double max) {
  return min + ((double)rand() / RAND_MAX) * (max - min);
}

// Sum Tree implementation

typedef struct {
  double *priority_tree;
  PyObject **data;
  size_t capacity;
  size_t current_index;
  size_t num_entries;
} SumTree;

typedef struct {
  size_t p_idx;
  size_t d_idx;
  double priority;
} SumTreeSample;

static inline size_t sumtree_tree_size(const SumTree *t) {
  return 2 * t->capacity - 1;
}

static inline size_t sumtree_leaf_base(const SumTree *t) {
  return t->capacity - 1;
}

static inline size_t sumtree_leaf_index(const SumTree *t, size_t data_index) {
  return sumtree_leaf_base(t) + data_index;
}

// static inline void *sumtree_data_ptr(SumTree *t, size_t data_index) {
//     return (char *)t->data + data_index * t->elem_size;
// }

SumTree *create_sum_tree(size_t capacity) {
  assert(capacity > 0);

  // Check if the capacity is power of two
  assert((capacity & (capacity - 1)) == 0);

  SumTree *sum_tree = (SumTree *)malloc(sizeof(SumTree));

  if (sum_tree == NULL) {
    return NULL;
  }

  sum_tree->capacity = capacity;
  sum_tree->num_entries = 0;
  sum_tree->current_index = 0;

  // sum_tree->data = malloc(elem_size * capacity);
  sum_tree->data = (PyObject **)malloc(sizeof(PyObject *) * capacity);
  if (sum_tree->data == NULL) {
    free(sum_tree);
    return NULL;
  }

  sum_tree->priority_tree =
      (double *)calloc((2 * capacity - 1), sizeof(double));
  if (sum_tree->priority_tree == NULL) {
    free(sum_tree->data);
    free(sum_tree);
    return NULL;
  }

  return sum_tree;
}

void sum_tree_update(SumTree *sum_tree, size_t tree_idx, double priority) {
  // Very unlikely but it can happen
  assert(tree_idx < sumtree_tree_size(sum_tree));

  double old_priority = sum_tree->priority_tree[tree_idx];
  double priority_change = priority - old_priority;
  sum_tree->priority_tree[tree_idx] = priority;

  while (tree_idx > 0) {
    tree_idx = (size_t)(tree_idx - 1) / 2;
    sum_tree->priority_tree[tree_idx] += priority_change;
  }
}

void sum_tree_add(SumTree *sum_tree, PyObject *item, double priority) {
  size_t elem_idx = sumtree_leaf_index(sum_tree, sum_tree->current_index);

  PyObject *old_item = sum_tree->data[sum_tree->current_index];
  Py_INCREF(item); // Increase reference count of the new item
  sum_tree->data[sum_tree->current_index] = item;
  Py_XDECREF(old_item); // Decrease reference count of the old item if not NULL

  sum_tree_update(sum_tree, elem_idx, priority);

  sum_tree->current_index++;
  sum_tree->current_index %= sum_tree->capacity;

  sum_tree->num_entries =
      min_size_t(sum_tree->num_entries + 1, sum_tree->capacity);
}

static int sum_tree_get(SumTree *sum_tree, double segment, SumTreeSample *out) {
  assert(sum_tree);
  assert(out);

  // Check if there are elements
  double total = sum_tree->priority_tree[0];
  if (total <= 0.0) {
    *out = (SumTreeSample){0};
    return 0;
  }

  // Make sure that segment is not negative otherwise it will land on the first
  // element
  if (segment < 0.0)
    segment = 0.0;

  // Make sure that segment is not larger than the sum of priorities
  if (segment >= total)
    segment = nextafter(total, 0.0);

  size_t idx = 0;
  size_t leaf_base = sumtree_leaf_base(sum_tree);

  while (idx < leaf_base) {
    size_t left = (idx << 1) + 1;
    double left_sum = sum_tree->priority_tree[left];

    if (segment <= left_sum)
      idx = left;
    else {
      segment -= left_sum;
      idx = left + 1;
    }
  }

  size_t data_index = idx - sumtree_leaf_base(sum_tree);

  out->p_idx = idx;
  out->d_idx = data_index;
  out->priority = sum_tree->priority_tree[idx];
  return 1;
}

void sum_tree_show(SumTree *sum_tree) {
  size_t priority_tree_size = sumtree_tree_size(sum_tree);
  for (size_t level_start = 0, level_count = 1;
       level_start < priority_tree_size;
       level_start += level_count, level_count *= 2) {
    for (size_t i = 0; i < level_count && level_start + i < priority_tree_size;
         i++) {
      printf("%f ", sum_tree->priority_tree[level_start + i]);
    }
    printf("\n");
  }
}

// void sum_data_show(SumTree *sum_tree) {
//     for (size_t index = 0; index < sum_tree->capacity; ++index) {
//         int   value;
//         void *src = sumtree_data_ptr(sum_tree, index);
//         memcpy(&value, src, sum_tree->elem_size);
//         printf("%d ", value);
//     }
//     printf("\n");
// }

void free_sum_tree(SumTree *sum_tree) {
  if (!sum_tree)
    return;

  if (sum_tree->data) {
    for (size_t i = 0; i < sum_tree->capacity; ++i) {
      Py_XDECREF(sum_tree->data[i]);
    }
    free(sum_tree->data);
  }

  free(sum_tree->priority_tree);
  free(sum_tree);
}

// Prioritized Experience Replay (PER) implementation

typedef struct {
  double *items;
  size_t count;
  size_t capacity;
} TD_ERRORS;

typedef struct {
  SumTree *tree;
  PyObject **items; // ring buffer of generic python objects
  double alpha;
  double beta;
  double max_priority;
} PER;

typedef struct {
  SumTreeSample *items;
  size_t count;
  double *importance_weights;
} Batch;

void free_per(PER *per) {
  if (!per)
    return;
  free_sum_tree(per->tree);
  free(per);
}

static PER *create_prioritized_replay(size_t capacity, double alpha,
                                      double beta) {
  PER *per = (PER *)malloc(sizeof(PER));
  if (per == NULL) {
    return NULL;
  }

  per->tree = create_sum_tree(capacity);

  if (!per->tree) {
    free(per);
    return NULL;
  }

  per->alpha = alpha;
  per->beta = beta;
  per->max_priority = 1.0;
  return per;
}

double calculate_priority(const PER *per, double td_error) {
  return pow(fabs(td_error) + EPS, per->alpha);
}

// void add_to_per(PER *per, PyObject *item) {
//     sum_tree_add(per->tree, item, per->max_priority);
// }

static void add_to_per(PER *p, PyObject *item, double priority,
                       int use_priority) {
  double pr = use_priority ? priority : p->max_priority;

  if (pr < 0.0) {
    pr = 0.0;
  }

  sum_tree_add(p->tree, item, pr);

  if (pr > p->max_priority)
    p->max_priority = pr;
}

void calculate_sampling_priorities(const Batch *batch, double tree_top_value,
                                   size_t total_entry_count, double beta) {
  if (total_entry_count == 0 || tree_top_value <= 0.0) {
    memset(batch->importance_weights, 0,
           batch->count * sizeof(batch->importance_weights[0]));
    return;
  }

  double max_importance_weight = 0.0;

  for (size_t i = 0; i < batch->count; ++i) {
    if (tree_top_value <= 0.0) {
      batch->importance_weights[i] = 0.0;
      continue;
    }

    double prob = batch->items[i].priority / tree_top_value;
    if (prob < 1e-12)
      prob = 1e-12;

    double w = pow(1.0 / ((double)total_entry_count * prob), beta);
    batch->importance_weights[i] = w;

    if (w > max_importance_weight)
      max_importance_weight = w;
  }

  // Normalise once - guard against division by zero
  if (max_importance_weight <= 0.0) {
    // all weights are 0 already
    return;
  }

  for (size_t i = 0; i < batch->count; ++i) {
    batch->importance_weights[i] /= max_importance_weight;
  }
}

Batch sample_from_per(PER *per, size_t batch_size) {
  assert(per->tree->num_entries >= batch_size);

  Batch batch = {0};
  batch.items = (SumTreeSample *)malloc(batch_size * sizeof(batch.items[0]));
  batch.importance_weights = (double *)malloc(batch_size * sizeof(double));

  if (!batch.items || !batch.importance_weights) {
    free(batch.items);
    free(batch.importance_weights);
    return (Batch){0};
  }

  batch.count = batch_size;

  double tree_top_value = per->tree->priority_tree[0];
  if (tree_top_value <= 0.0) {
    for (size_t i = 0; i < batch_size; ++i) {
      batch.items[i] = (SumTreeSample){0};
      batch.importance_weights[i] = 0.0;
    }
    return batch;
  }

  double segment = tree_top_value / (double)batch_size;

  per->beta = fmin(1.0, per->beta + BETA_INC);

  for (size_t i = 0; i < batch_size; ++i) {
    double a = segment * (double)i;
    double b = segment * (double)(i + 1);
    double x = rand_double_range(a, b);

    // keep strictly inside [0, tree_top_value)
    if (x >= tree_top_value)
      x = nextafter(tree_top_value, 0.0);

    sum_tree_get(per->tree, x, &batch.items[i]);
  }

  calculate_sampling_priorities(&batch, tree_top_value, per->tree->num_entries,
                                per->beta);
  return batch;
}

static inline void free_batch(Batch *b) {
  free(b->items);
  free(b->importance_weights);
  b->items = NULL;
  b->importance_weights = NULL;
}

// PyCapsule API

static void capsule_per_destructor(PyObject *capsule) {
  PER *per = (PER *)PyCapsule_GetPointer(capsule, "PER");
  free_per(per);
}

static PyObject *create_py_per(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  (void)self;
  static char *kwlist[] = {"capacity", "alpha", "beta", NULL};

  Py_ssize_t capacity_ss = 0;
  double alpha = 0.6;
  double beta = 0.4;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ndd", kwlist, &capacity_ss,
                                   &alpha, &beta)) {
    return NULL;
  }

  if (capacity_ss <= 0) {
    PyErr_SetString(PyExc_ValueError, "capacity must be > 0");
    return NULL;
  }

  size_t capacity = (size_t)capacity_ss;
  if ((capacity & (capacity - 1)) != 0) {
    PyErr_SetString(PyExc_ValueError, "capacity must be a power of two");
    return NULL;
  }

  PER *per = create_prioritized_replay(capacity, alpha, beta);

  if (!per) {
    PyErr_SetString(PyExc_MemoryError, "Failed to create PER instance");
    return NULL;
  }

  PyObject *capsule = PyCapsule_New((void *)per, "PER", capsule_per_destructor);
  return capsule;
}

void update_per_priorities(PER *per, TD_ERRORS *td_errors,
                           size_t *priority_indices) {
  assert(per && per->tree && td_errors && priority_indices);

  for (size_t idx = 0; idx < td_errors->count; ++idx) {
    double new_priority = calculate_priority(per, td_errors->items[idx]);
    sum_tree_update(per->tree, priority_indices[idx], new_priority);
    per->max_priority = fmax(per->max_priority, new_priority);
  }
}

static PER *get_per_from_capsule(PyObject *capsule) {
  PER *per = (PER *)PyCapsule_GetPointer(capsule, "PER");
  if (!per) {
    PyErr_SetString(PyExc_ValueError, "Invalid PER capsule");
    return NULL;
  }
  return per;
}

// per.size(handle)
static PyObject *py_per_size(PyObject *self, PyObject *args) {
  (void)self;

  PyObject *capsule;
  if (!PyArg_ParseTuple(args, "O", &capsule)) {
    return NULL;
  }

  PER *p = get_per_from_capsule(capsule);
  if (!p) {
    return NULL;
  }

  return PyLong_FromSize_t(p->tree->num_entries);
}

// per.total_priority(handle)
static PyObject *py_per_total_priority(PyObject *self, PyObject *args) {
  (void)self;

  PyObject *capsule;

  if (!PyArg_ParseTuple(args, "O", &capsule)) {
    return NULL;
  }

  PER *p = get_per_from_capsule(capsule);
  if (!p) {
    return NULL;
  }

  return PyFloat_FromDouble(p->tree->priority_tree[0]);
}

// per.add(handle, item)
static PyObject *py_per_add(PyObject *self, PyObject *args) {
  (void)self;

  PyObject *capsule;
  PyObject *item;
  PyObject *priority_obj = NULL;

  if (!PyArg_ParseTuple(args, "OO|O", &capsule, &item, &priority_obj)) {
    return NULL;
  }

  PER *p = get_per_from_capsule(capsule);
  if (!p) {
    return NULL;
  }

  int use_priority = 0;
  double priority = 0.0;

  if (priority_obj && priority_obj != Py_None) {
    priority = PyFloat_AsDouble(priority_obj);
    use_priority = 1;
  }

  add_to_per(p, item, priority, use_priority);

  Py_RETURN_NONE;
}

// per.sample(handle, batch_size)
static PyObject *py_per_sample(PyObject *self, PyObject *args) {
  (void)self;

  PyObject *capsule;
  size_t batch_size = BATCH_SIZE;

  if (!PyArg_ParseTuple(args, "O|n", &capsule, &batch_size)) {
    return NULL;
  }

  PER *p = get_per_from_capsule(capsule);
  if (!p) {
    return NULL;
  }

  Batch batch = sample_from_per(p, batch_size);
  if (batch.count == 0) {
    PyErr_SetString(PyExc_MemoryError, "Failed to sample batch from PER");
    return NULL;
  }

  PyObject *result_list = PyList_New((Py_ssize_t)batch.count);
  if (!result_list) {
    free_batch(&batch);
    PyErr_SetString(PyExc_MemoryError, "Failed to create result list");
    return NULL;
  }

  for (size_t i = 0; i < batch.count; ++i) {
    PyObject *tuple = PyTuple_New(4);
    if (!tuple) {
      Py_DECREF(result_list);
      free_batch(&batch);
      PyErr_SetString(PyExc_MemoryError, "Failed to create result tuple");
      return NULL;
    }

    PyObject *data_index = PyLong_FromSize_t(batch.items[i].d_idx);
    PyObject *priority = PyFloat_FromDouble(batch.items[i].priority);
    PyObject *importance_weight =
        PyFloat_FromDouble(batch.importance_weights[i]);
    PyObject *item =
        p->tree->data[batch.items[i].d_idx]; // Retrieve the sampled item

    Py_INCREF(item); // Increase reference count for the sampled item

    PyTuple_SetItem(tuple, 0, data_index);
    PyTuple_SetItem(tuple, 1, priority);
    PyTuple_SetItem(tuple, 2, importance_weight);
    PyTuple_SetItem(tuple, 3, item);

    PyList_SetItem(result_list, (Py_ssize_t)i, tuple);
  }

  free_batch(&batch);
  return result_list;
}

// per.update_priorities(handle, td_errors, indices)
static PyObject *py_per_update_priorities(PyObject *self, PyObject *args) {
  (void)self;

  PyObject *capsule;
  PyObject *td_errors_list;
  PyObject *indices_list;

  if (!PyArg_ParseTuple(args, "OOO", &capsule, &indices_list,
                        &td_errors_list)) {
    return NULL;
  }

  PER *p = get_per_from_capsule(capsule);
  if (!p) {
    return NULL;
  }

  size_t td_errors_count = (size_t)PyList_Size(td_errors_list);
  size_t indices_count = (size_t)PyList_Size(indices_list);

  if (td_errors_count != indices_count) {
    PyErr_SetString(PyExc_ValueError,
                    "td_errors and indices lists must have the same length");
    return NULL;
  }

  TD_ERRORS td_errors = {0};
  td_errors.count = td_errors_count;
  td_errors.items = (double *)malloc(td_errors_count * sizeof(double));
  if (!td_errors.items) {
    PyErr_SetString(PyExc_MemoryError,
                    "Failed to allocate memory for td_errors");
    return NULL;
  }

  size_t *priority_indices = (size_t *)malloc(indices_count * sizeof(size_t));
  if (!priority_indices) {
    free(td_errors.items);
    PyErr_SetString(PyExc_MemoryError,
                    "Failed to allocate memory for priority_indices");
    return NULL;
  }

  for (size_t i = 0; i < td_errors_count; ++i) {
    PyObject *td_error_obj = PyList_GetItem(td_errors_list, (Py_ssize_t)i);
    td_errors.items[i] = PyFloat_AsDouble(td_error_obj);

    PyObject *index_obj = PyList_GetItem(indices_list, (Py_ssize_t)i);
    priority_indices[i] = (size_t)PyLong_AsSize_t(index_obj);
  }

  update_per_priorities(p, &td_errors, priority_indices);

  free(td_errors.items);
  free(priority_indices);

  Py_RETURN_NONE;
}

static PyMethodDef PerMethods[] = {
    {"create", (PyCFunction)create_py_per, METH_VARARGS | METH_KEYWORDS,
     "Create a Prioritized Experience Replay (PER) instance."},
    {"size", py_per_size, METH_VARARGS, "Get the current size of the PER."},
    {"total_priority", py_per_total_priority, METH_VARARGS,
     "Get the total priority of the PER."},
    {"add", py_per_add, METH_VARARGS, "Add an item to the PER."},
    {"sample", py_per_sample, METH_VARARGS, "Sample a batch from the PER."},
    {"update_priorities", py_per_update_priorities, METH_VARARGS,
     "Update priorities in the PER."},
    {NULL, NULL, 0, NULL} // Sentinel
};

static struct PyModuleDef permodule = {
    PyModuleDef_HEAD_INIT, "per",
    "Prioritized Experience Replay (C-based, generic Python items via "
    "PyCapsule)",
    -1, PerMethods};

PyMODINIT_FUNC PyInit_per(void) {
  // seed rand() once
  srand((unsigned int)time(NULL));
  return PyModule_Create(&permodule);
}