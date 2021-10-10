import numpy as np

# coefficients randomly sampled from U(-1, 1)
coeffs = [-0.71347752, -0.6062312 ,  0.66062073, -0.61213017, -0.82631877,
          0.75908621, -0.36274866,  0.84383333, -0.37908235,  0.46539368,
          0.98675292,  0.20388434, -0.78718797, -0.74316581,  0.11539548,
          0.35559215,  0.32584524,  0.58590426,  0.95284985, -0.77999937,
          -0.660806  , -0.35542737,  0.93532822,  0.75049373, -0.27825577]

def search(x, y, minimize=True):
    # x - (num_points x (length - 1))
    # y - (num_points)

    y = np.reshape(y, [-1,1])

    deltas = np.abs(x - y) # (num_points x (length - 1))
    if minimize:
        indices = np.argmin(deltas, 1)
    else:
        indices = np.argmax(deltas, 1)

    return indices

def search_by_indices(data, time, index, minimize=True):
    curr = data[:, time, index]
    search_data = data[:,:,index].copy()
    search_data[:, time] = float('inf')

    winners = search(search_data, curr, minimize)

    return winners

def retrieve(data, winners, index):
    return data[range(data.shape[0]), winners, index]

def sum(a, b):
    return a+b

def subtract(a,b):
    return a-b

def max(a, b):
    return np.maximum(a,b)

def rule(data, time, search1, search2, retrieve1, retrieve2, func):
    s1 = search_by_indices(data, time, search1)
    out1 = retrieve(data, s1, retrieve1)

    s2 = search_by_indices(data, time, search2)
    out2 = retrieve(data, s2, retrieve2)

    return func(out1, out2), s1, s2

def onehot(task, num_points, length, v_s, v_p):
    task_onehot = np.zeros((task.size, v_p))
    task_onehot[np.arange(task.size), task] = 1.
    task_onehot = np.reshape(task_onehot, (num_points, length, v_s, v_p))
    return task_onehot

def dataset(num_points, length, v_s, v_p, cff=False):
    data = np.random.randn(num_points, length, v_p+v_s)
    task = np.random.choice(v_p, [num_points * length * v_s])
    coeff = np.reshape(coeffs[:v_s], [1,1,v_s])

    task_onehot = onehot(task, num_points, length, v_s, v_p)

    searches = np.zeros([num_points, length, length])
    retrievals = np.zeros([num_points, length, v_s, v_p])

    for i in range(length):
        for search in range(v_s):
            s = search_by_indices(data, i, search)
            searches[range(data.shape[0]), i, s] = 1.
            for r in range(v_p):
                retrievals[:,i,search,r] = retrieve(data, s, v_s + r)

    if cff:
        labels = np.sum(coeff * np.sum(task_onehot * retrievals, axis=-1), axis=-1)
    else:
        labels = np.sum(np.sum(task_onehot * retrievals, axis=-1), axis=-1)

    inp = np.concatenate((data, np.reshape(task_onehot, (num_points, length, v_s * v_p))), axis=-1)

    return inp, labels, searches

def dataset_ood(num_points, length, v_s, v_p, all_v_combs, cff=False, train=True):
    data = np.random.randn(num_points, length, v_p + v_s)
    coeff = np.reshape(coeffs[:v_s], [1, 1, v_s])
    train_split_frac = 0.8
    train_split = int(len(all_v_combs) * train_split_frac)

    train_combs = [all_v_combs[i] for i in range(len(all_v_combs)) if i % 2 == 0]
    test_combs = [all_v_combs[i] for i in range(len(all_v_combs)) if i % 2 != 0]

    if len(train_combs) < train_split:
        res = train_split - len(train_combs)
        train_combs.extend(test_combs[:res])
        train_combs = np.asarray(train_combs)
        test_combs = np.asarray(test_combs[res:])

    train_combs = train_combs[np.random.choice(train_combs.shape[0], num_points * length, replace=True)]
    test_combs = test_combs[np.random.choice(test_combs.shape[0], num_points * length, replace=True)]
    train_combs = train_combs.reshape(-1)
    test_combs = test_combs.reshape(-1)

    searches = np.zeros([num_points, length, length])
    retrievals = np.zeros([num_points, length, v_s, v_p])

    for i in range(length):
        for search in range(v_s):
            s = search_by_indices(data, i, search)
            searches[range(data.shape[0]), i, s] = 1.
            for r in range(v_p):
                retrievals[:, i, search, r] = retrieve(data, s, v_s + r)

    if train:
        task_onehot = onehot(train_combs, num_points, length, v_s, v_p)
        inp = np.concatenate((data, np.reshape(task_onehot, (num_points, length, v_s * v_p))), axis=-1)
        if cff:
            labels = np.sum(coeff * np.sum(task_onehot * retrievals, axis=-1), axis=-1)
        else:
            labels = np.sum(np.sum(task_onehot * retrievals, axis=-1), axis=-1)
        return inp, labels, searches
    else:
        task_onehot_test = onehot(test_combs, num_points, length, v_s, v_p)
        test_inp = np.concatenate((data, np.reshape(task_onehot_test, (num_points, length, v_s * v_p))), axis=-1)
        if cff:
            test_labels = np.sum(coeff * np.sum(task_onehot_test * retrievals, axis=-1), axis=-1)
        else:
            test_labels = np.sum(np.sum(task_onehot_test * retrievals, axis=-1), axis=-1)
        return test_inp, test_labels, searches