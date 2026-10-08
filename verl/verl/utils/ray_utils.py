pass

import concurrent.futures

import ray

def parallel_put(data_list, max_workers=None):
    def put_data(index, data):
        return index, ray.put(data)

    if max_workers is None:
        max_workers = min(len(data_list), 16)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        data_list_f = [executor.submit(put_data, i, data) for i, data in enumerate(data_list)]
        res_lst = []
        for future in concurrent.futures.as_completed(data_list_f):
            res_lst.append(future.result())

        output = [None for _ in range(len(data_list))]
        for res in res_lst:
            index, data_ref = res
            output[index] = data_ref

    return output
