from __future__ import print_function

from torchy.utils import timeit
from torchy.pipeline import dask_controller
from torchy.learning import learning


if __name__ == '__main__':


    P = dask_controller.Controller()
    P.process_requests("datasets/sample_req*.json")

    with timeit() as time_info:
        output = P.get_results()
    print(output)
    print("Time Elapsed: {}".format(time_info.elapsed_time))

